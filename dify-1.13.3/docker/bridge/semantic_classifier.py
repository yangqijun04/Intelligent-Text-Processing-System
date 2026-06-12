import json
import logging
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed, wait, FIRST_COMPLETED
from difflib import SequenceMatcher
from typing import Generator

import requests

logger = logging.getLogger("classifier")

CLASSIFY_PROMPT = """你是军事语音通信文本分析专家。请完成两步任务：

第一步：纠错预处理
对语音转文字结果纠错：同音替换、字符丢失、军用术语误识别、标点缺失、字符重复

第二步：分类
对纠错后文本分类，类别包括：
- 指令：作战命令、行动指令、指挥调度
- 情报：侦察获取的敌情、地形、兵力部署情报
- 态势：部队状态、战况进展、阵地情况汇报
- 火力：火力请求、炮击支援、空中打击请求
- 补给：弹药、油料、食品、医疗后勤请求
- 通信：无线电台通联、信号检查、频率调整
- 敌情：发现敌军动向、威胁预警、来袭警报
- 计划：作战方案、兵力部署、行动路线规划
- 噪声：非军事对话、环境噪音、无效语音

原文：{text}
参考关键词：{keywords}

输出严格JSON，不要添加任何其他内容：
{"preprocessed_text":"纠错后完整文本","label":"类别","confidence":0.95,"evidence":["依据"]}"""


class KeywordMatcher:
    """涉军关键词模糊匹配引擎，容忍语音转文字误差"""

    def __init__(self, keywords: list[str]):
        self.keywords: list[str] = [k.strip() for k in keywords if k.strip()]
        self._lower_keywords: list[str] = [k.lower() for k in self.keywords]

    @property
    def count(self) -> int:
        return len(self.keywords)

    def match(self, text: str) -> dict:
        """
        对文本执行关键词匹配。

        Returns:
            dict with keys:
                matched (bool): 是否有命中
                hits (list[str]): 命中的关键词
                score (float): 最高匹配分数 (0.0-1.0)
                is_fuzzy (bool): 是否为模糊匹配（非精确）
        """
        if not self.keywords or not text:
            return {"matched": False, "hits": [], "score": 0.0, "is_fuzzy": False}

        text_lower = text.lower()
        hits: list[str] = []
        best_score = 0.0
        is_fuzzy = False

        # 第1轮：精确匹配（最快）
        for i, kw in enumerate(self._lower_keywords):
            if kw in text_lower:
                hits.append(self.keywords[i])
                best_score = 1.0

        if hits:
            return {"matched": True, "hits": hits, "score": 1.0, "is_fuzzy": False}

        # 第2轮：模糊匹配（使用 SequenceMatcher）
        # 对长文本取滑动窗口与关键词比较
        FUZZY_MIN = 0.78  # 同尺寸窗口最小阈值
        FUZZY_DIFF_SIZE_MIN = 0.88  # 不同尺寸窗口最小阈值（更严格）
        
        for i, kw in enumerate(self.keywords):
            kw_len = len(kw)
            if kw_len < 2:
                continue
            text_len = len(text)
            if text_len < kw_len:
                score = SequenceMatcher(None, kw, text).ratio()
            else:
                max_score = 0.0
                step = max(1, kw_len // 2)
                for start in range(0, text_len - kw_len + 1, step):
                    window = text[start:start + kw_len]
                    score = SequenceMatcher(None, kw, window).ratio()
                    if score > max_score:
                        max_score = score
                score = max_score
                # 额外检查部分匹配（使用更严格的阈值）
                if score < FUZZY_MIN:
                    for size in (kw_len - 1, kw_len + 1, kw_len - 2):
                        if size < 2:
                            continue
                        for start in range(0, max(1, text_len - size + 1), step):
                            window = text[start:start + size]
                            s = SequenceMatcher(None, kw, window).ratio()
                            if s > score:
                                score = s

            if score >= FUZZY_MIN:
                hits.append(self.keywords[i])
                if score > best_score:
                    best_score = score
                is_fuzzy = True
                # 不同尺寸窗口匹配降低置信度
                if max_score < FUZZY_DIFF_SIZE_MIN:
                    score = score * 0.85  # 惩罚分

        return {
            "matched": len(hits) > 0,
            "hits": hits,
            "score": best_score,
            "is_fuzzy": is_fuzzy,
        }


class BatchClassifier:
    """批量分类管理器，协调关键词匹配和Dify LLM语义分类"""

    def __init__(
        self,
        dify_api_url: str,
        dify_api_key: str,
        keyword_categories: dict[str, list[str]],
        llm_mode: str = "local",
        api_base: str = "",
        api_key: str = "",
        api_model: str = "qwen-plus",
    ):
        self.dify_api_url = dify_api_url
        self.dify_api_key = dify_api_key
        self.keyword_categories = keyword_categories
        self.llm_mode = llm_mode
        self.api_base = api_base.rstrip("/") if api_base else ""
        self.api_key = api_key
        self.api_model = api_model
        all_keywords: list[str] = []
        self._kw_to_category: dict[str, str] = {}
        for cat, kws in keyword_categories.items():
            for kw in kws:
                self._kw_to_category[kw] = cat
                all_keywords.append(kw)
        self.keywords = all_keywords
        self.matcher = KeywordMatcher(all_keywords)
        self._progress: dict = {}
        self._lock = threading.Lock()
        self._cancelled_tasks: set = set()

    def cancel(self, task_id: str) -> None:
        self._cancelled_tasks.add(task_id)

    # 战场通信类别标签
    CATEGORY_LABELS: dict[str, str] = {
        "指令": "指令",
        "情报": "情报",
        "态势": "态势",
        "火力": "火力",
        "补给": "补给",
        "通信": "通信",
        "敌情": "敌情",
        "计划": "计划",
        "噪声": "噪声",
    }

    def _resolve_category(self, hits: list[str]) -> tuple[str, bool]:
        """根据命中的关键词判断最可能的类别"""
        if not hits:
            return ("噪声", False)
        cat_scores: dict[str, int] = {}
        for kw in hits:
            cat = self._kw_to_category.get(kw, "")
            if cat:
                cat_scores[cat] = cat_scores.get(cat, 0) + 1
        if not cat_scores:
            return ("噪声", False)
        # 返回得分最高的类别
        best = max(cat_scores, key=lambda k: cat_scores[k])
        # 如果最高分只有1分且仅命中1个关键词，标记为模糊
        single_weak = cat_scores[best] == 1 and len(hits) == 1
        return (best, single_weak)

    def _process_single_line(self, idx: int, line: str, total: int) -> dict:
        """处理单条文本的完整流程（供线程池调用）"""
        if not line.strip():
            return self._empty_result(idx, line)
        return self._classify_line(idx, line)

    def classify_lines(self, lines: list[str], task_id: str) -> Generator[dict, None, None]:
        """10线程并行：预处理 → 关键词+LLM分类 → 比较"""
        total = len(lines)
        results: list[dict] = []
        self._progress[task_id] = {"completed": 0, "total": total, "results": results}

        executor = ThreadPoolExecutor(max_workers=10)
        try:
            futures_dict = {}
            for idx, line in enumerate(lines):
                futures_dict[executor.submit(self._process_single_line, idx, line, total)] = idx

            pending = set(futures_dict.keys())
            while pending:
                if task_id in self._cancelled_tasks:
                    logger.info("task %s: 收到取消信号，已处理 %d/%d 条，丢弃 %d 条待处理",
                        task_id, len(results), total, len(pending))
                    break
                done, pending = wait(pending, timeout=1, return_when=FIRST_COMPLETED)
                for future in done:
                    result = future.result()
                    with self._lock:
                        results.append(result)
                        self._progress[task_id] = {"completed": len(results), "total": total, "results": list(results)}
                    yield {"line_no": result["line_no"], "total": total, "result": result}
        finally:
            executor.shutdown(wait=False)

    def _classify_line(self, idx: int, line: str) -> dict:
        """Dify LLM 纠错+分类一次完成 → 关键词匹配预处理文本 → 比对"""
        n = idx + 1
        logger.info("[语音文本%d] %s", n, "─" * 30)
        logger.info("[语音文本%d] 原文: %s", n, line.strip()[:100])

        # 第1步：Dify 纠错+分类一次完成
        llm_result = self._call_llm(line)
        preprocessed = llm_result.get("preprocessed_text", line.strip())
        llm_label = llm_result.get("label", "解析失败")
        llm_confidence = round(float(llm_result.get("confidence", 0.0)), 3)
        llm_evidence = llm_result.get("evidence", [])[:5]

        logger.info("[语音文本%d] 预处理: %s", n, preprocessed[:100])
        logger.info("[语音文本%d] LLM: %s (%.2f) | 依据: %s",
            n, llm_label, llm_confidence, ", ".join(llm_evidence[:2]))

        # 第2步：关键词匹配（基于预处理后文本）
        kw_result = self.matcher.match(preprocessed)
        cat, weak = self._resolve_category(kw_result["hits"]) if kw_result["matched"] else ("噪声", True)
        kw_conf = kw_result["score"]
        if kw_result.get("is_fuzzy"):
            kw_conf *= 0.85
        if weak:
            kw_conf *= 0.9
        kw_evidence = kw_result.get("hits", [])[:5]
        kw_label = cat
        kw_confidence = round(kw_conf, 3)
        logger.info("[语音文本%d] 关键词: %s (%.2f) | 命中: %s",
            n, kw_label, kw_confidence, ", ".join(kw_evidence[:3]))

        # 第3步：比对
        labels_match = kw_label == llm_label
        if labels_match:
            logger.info("[语音文本%d] ✅ 一致通过 | label=%s", n, kw_label)
        else:
            logger.info("[语音文本%d] ⚠ 冲突 | kw=%s llm=%s", n, kw_label, llm_label)

        return {
            "line_no": idx + 1,
            "original": line.strip(),
            "preprocessed": preprocessed,
            "kw_label": kw_label,
            "kw_confidence": kw_confidence,
            "kw_evidence": kw_evidence,
            "llm_label": llm_label,
            "llm_confidence": llm_confidence,
            "llm_evidence": llm_evidence,
            "label": kw_label if labels_match else "待处理",
            "confidence": kw_confidence if labels_match else 0.0,
            "evidence": kw_evidence if labels_match else [],
            "status": "verified" if labels_match else "conflict",
            "corrected": False,
        }

    def _call_llm(self, text: str) -> dict:
        """根据 llm_mode 路由到 Dify 或 OpenAI API"""
        if self.llm_mode == "api" and self.api_base and self.api_key:
            return self._call_openai(text)
        return self._call_dify_classify(text)

    def _call_openai(self, text: str) -> dict:
        """调用 OpenAI 兼容 API（纠错+分类一次完成）"""
        prompt = CLASSIFY_PROMPT.replace("{text}", text).replace("{keywords}", ", ".join(self.keywords[:40]))
        try:
            resp = requests.post(
                f"{self.api_base}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.api_model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.3,
                },
                timeout=120,
            )
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"]
            m = re.search(r'\{[\s\S]*\}', content)
            if m:
                return json.loads(m.group())
            return {"label": "解析失败", "confidence": 0.0, "evidence": [content[:100]]}
        except Exception as exc:
            logger.error("OpenAI API error: %s", exc)
            return {"label": "分类失败", "confidence": 0.0, "evidence": [str(exc)[:100]]}

    def _call_dify_classify(self, text: str) -> dict:
        """调用Dify workflow进行LLM纯分类（无预处理）"""
        try:
            headers = {"Authorization": f"Bearer {self.dify_api_key}", "Content-Type": "application/json"}
            payload = {
                "inputs": {
                    "original_text": text,
                    "keywords": ", ".join(self.keywords[:40]),
                },
                "response_mode": "blocking",
                "user": "battlefield-classifier",
            }
            resp = requests.post(f"{self.dify_api_url}/workflows/run", headers=headers, json=payload, timeout=120)
            resp.raise_for_status()
            data = resp.json()
            outputs = data.get("data", {}).get("outputs", {})
            raw_result = outputs.get("classification_result", "{}")
            return json.loads(raw_result)
        except Exception as exc:
            logger.error("Dify classify error: %s", exc)
            return {"label": "分类失败", "confidence": 0.0, "evidence": [str(exc)[:100]]}

    @staticmethod
    def _empty_result(idx: int, line: str) -> dict:
        return {
            "line_no": idx + 1,
            "original": line,
            "preprocessed": "",
            "kw_label": "空行", "kw_confidence": 0.0, "kw_evidence": [],
            "llm_label": "空行", "llm_confidence": 0.0, "llm_evidence": [],
            "label": "空行", "confidence": 0.0, "evidence": [],
            "status": "verified", "corrected": False,
        }

    def get_progress(self, task_id: str) -> dict:
        with self._lock:
            return self._progress.get(task_id, {"completed": 0, "total": 0, "results": []})

    def generate_report(self, results: list[dict], original_filename: str = "") -> str:
        """生成Markdown格式的战场通信分类报告"""
        total = len(results)
        effective = [r for r in results if r["label"] != "空行"]
        failed = [r for r in effective if r["label"] in ("分类失败", "解析失败", "噪声")]
        categorized: dict[str, list[dict]] = {}
        for r in effective:
            cat = r.get("label", "未分类")
            categorized.setdefault(cat, []).append(r)

        now = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        parts: list[str] = []
        parts.append(f"## 战场通信文本分类报告\n\n")
        parts.append(f"- 处理时间：{now}\n")
        parts.append(f"- 数据来源：{original_filename}\n")
        parts.append(f"- 分类模式：关键词 + LLM 并行分类，结果比对\n")
        parts.append(f"- 总条目：{total}\n")
        parts.append(f"- 一致通过：{len([r for r in results if r.get('status') == 'verified' and r.get('label') != '空行'])} 条\n")
        parts.append(f"- 冲突待处理：{len([r for r in results if r.get('status') == 'conflict'])} 条\n\n")

        category_order = ["指令", "情报", "态势", "火力", "补给", "通信", "敌情", "计划", "噪声"]
        for cat in category_order:
            items = categorized.get(cat, [])
            if not items:
                continue
            high = len([r for r in items if r["confidence"] >= 0.8])
            mid = len([r for r in items if 0.5 <= r["confidence"] < 0.8])
            low = len([r for r in items if r["confidence"] < 0.5])
            parts.append(f"### {cat}（{len(items)} 条，高{high} 中{mid} 低{low}）\n\n")
            for r in items[:30]:
                orig = r["original"].replace("|", "｜")[:50]
                ev = ", ".join(r.get("evidence", [])[:3]).replace("|", "｜")
                parts.append(f"- [{r['line_no']}] [{r['confidence']:.2f}] {orig} | {ev}\n")
            if len(items) > 30:
                parts.append(f"\n*(共{len(items)}条，仅展示前30条)*\n")
            parts.append("\n")

        # 冲突章节
        conflicts = [r for r in results if r.get("status") == "conflict"]
        if conflicts:
            parts.append(f"### ⚠ 分类冲突 — 需人工处理（{len(conflicts)} 条）\n\n")
            parts.append("| 序号 | 原文 | KW分类 | LLM分类 |\n")
            parts.append("|------|------|--------|--------|\n")
            for r in conflicts[:30]:
                orig = r["original"].replace("|", "｜")[:40]
                parts.append(f"| {r['line_no']} | {orig} | {r.get('kw_label','?')} | {r.get('llm_label','?')} |\n")
            if len(conflicts) > 30:
                parts.append(f"\n*(共{len(conflicts)}条冲突，仅展示前30条)*\n")
            parts.append("\n")

        return "".join(parts)
