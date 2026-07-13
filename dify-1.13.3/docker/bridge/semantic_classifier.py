import json
import logging
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed, wait, FIRST_COMPLETED
from difflib import SequenceMatcher
from typing import Generator

import requests

from preprocessing import estimate_asr_noise_level

logger = logging.getLogger("classifier")

NOISE_DETECT_PROMPT = """你是军事语音通信噪声过滤器。你的唯一任务是判断给定文本是否为非军事噪声。

【判定为噪声】满足以下任一条件：
1. 日常闲聊：吃饭、天气、钓鱼、娱乐、八卦、家庭琐事等非工作话题
2. 军事无关的工作对话：行政通知、后勤杂务等不含战术/作战内容的对话
3. 无意义碎片：无法辨识的断句、不构成完整语义的字符堆砌

【判定为军事通信】即使包含ASR转录错误也判为军事：
- 包含作战、战术、侦察、火力、部署、敌情等军事行动内容
- 电台通联、信号检查、频率调整等通信保障对话
- 态势汇报、弹药/物资统计、人员状态报告

【关键原则 - 务必遵守】
- "上网"写成"伤亡"、"剩鱼"写成"剩余"、"花现"写成"发现"、"指挥锁"写成"指挥所" "蜜令"写成"命令" → 这些是ASR同音错误，不影响判定！只要上下文是军事行动，就是军事通信。
- 如果一句话同时包含军事词汇和日常用语（如"带上装备去钓鱼"），根据核心意图判断：该句的主要目的是军事行动还是日常活动？
- 不确定时优先判定为军事通信（宁可误判不漏判）

原文：{text}
算法噪声分：{noise_score}（规则引擎评分，>0.3表示可能有ASR噪声特征）
关键词命中：{keyword_hits}

只输出JSON，不要添加任何其他内容：
{"is_noise": false, "confidence": 0.95, "reason": "判断理由"}"""

CLASSIFY_PROMPT = """你是军事语音通信文本分析专家。请完成两步任务。

【第一步：ASR纠错】
语音转文字常见错误类型及纠正示例：
- 同音替换：上网→伤亡、剩鱼→剩余、花现→发现、井报→警报、完笔→完毕、蜜令→命令、守留→手榴、但→弹
- 声调错误：但要→弹药、移似→疑似、真查→侦察、秋火→请求火力、负盖→覆盖
- 字符丢失：无机→无人机、急需求充→急需补充、方按→方案、滴火立点→敌火力点
- 字符错位/多余：我不已倒达→我部已到达、雷留→手榴、力之源→支援
- 军事术语恢复：指挥锁→指挥所、东么→洞幺、佐一→左翼、真地→阵地、上网→伤亡
注意：根据上下文推断正确词汇，不要过度纠正，保留明确的军事格式和代号。

【第二步：军事通信分类】
类别及核心判定特征：
- 指令：传达战斗指令。特征词"命令/执行/进攻/撤退/立即/按N号方案/总攻/突击/完毕"
- 情报：汇报侦察信息。特征词"侦察/发现/方位/坐标/距离/数量/疑似/方向/报告"
- 态势：汇报当前状态。特征词"报告/到达/伤亡/弹药剩余/进展/稳定/正常/异常/指示"
- 火力：火力支援相关。特征词"炮火/压制/支援/覆盖/效力射/射击/开火/摧毁/弹着"
- 补给：后勤物资请求。特征词"急需/补充/后送/弹药/油料/药品/伤员/前送/剩余"
- 通信：电台通联内容。特征词"信号/频率/通联/听不清/切换/备用/完毕/再说一遍"
- 敌情：敌方威胁信息。特征词"敌军/来袭/警报/无人机/伏击/渗透/坦克/装甲车/防空"
- 计划：作战方案规划。特征词"方案/部署/梯队/协同/佯攻/主攻/突破口/集结/计划"
注意：按核心意图选择最匹配的单一类别。多类别词汇同时出现时，选占比最高或最具决定性的那个。

原文：{text}
参考关键词：{keywords}

输出严格JSON，不要添加任何其他内容：
{"preprocessed_text":"纠错后完整文本","label":"类别","confidence":0.95,"evidence":["依据1","依据2"]}"""

EXTRACT_PROMPT = """你是战场通信情报分析专家。以下是标记为「{labels}」的通信记录，请提取关键要素并组织为结构化摘要。

通信记录：
{filtered_lines}

核心原则：少条目、精内容，宁可合并同类项也不要拆出过多碎片。

要求：
1. 按"{labels}类关键要素"为标题输出Markdown
2. 同类内容强制合并为一段，每类最终不超过6条
3. 合并规则（必须遵守）：
   - 所有警戒/隐蔽类命令合并为一段
   - 所有开火/火力打击类命令合并为一段
   - 所有推进/包抄/追击类命令合并为一段
   - 所有休整/收队/结束类命令合并为一段
   - 同一作战阶段的命令合并在一起
   - 彼此无关联的独立指令才分开
4. 每段以"● "开头，段末标注涉及行号，如 [行1][行3][行8]
5. 不添加原文中没有的信息，不推断、不补充
6. 保留数字、代号、单位等精确值

示例输出：
## 指令类关键要素

● 攻击部署：一梯队正面突击，二梯队左翼迂回。 [行1]
● 开火与火力：对坐标066-142急速射；首群命中后效力射。 [行3][行8]
● 突破与推进：一梯队突破防线继续推进，二梯队配合夹击。 [行4][行5]"""

VERIFY_PROMPT = """你是战场通信情报审核专家。以下是初步提取结果和全文，请核对并输出最终的完整提取结果。

初步提取：
{extracted_summary}

通信全文：
{all_lines}

要求：
1. 保留初步提取中所有正确内容
2. 逐条对照全文，补充任何遗漏的关键信息（时间/地点/兵力/装备/行动/伤亡/补给/敌情），标注来源行号
3. 按类别分节输出 Markdown，格式：## 类别名 关键要素
4. 只输出与「{labels}」相关的内容，不要添加情报/火力/补给/通信/敌情/计划/态势等其他类别信息
5. 只输出最终的完整提取结果，不要输出检查过程、遗漏清单、对比表格或"检验通过"等过程性文字"""


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
        ollama_url: str = "http://ollama:11434",
        ollama_model: str = "qwen3:8b",
    ):
        self.dify_api_url = dify_api_url
        self.dify_api_key = dify_api_key
        self.keyword_categories = keyword_categories
        self.llm_mode = llm_mode
        self.api_base = api_base.rstrip("/") if api_base else ""
        self.api_key = api_key
        self.api_model = api_model
        self.ollama_url = ollama_url.rstrip("/") if ollama_url else "http://ollama:11434"
        self.ollama_model = ollama_model or "qwen3:8b"
        all_keywords: list[str] = []
        self._kw_to_category: dict[str, str] = {}
        for cat, kws in keyword_categories.items():
            for kw in kws:
                self._kw_to_category[kw] = cat
                all_keywords.append(kw)
        self.keywords = all_keywords
        self.matcher = KeywordMatcher(all_keywords)
        self._max_workers = 10 if llm_mode == "api" else 1
        self._progress: dict = {}
        self._lock = threading.Lock()
        self._cancelled_tasks: set = set()

    def cancel(self, task_id: str) -> None:
        self._cancelled_tasks.add(task_id)

    @staticmethod
    def _load_correction_examples(limit: int = 10) -> str:
        """从 corrections.jsonl 加载人工纠正记录作为 few-shot"""
        corrections_file = os.path.join(os.path.dirname(__file__), "corrections.jsonl")
        try:
            with open(corrections_file, "r", encoding="utf-8") as f:
                lines = f.readlines()
            corrections = [json.loads(l) for l in lines[-limit:] if l.strip()]
        except (FileNotFoundError, json.JSONDecodeError):
            return ""
        if not corrections:
            return ""
        parts = ["【人工纠正记录 — 请注意：以下为已确认的正确分类，遇到相似文本时优先参考】"]
        for c in corrections:
            parts.append(
                f"- 原文: \"{c['original_text'][:60]}\" → 正确: {c['correct_label']}"
                f"（LLM曾误判{c.get('llm_label','?')}）"
            )
        return "\n".join(parts)

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

        executor = ThreadPoolExecutor(max_workers=self._max_workers)
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
        """算法预筛 → 噪声门LLM → 关键词匹配 → 军事分类LLM → 比对"""
        n = idx + 1
        text = line.strip()
        logger.info("[语音文本%d] %s", n, "─" * 30)
        logger.info("[语音文本%d] 原文: %s", n, text[:100])

        # 阶段0：算法预筛
        noise_info = estimate_asr_noise_level(text)
        kw_result_raw = self.matcher.match(text)
        noise_score = noise_info["score"]
        kw_hits_raw = kw_result_raw.get("hits", [])
        kw_score_raw = kw_result_raw.get("score", 0.0)
        is_likely_noisy = noise_info.get("is_likely_noisy", False)

        logger.info("[语音文本%d] 预筛: noise=%.2f noisy=%s kw_hits=%d kw_score=%.2f",
            n, noise_score, is_likely_noisy, len(kw_hits_raw), kw_score_raw)

        # 预筛决策：是否可以跳过噪声门（仅强确信跳过，避免"装备"等泛词误判）
        skip_noise_gate = (
            not is_likely_noisy
            and kw_score_raw >= 0.9
            and len(kw_hits_raw) >= 5
        )

        is_noise = False
        noise_confidence = 0.0
        noise_reason = ""

        if not skip_noise_gate:
            # 阶段1：噪声门LLM（二分类）
            noise_result = self._call_noise_gate(text, noise_score, kw_hits_raw)
            is_noise = noise_result.get("is_noise", False)
            noise_confidence = noise_result.get("confidence", 0.0)
            noise_reason = noise_result.get("reason", "")
            logger.info("[语音文本%d] 噪声门: is_noise=%s confidence=%.2f reason=%s",
                n, is_noise, noise_confidence, noise_reason)
        else:
            logger.info("[语音文本%d] 跳过噪声门（算法确信军事通信）", n)

        if is_noise:
            # 确认为噪声，直接返回
            logger.info("[语音文本%d] 🚫 判定为噪声 | reason=%s", n, noise_reason)
            return self._build_noise_result(idx, text, noise_confidence, noise_reason)

        # 阶段2：军事分类LLM（纠错+8分类）
        llm_result = self._call_military_llm(text)
        preprocessed = llm_result.get("preprocessed_text", text)
        llm_label = llm_result.get("label", "解析失败")
        llm_confidence = round(float(llm_result.get("confidence", 0.0)), 3)
        llm_evidence = llm_result.get("evidence", [])[:5]

        logger.info("[语音文本%d] 预处理: %s", n, preprocessed[:100])
        logger.info("[语音文本%d] LLM: %s (%.2f) | 依据: %s",
            n, llm_label, llm_confidence, ", ".join(llm_evidence[:2]))

        # 阶段3：关键词匹配（基于预处理后文本）
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

        # 阶段4：比对冲突裁决（LLM高置信度优先，KW做辅助校验）
        labels_match = kw_label == llm_label
        llm_conf = llm_confidence
        kw_conf = kw_confidence

        if labels_match:
            final_label = kw_label
            final_conf = max(kw_conf, llm_conf)
            final_evidence = llm_evidence if llm_conf >= kw_conf else kw_evidence
            final_status = "verified"
        elif llm_conf >= 0.9 and llm_label != "解析失败" and llm_label != "分类失败":
            # LLM高置信度，以LLM为准
            final_label = llm_label
            final_conf = llm_conf
            final_evidence = llm_evidence
            final_status = "verified"
            logger.info("[语音文本%d] 🔧 裁决: LLM置信度=%.2f，以LLM为准 | kw=%s -> llm=%s",
                n, llm_conf, kw_label, llm_label)
        elif llm_conf < 0.5 and kw_conf >= 0.9:
            # LLM不确定，KW强匹配，以KW为准
            final_label = kw_label
            final_conf = kw_conf
            final_evidence = kw_evidence
            final_status = "verified"
            logger.info("[语音文本%d] 🔧 裁决: LLM置信度低(%.2f)，以KW为准 | kw=%s -> llm=%s",
                n, llm_conf, kw_label, llm_label)
        else:
            # 双方都有一定置信度但分歧，标记为冲突
            final_label = "待处理"
            final_conf = 0.0
            final_evidence = []
            final_status = "conflict"
            logger.info("[语音文本%d] ⚠ 冲突 | kw=%s(%.2f) llm=%s(%.2f)",
                n, kw_label, kw_conf, llm_label, llm_conf)

        return {
            "line_no": idx + 1,
            "original": text,
            "preprocessed": preprocessed,
            "kw_label": kw_label,
            "kw_confidence": kw_conf,
            "kw_evidence": kw_evidence,
            "llm_label": llm_label,
            "llm_confidence": llm_conf,
            "llm_evidence": llm_evidence,
            "label": final_label,
            "confidence": final_conf,
            "evidence": final_evidence,
            "status": final_status,
            "corrected": False,
        }

    def _call_military_llm(self, text: str) -> dict:
        """军事分类LLM路由（纠错+8分类），根据 llm_mode 路由到 Dify 或 OpenAI API"""
        if self.llm_mode == "api" and self.api_base and self.api_key:
            return self._call_openai_military(text)
        return self._call_dify_classify(text)

    def _call_noise_gate(self, text: str, noise_score: float, kw_hits: list[str]) -> dict:
        """噪声门：二分类判断是否为非军事噪声"""
        if self.llm_mode == "api" and self.api_base and self.api_key:
            return self._call_openai_noise(text, noise_score, kw_hits)

        prompt = NOISE_DETECT_PROMPT.replace("{text}", text).replace(
            "{noise_score}", str(round(noise_score, 3))
        ).replace("{keyword_hits}", ", ".join(kw_hits[:10]) if kw_hits else "无")
        result = self._call_ollama(prompt)
        return {
            "is_noise": result.get("is_noise", False),
            "confidence": result.get("confidence", 0.0),
            "reason": result.get("reason", ""),
        }

    def _call_ollama(self, prompt: str) -> dict:
        """调用本地 Ollama API，解析 JSON 返回"""
        try:
            resp = requests.post(
                f"{self.ollama_url}/api/generate",
                headers={"Content-Type": "application/json"},
                json={
                    "model": self.ollama_model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.3},
                },
                timeout=120,
            )
            resp.raise_for_status()
            content = resp.json().get("response", "")
            m = re.search(r'\{[\s\S]*\}', content)
            if m:
                return json.loads(m.group())
            return {"label": "解析失败", "confidence": 0.0, "evidence": [content[:100]]}
        except Exception as exc:
            logger.error("Ollama API error: %s", exc)
            return {"label": "分类失败", "confidence": 0.0, "evidence": [str(exc)[:100]]}

    def _call_openai_military(self, text: str) -> dict:
        """调用 OpenAI 兼容 API 进行军事分类（纠错+8分类）"""
        prompt = CLASSIFY_PROMPT.replace("{text}", text).replace("{keywords}", ", ".join(self.keywords[:40]))
        return self._call_openai_raw(prompt)

    def _call_openai_noise(self, text: str, noise_score: float, kw_hits: list[str]) -> dict:
        """调用 OpenAI 兼容 API 进行噪声门二分类"""
        prompt = NOISE_DETECT_PROMPT.replace("{text}", text).replace(
            "{noise_score}", str(round(noise_score, 3))
        ).replace("{keyword_hits}", ", ".join(kw_hits[:10]) if kw_hits else "无")
        result = self._call_openai_raw(prompt)
        return {
            "is_noise": result.get("is_noise", False),
            "confidence": result.get("confidence", 0.0),
            "reason": result.get("reason", ""),
        }

    def _call_openai_raw(self, prompt: str) -> dict:
        """通用 OpenAI 兼容 API 调用，解析 JSON 返回"""
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
        """调用Dify workflow进行军事分类（纠错+分类）"""
        try:
            # 构建分类词典 JSON（每类取前 5 个核心关键词）
            cat_map = {}
            for cat, kws in self.keyword_categories.items():
                filtered = [kw for kw in kws if kw in self.keywords[:40]][:5]
                if filtered:
                    cat_map[cat] = filtered
            categories_json = json.dumps(cat_map, ensure_ascii=False)
            # 加载人工纠正记录作为 few-shot 示例
            correction_examples = self._load_correction_examples()
            headers = {"Authorization": f"Bearer {self.dify_api_key}", "Content-Type": "application/json"}
            payload = {
                "inputs": {
                    "original_text": text,
                    "keywords": ", ".join(self.keywords[:40]),
                    "keyword_categories": categories_json,
                    "correction_examples": correction_examples,
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

    @staticmethod
    def _build_noise_result(idx: int, line: str, confidence: float = 0.0, reason: str = "") -> dict:
        return {
            "line_no": idx + 1,
            "original": line,
            "preprocessed": line,
            "kw_label": "噪声", "kw_confidence": 0.0, "kw_evidence": [],
            "llm_label": "噪声", "llm_confidence": confidence, "llm_evidence": [reason] if reason else [],
            "label": "噪声", "confidence": confidence, "evidence": [reason] if reason else [],
            "status": "verified", "corrected": False,
            "noise_gate": True,
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
        model_info = f"API ({self.api_model})" if self.llm_mode == "api" else "本地 Ollama (qwen3:8b)"
        parts.append(f"- 使用模型：{model_info}\n")
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

    @staticmethod
    def generate_preprocessed_txt(results: list[dict]) -> str:
        """生成预处理后的纯文本：去掉噪声行，取纠错文本，按行号排序。"""
        valid = [r for r in results
                 if r.get("label") not in ("噪声", "空行", "分类失败", "解析失败")
                 and not r.get("noise_gate")]
        valid.sort(key=lambda r: r.get("line_no", 0))
        lines = [(r.get("preprocessed") or r.get("original", "")) + f"（{r.get('label', '')}）" for r in valid]
        return "\n".join(lines)
