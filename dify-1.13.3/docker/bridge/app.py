import json
import logging
import os
import re
import threading
import time
import uuid
from pathlib import Path

import requests
from flask import Flask, jsonify, redirect, request, send_from_directory
from flask import Response as FlaskResponse

import db
from preprocessing import clean_text, estimate_asr_noise_level, split_text_to_lines
from semantic_classifier import BatchClassifier, KeywordMatcher
from semantic_classifier import EXTRACT_PROMPT, VERIFY_PROMPT, CLASSIFY_PROMPT as _CPT

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DIFY_API_URL = os.environ.get("DIFY_API_URL", "http://api:5001/v1")
DIFY_API_KEY = os.environ.get("DIFY_API_KEY", "")
DIFY_CLASSIFY_API_KEY = os.environ.get("DIFY_CLASSIFY_API_KEY", "")
DIFY_EXTRACT_API_KEY = os.environ.get("DIFY_EXTRACT_API_KEY", "")
PORT = int(os.environ.get("BRIDGE_PORT", "8088"))
OLLAMA_API_URL = os.environ.get("OLLAMA_API_URL", "http://ollama:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen3:8b")

ANALYZE_PROMPT = """你是高级情报分析专家，拥有20年经验。请严格遵循以下指令处理文档。

📋 用户指令：{instruction}

📄 文档内容：
{documents}

📌 处理要求：
1. 如果是"要素提取"类指令：从文档中逐条提取关键要素（人名、地名、时间、组织、事件），分节罗列
2. 如果是"总结/概括"类指令：提炼核心要点，逻辑清晰，简明扼要
3. 如果是"文本聚合/拟文/报文"类指令：重塑为标准公文/简报/报文格式，语言官方化
4. 如果是"交叉对比"类指令：分析多文档间异同、矛盾点、互补信息，给出综合研判
5. 如果与文档内容无关：礼貌提示，并给出正确操作建议
6. 输出纯Markdown格式，不要添加额外解释说明
7. 引用原文时使用 > 标注，注明来源文档序号"""

# 最大文档内容长度（字符数）
MAX_DOC_CHARS = 50000

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("bridge")

# 屏蔽进度轮询日志
class _ProgressFilter(logging.Filter):
    def filter(self, record):
        return '"POST' in record.getMessage()

logging.getLogger('werkzeug').addFilter(_ProgressFilter())

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = Flask(__name__, static_folder="static", static_url_path="")

# ---------------------------------------------------------------------------
# In-memory task store  (task_id -> dict)
# ---------------------------------------------------------------------------
_tasks: dict[str, dict] = {}
_lock = threading.Lock()

# ---------------------------------------------------------------------------
# Classification state
# ---------------------------------------------------------------------------
_batch_tasks: dict[str, dict] = {}  # task_id -> {"classifier": BatchClassifier, "results": list, "status": str}
_batch_lock = threading.Lock()

# 预设关键词（从文件加载）
_KEYWORDS_FILE = os.path.join(os.path.dirname(__file__), "military_keywords.txt")
_cached_keywords: list[str] = []
_cached_keyword_categories: dict[str, list[str]] = {}


def _load_keywords() -> tuple[list[str], dict[str, list[str]]]:
    global _cached_keywords, _cached_keyword_categories
    if _cached_keywords and _cached_keyword_categories:
        return _cached_keywords, _cached_keyword_categories
    try:
        with open(_KEYWORDS_FILE, "r", encoding="utf-8") as f:
            keywords: list[str] = []
            categories: dict[str, list[str]] = {}
            current_cat = ""
            for line in f:
                line = line.strip()
                if line.startswith("# --- ") and line.endswith("---"):
                    m = re.search(r'\(([^)]+)\)', line)
                    current_cat = m.group(1) if m else ""
                    if current_cat:
                        categories[current_cat] = []
                elif line and not line.startswith("#"):
                    keywords.append(line)
                    if current_cat and current_cat in categories:
                        categories.setdefault(current_cat, []).append(line)
            _cached_keywords = keywords
            _cached_keyword_categories = categories
            logger.info("Loaded %d keywords in %d categories", len(keywords), len(categories))
            return keywords, categories
    except FileNotFoundError:
        logger.warning("Keywords file not found: %s", _KEYWORDS_FILE)
        return [], {}


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())


def _run_workflow(task_id: str, file_ids: list[str], instruction: str, rag_switch: str, folder_name: str) -> None:
    """
    Background thread: call Dify workflow API (blocking mode), store result in _tasks.
    """
    try:
        headers = {"Authorization": f"Bearer {DIFY_API_KEY}", "Content-Type": "application/json"}
        file_refs = [
            {"transfer_method": "local_file", "type": "document", "upload_file_id": fid}
            for fid in file_ids
        ]
        payload = {
            "inputs": {
                "instruction": instruction,
                "rag_switch": rag_switch,
                "doc_files": file_refs,
            },
            "files": file_refs,
            "response_mode": "blocking",
            "user": "bridge-user",
        }

        logger.info("task %s: starting workflow (%d files, instruction=%s)", task_id, len(file_ids), instruction[:80])
        resp = requests.post(f"{DIFY_API_URL}/workflows/run", headers=headers, json=payload, timeout=600)
        resp.raise_for_status()
        data = resp.json()

        workflow_run_id = data.get("workflow_run_id", "")
        wf_data = data.get("data", {})
        status = wf_data.get("status", "unknown")
        outputs = wf_data.get("outputs", {})
        result_text = outputs.get("final_report", outputs.get("text", json.dumps(outputs, ensure_ascii=False)))
        error = wf_data.get("error") or ""

        with _lock:
            _tasks[task_id] = {
                "task_id": task_id,
                "folder_name": folder_name,
                "status": "completed" if status == "succeeded" and not error else "failed",
                "result": result_text,
                "error": str(error) if error else "",
                "created_at": _now_iso(),
                "finished_at": _now_iso(),
                "workflow_run_id": workflow_run_id,
            }
        final_status = "completed" if status == "succeeded" and not error else "failed"
        db.save_analyze_record(task_id, folder_name, instruction, len(file_ids),
                               [], "local", "", rag_switch == "true",
                               final_status, result_text if final_status == "completed" else "",
                               str(error) if error else "",
                               _tasks[task_id]["created_at"], _tasks[task_id]["finished_at"])
        logger.info("task %s: completed (status=%s)", task_id, status)

    except Exception as exc:
        logger.exception("task %s: failed", task_id)
        with _lock:
            _tasks[task_id] = {
                "task_id": task_id,
                "folder_name": folder_name,
                "status": "failed",
                "result": "",
                "error": str(exc),
                "created_at": _now_iso(),
                "finished_at": _now_iso(),
                "workflow_run_id": "",
            }
        db.save_analyze_record(task_id, folder_name, instruction,
                               len(file_ids), [], "local", "",
                               rag_switch == "true", "failed", "", str(exc),
                               _tasks[task_id]["created_at"], _tasks[task_id]["finished_at"])


def _run_workflow_api(task_id: str, files: list, instruction: str, folder_name: str,
                      api_base: str, api_key: str, api_model: str) -> None:
    """
    Background thread: read uploaded files, construct prompt, call OpenAI API, store result.
    """
    try:
        # 1) 读取文件内容
        doc_parts: list[str] = []
        total_chars = 0
        for i, f in enumerate(files):
            try:
                f.stream.seek(0)
                content = f.read().decode("utf-8", errors="replace")
                safe_name = f.filename or f"文档{i+1}"
                if total_chars + len(content) > MAX_DOC_CHARS:
                    remaining = MAX_DOC_CHARS - total_chars
                    if remaining > 500:
                        content = content[:remaining] + "\n\n[提示] 文档过长，已截断..."
                doc_parts.append(f"\n### 文档{i+1}：{safe_name}\n{content}")
                total_chars += len(content)
            except Exception as exc:
                logger.warning("task %s: failed to read file %s: %s", task_id, f.filename, exc)
                doc_parts.append(f"\n### 文档{i+1}：{f.filename}\n[读取失败]")

        docs_text = "\n".join(doc_parts)
        if not docs_text.strip():
            raise ValueError("所有文件读取失败或为空")

        # 2) 构造 prompt
        prompt = ANALYZE_PROMPT.replace("{instruction}", instruction).replace("{documents}", docs_text)

        # 3) 调用 API
        logger.info("task %s: calling API %s model=%s", task_id, api_base, api_model)
        resp = requests.post(
            f"{api_base.rstrip('/')}/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": api_model,
                "messages": [
                    {"role": "system", "content": "你是一个专业的情报分析助手，请严格按指令处理文档。"},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.3,
            },
            timeout=600,
        )
        resp.raise_for_status()
        body = resp.json()
        result_text = body["choices"][0]["message"]["content"]

        # 4) 存储结果
        with _lock:
            _tasks[task_id] = {
                "task_id": task_id,
                "folder_name": folder_name,
                "status": "completed",
                "result": result_text,
                "error": "",
                "created_at": _now_iso(),
                "finished_at": _now_iso(),
                "workflow_run_id": "api-mode",
            }
        db.save_analyze_record(task_id, folder_name, instruction, len(files),
                               [f.filename for f in files], "api", api_model,
                               False, "completed", result_text, "",
                               _tasks[task_id]["created_at"], _tasks[task_id]["finished_at"])
        logger.info("task %s: API completed (%d chars)", task_id, len(result_text))

    except Exception as exc:
        logger.exception("task %s: API failed", task_id)
        with _lock:
            _tasks[task_id] = {
                "task_id": task_id,
                "folder_name": folder_name,
                "status": "failed",
                "result": "",
                "error": str(exc),
                "created_at": _now_iso(),
                "finished_at": _now_iso(),
                "workflow_run_id": "",
            }
        db.save_analyze_record(task_id, folder_name, instruction,
                               len(files), [f.filename for f in files],
                               "api", api_model, False, "failed", "", str(exc),
                               _tasks[task_id]["created_at"], _tasks[task_id]["finished_at"])


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")  # type: ignore[arg-type]


@app.route("/<path:p>")
def static_files(p: str):
    """Serve static assets from the same directory."""
    return send_from_directory(app.static_folder, p)  # type: ignore[arg-type]


@app.post("/upload_folder")
def upload_folder():
    """Accept files + instruction, start workflow or API in background, return task_id."""
    files = request.files.getlist("files")
    instruction = (request.form.get("instruction") or "").strip()
    rag_switch_raw = request.form.get("rag_switch", "false")
    rag_switch = "true" if rag_switch_raw in ("true", "True", "1", True) else "false"
    folder_name = (request.form.get("folder_name") or "未命名任务").strip()
    llm_mode = (request.form.get("llm_mode") or "local").strip()
    api_base = (request.form.get("api_base") or "").strip()
    api_key = (request.form.get("api_key") or "").strip()
    api_model = (request.form.get("api_model") or "").strip()

    if not files:
        return jsonify({"error": "未上传任何文件"}), 400
    if not instruction:
        return jsonify({"error": "未输入处理指令"}), 400

    task_id = str(uuid.uuid4())
    with _lock:
        _tasks[task_id] = {
            "task_id": task_id,
            "folder_name": folder_name,
            "status": "processing",
            "result": "",
            "error": "",
            "created_at": _now_iso(),
            "finished_at": "",
            "workflow_run_id": "",
        }

    # API模式：直接调用 OpenAI 兼容 API
    if llm_mode == "api" and api_base and api_key:
        threading.Thread(
            target=_run_workflow_api,
            args=(task_id, files, instruction, folder_name, api_base, api_key, api_model or "gpt-4"),
            daemon=True,
        ).start()
        return jsonify({"task_id": task_id, "mode": "api"})

    # Dify模式：上传文件到Dify → 启动工作流
    file_ids: list[str] = []
    upload_headers = {"Authorization": f"Bearer {DIFY_API_KEY}"}
    for f in files:
        try:
            safe_name = os.path.basename(f.filename) if f.filename else "untitled"
            up_resp = requests.post(
                f"{DIFY_API_URL}/files/upload",
                headers=upload_headers,
                files={"file": (safe_name, f.stream, f.content_type or "application/octet-stream")},
                timeout=120,
            )
            up_resp.raise_for_status()
            up_data = up_resp.json()
            fid = up_data.get("id", "")
            if not fid:
                return jsonify({"error": f"文件 {f.filename} 上传失败：未返回文件 ID"}), 500
            file_ids.append(fid)
            logger.info("uploaded file %s -> id=%s", f.filename, fid)
        except Exception as exc:
            logger.exception("file upload failed: %s", f.filename)
            return jsonify({"error": f"文件 {f.filename} 上传失败: {exc}"}), 500

    threading.Thread(
        target=_run_workflow,
        args=(task_id, file_ids, instruction, rag_switch, folder_name),
        daemon=True,
    ).start()

    return jsonify({"task_id": task_id, "mode": "dify"})


@app.get("/task_status/<task_id>")
def task_status(task_id: str):
    """Return task status and result."""
    with _lock:
        t = _tasks.get(task_id)
    if t:
        return jsonify(t)
    # 回退：查数据库
    rec = db.get_record(task_id)
    if rec:
        return jsonify({
            "task_id": rec.get("task_id"),
            "folder_name": rec.get("folder_name", ""),
            "status": rec.get("status", ""),
            "result": rec.get("result", rec.get("report", "")),
            "error": rec.get("error", ""),
            "created_at": str(rec.get("created_at", "")),
            "finished_at": str(rec.get("finished_at", "")),
            "workflow_run_id": "",
        })
    return jsonify({"status": "not_found", "error": "任务不存在"}), 404


@app.get("/tasks_recent")
def tasks_recent():
    """Return recent tasks (up to limit)."""
    limit = request.args.get("limit", 10, type=int)
    with _lock:
        items = list(_tasks.values())
    items.sort(key=lambda x: x.get("created_at", ""), reverse=True)
    # 合并数据库记录（排除内存中已有的完成/失败任务）
    mem_ids = {i.get("task_id") for i in items if i.get("status") in ("completed", "failed", "cancelled")}
    db_records = db.get_recent_records(limit)
    for rec in db_records:
        tid = rec.get("task_id", "")
        if tid and tid not in mem_ids:
            items.append({
                "task_id": tid,
                "folder_name": rec.get("folder_name", ""),
                "status": rec.get("status", ""),
                "result": "",
                "error": "",
                "created_at": rec.get("created_at", ""),
                "finished_at": rec.get("finished_at", ""),
                "workflow_run_id": "",
            })
    items.sort(key=lambda x: str(x.get("created_at", "")), reverse=True)
    return jsonify({"items": items[:limit]})


@app.post("/tasks/<task_id>/cancel")
def cancel_task(task_id: str):
    """Mark task as cancelled."""
    with _lock:
        t = _tasks.get(task_id)
        if t and t["status"] in ("processing", "waiting"):
            t["status"] = "cancelled"
            t["finished_at"] = _now_iso()
            return jsonify({"status": "success"})
    return jsonify({"status": "error", "message": "任务不存在或无法终止"}), 400


@app.delete("/tasks/<task_id>")
def delete_task(task_id: str):
    """Delete task record."""
    with _lock:
        if task_id in _tasks:
            del _tasks[task_id]
            db.delete_record(task_id)
            return jsonify({"status": "success"})
    db.delete_record(task_id)
    if db.get_record(task_id):
        return jsonify({"status": "error", "message": "删除失败"}), 500
    return jsonify({"status": "success"})


@app.get("/api/history")
def history_list():
    """分页查询历史记录"""
    module = request.args.get("module", "").strip()
    limit = request.args.get("limit", 20, type=int)
    offset = request.args.get("offset", 0, type=int)
    search = request.args.get("search", "").strip()
    items = db.get_history(module, limit, offset, search)
    stats = db.get_history_stats()
    return jsonify({"items": items, "stats": stats, "limit": limit, "offset": offset})


@app.get("/api/history/stats")
def history_stats():
    """历史记录统计"""
    return jsonify(db.get_history_stats())


# ═══════════════════════════════════════════════════════════════════════
# 语义分类 API
# ═══════════════════════════════════════════════════════════════════════


def _run_batch_classify(task_id: str, lines: list[str], keyword_categories: dict[str, list[str]],
                        folder_name: str = "", original_filename: str = "",
                        llm_mode: str = "local", api_base: str = "",
                        api_key: str = "", api_model: str = "") -> None:
    """Background thread: batch classify all lines."""
    try:
        classifier = BatchClassifier(
            dify_api_url=DIFY_API_URL,
            dify_api_key=DIFY_CLASSIFY_API_KEY or DIFY_API_KEY,
            keyword_categories=keyword_categories,
            llm_mode=llm_mode,
            api_base=api_base,
            api_key=api_key,
            api_model=api_model,
        )
        results: list[dict] = []
        error_count = 0
        with _batch_lock:
            model_label = f"API ({api_model})" if llm_mode == "api" else "本地 Ollama (qwen3:8b)"
            _batch_tasks[task_id] = {
                "status": "processing", "total": len(lines), "cancelled": False,
                "results": results, "classifier": classifier,
                "folder_name": folder_name, "original_filename": original_filename,
                "model_info": model_label,
            }

        for progress in classifier.classify_lines(lines, task_id):
            results.append(progress["result"])
            if progress["result"].get("method") == "error":
                error_count += 1

        with _batch_lock:
            if _batch_tasks.get(task_id, {}).get("cancelled"):
                _batch_tasks[task_id]["status"] = "cancelled"
                _batch_tasks[task_id]["results"] = results
                db.save_classify_record(task_id, folder_name, original_filename,
                                        len(lines), llm_mode, api_model, "cancelled",
                                        {}, results, "用户取消", _now_iso(), _now_iso())
                logger.info("Batch classify %s: cancelled, %d lines saved", task_id, len(results))
            else:
                _batch_tasks[task_id]["status"] = "completed"
                _batch_tasks[task_id]["results"] = results
                stats = {}
                for r in results:
                    lbl = r.get("label", "未知")
                    status = r.get("status", "verified")
                    if status == "conflict":
                        stats["冲突"] = stats.get("冲突", 0) + 1
                    else:
                        stats[lbl] = stats.get(lbl, 0) + 1
                created = _now_iso()
                finished = _now_iso()
                db.save_classify_record(task_id, folder_name, original_filename,
                                        len(lines), llm_mode, api_model, "completed",
                                        stats, results, "", created, finished)
                logger.info("Batch classify %s: completed, %d lines, %d errors", task_id, len(results), error_count)

    except Exception as exc:
        logger.exception("Batch classify %s: failed", task_id)
        with _batch_lock:
            _batch_tasks[task_id] = {"status": "failed", "results": [], "error": str(exc)}
        db.save_classify_record(task_id, folder_name, original_filename,
                                len(lines), llm_mode, api_model, "failed",
                                {}, [], str(exc), _now_iso(), _now_iso())

@app.post("/api/classify/batch")
def classify_batch():
    """Submit a batch classification task."""
    # 先取消所有正在运行的旧任务
    with _batch_lock:
        for tid, info in list(_batch_tasks.items()):
            if info.get("status") == "processing" and not info.get("cancelled"):
                info["cancelled"] = True
                classifier = info.get("classifier")
                if classifier:
                    classifier.cancel(tid)
                logger.info("Batch classify %s: cancelled by new submission", tid)

    files = request.files.getlist("files")
    folder_name = (request.form.get("folder_name") or "批量分类").strip()
    llm_mode = (request.form.get("llm_mode") or "local").strip()
    api_base = (request.form.get("api_base") or "").strip()
    api_key = (request.form.get("api_key") or "").strip()
    api_model = (request.form.get("api_model") or "qwen-plus").strip()

    if not files:
        return jsonify({"error": "未上传任何文件"}), 400

    # 解析文本文件内容
    all_lines: list[str] = []
    original_filename = "unknown"
    for f in files:
        original_filename = f.filename or "unknown"
        try:
            content = f.read().decode("utf-8", errors="replace")
            lines = split_text_to_lines(content)
            all_lines.extend(lines)
        except Exception as exc:
            return jsonify({"error": f"文件 {f.filename} 读取失败: {exc}"}), 400

    if not all_lines:
        return jsonify({"error": "文件中未检测到有效文本内容"}), 400

    _, keyword_categories = _load_keywords()
    if not keyword_categories:
        return jsonify({"error": "未找到预设关键词词典"}), 400

    # 创建任务并启动后台处理
    task_id = str(uuid.uuid4())
    threading.Thread(
        target=_run_batch_classify,
        args=(task_id, all_lines, keyword_categories, folder_name, original_filename,
              llm_mode, api_base, api_key, api_model),
        daemon=True,
    ).start()

    return jsonify({
        "task_id": task_id,
        "total_lines": len(all_lines),
        "original_filename": original_filename,
        "folder_name": folder_name,
    })


@app.get("/api/classify/progress/<task_id>")
def classify_progress(task_id: str):
    """Return batch classification progress."""
    with _batch_lock:
        info = _batch_tasks.get(task_id)

    if not info:
        return jsonify({"status": "not_found", "error": "任务不存在"}), 404

    results = info.get("results", [])
    completed = len(results)
    total = info.get("total", len(results))

    return jsonify({
        "status": info["status"],
        "completed": completed,
        "total": max(completed, total),
        "results": results,
    })


@app.post("/api/classify/cancel/<task_id>")
def classify_cancel(task_id: str):
    """Cancel a running batch classification task."""
    with _batch_lock:
        if task_id in _batch_tasks:
            _batch_tasks[task_id]["cancelled"] = True
            classifier = _batch_tasks[task_id].get("classifier")
            if classifier:
                classifier.cancel(task_id)
            return jsonify({"status": "success"})
    return jsonify({"status": "error", "message": "任务不存在或已完成"}), 404


@app.get("/api/classify/active")
def classify_active():
    """Return the currently active (processing) task, if any."""
    with _batch_lock:
        for task_id, info in _batch_tasks.items():
            if info.get("status") == "processing" and not info.get("cancelled"):
                return jsonify({
                    "active": True,
                    "task_id": task_id,
                    "total": info.get("total", 0),
                    "completed": len(info.get("results", [])),
                    "folder_name": info.get("folder_name", ""),
                    "original_filename": info.get("original_filename", ""),
                    "results": info.get("results", []),
                })
    return jsonify({"active": False})


@app.get("/api/classify/download/<task_id>")
def classify_download(task_id: str):
    """Download classification report as Markdown text file."""
    with _batch_lock:
        info = _batch_tasks.get(task_id)

    if not info or info["status"] != "completed":
        return jsonify({"error": "任务未完成或不存在"}), 400

    results: list[dict] = info.get("results", [])
    original_filename = ""  # Stored in the initial response, but we don't persist it
    classifier = info.get("classifier")
    model_info = info.get("model_info", "")
    if classifier and hasattr(classifier, "generate_report"):
        report = classifier.generate_report(results, original_filename)
    else:
        report = _generate_report_simple(results, model_info)

    return FlaskResponse(
        report,
        mimetype="text/plain; charset=utf-8",
        headers={"Content-Disposition": f"attachment; filename=classification_report_{task_id[:8]}.md"},
    )


@app.get("/api/classify/preprocessed/<task_id>")
def classify_preprocessed(task_id: str):
    """Download preprocessed text (纠错后，去掉噪声，按行号排序)."""
    with _batch_lock:
        info = _batch_tasks.get(task_id)

    if not info or info["status"] != "completed":
        return jsonify({"error": "任务未完成或不存在"}), 400

    results: list[dict] = info.get("results", [])
    classifier = info.get("classifier")
    if classifier and hasattr(classifier, "generate_preprocessed_txt"):
        txt = classifier.generate_preprocessed_txt(results)
    else:
        valid = [r for r in results if r.get("label") not in ("噪声", "空行", "分类失败", "解析失败")]
        valid.sort(key=lambda r: r.get("line_no", 0))
        txt = "\n".join([r.get("preprocessed") or r.get("original", "") for r in valid])

    return FlaskResponse(
        txt,
        mimetype="text/plain; charset=utf-8",
        headers={"Content-Disposition": f"attachment; filename=preprocessed_{task_id[:8]}.txt"},
    )


def _generate_report_simple(results: list[dict], model_info: str = "") -> str:
    """Standalone report generator."""
    categorized: dict[str, list[dict]] = {}
    for r in results:
        if r.get("status") == "verified":
            cat = r.get("label", "未分类")
            categorized.setdefault(cat, []).append(r)
    now = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    parts = [f"## 语音文本分类报告\n\n", f"- 处理时间：{now}\n"]
    if model_info:
        parts.append(f"- 使用模型：{model_info}\n")
    parts.append(f"- 总条目：{len(results)}\n")
    verified = len([r for r in results if r.get("status") == "verified" and r.get("label") != "空行"])
    conflicts = len([r for r in results if r.get("status") == "conflict"])
    parts.append(f"- 一致通过：{verified} 条\n")
    parts.append(f"- 冲突待处理：{conflicts} 条\n\n")
    for cat in ["指令", "情报", "态势", "火力", "补给", "通信", "敌情", "计划", "噪声"]:
        items = categorized.get(cat, [])
        if not items:
            continue
        parts.append(f"### {cat}（{len(items)} 条）\n\n")
        for r in items[:50]:
            parts.append(f"- [{r['line_no']}] [{r['confidence']:.2f}] {r['original'][:60]}\n")
        parts.append("\n")
    if conflicts:
        parts.append(f"### ⚠ 冲突待处理（{conflicts} 条）\n\n")
        parts.append("| 序号 | 原文 | KW分类 | LLM分类 |\n")
        parts.append("|------|------|--------|--------|\n")
        for r in [x for x in results if x.get("status") == "conflict"][:30]:
            orig = r["original"].replace("|", "｜")[:40]
            parts.append(f"| {r['line_no']} | {orig} | {r.get('kw_label','?')} | {r.get('llm_label','?')} |\n")
        parts.append("\n")
    return "".join(parts)


# ---------------------------------------------------------------------------
# 纠正反馈闭环
# ---------------------------------------------------------------------------
_CORRECTIONS_FILE = os.path.join(os.path.dirname(__file__), "corrections.jsonl")


def _load_corrections(limit: int = 10) -> list[dict]:
    try:
        with open(_CORRECTIONS_FILE, "r", encoding="utf-8") as f:
            lines = f.readlines()
        return [json.loads(l) for l in lines[-limit:] if l.strip()]
    except FileNotFoundError:
        return []


def _save_correction(entry: dict) -> None:
    with open(_CORRECTIONS_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _build_correction_examples(limit: int = 10) -> str:
    corrections = _load_corrections(limit)
    if not corrections:
        return ""
    lines = ["\n\n【人工纠正记录（最近10条）注意：以下为已纠正确认的分类结果，遇到相似文本时优先参考这些人为判断】"]
    for c in corrections[-10:]:
        lines.append(f"- 原文: \"{c['original_text'][:60]}\" → 正确分类: {c['correct_label']}（LLM曾误判为{c.get('llm_label','?')}）")
    return "\n".join(lines)


@app.post("/api/classify/correct")
def classify_correct():
    """提交人工纠正结果"""
    data = request.get_json(force=True, silent=True) or {}
    for k in ("original_text", "correct_label", "line_no"):
        if k not in data:
            return jsonify({"error": f"缺少参数: {k}"}), 400
    entry = {
        "original_text": data["original_text"],
        "correct_label": data["correct_label"],
        "kw_label": data.get("kw_label", ""),
        "llm_label": data.get("llm_label", ""),
        "line_no": data["line_no"],
        "task_id": data.get("task_id", ""),
        "corrected_at": _now_iso(),
    }
    _save_correction(entry)
    with _batch_lock:
        task_id = data.get("task_id", "")
        if task_id and task_id in _batch_tasks:
            results = _batch_tasks[task_id].get("results", [])
            for r in results:
                if r.get("line_no") == data["line_no"]:
                    r["label"] = data["correct_label"]
                    r["status"] = "verified"
                    r["confidence"] = 1.0
                    r["evidence"] = [f"人工纠正: {data.get('kw_label','?')}→{data['correct_label']}"]
                    break
    logger.info("Correction saved: line %s → %s", data["line_no"], data["correct_label"])
    return jsonify({"status": "success"})


@app.get("/api/classify/corrections")
def classify_corrections():
    """获取纠正记录"""
    limit = request.args.get("limit", 10, type=int)
    return jsonify({"items": _load_corrections(limit)})


# ---------------------------------------------------------------------------
# 关键要素提取
# ---------------------------------------------------------------------------
_extract_tasks: dict[str, dict] = {}
_extract_lock = threading.Lock()


def _run_extract_task(task_id: str, lines: list[str], selected_labels: list[str],
                      llm_mode: str, api_base: str, api_key: str, api_model: str,
                      file_name: str = "") -> None:
    """从文本标签解析 → 过滤 → LLM提取 → LLM检验。"""
    try:
        with _extract_lock:
            _extract_tasks[task_id] = {"status": "extracting", "stage": "解析标签中"}

        # 解析每行末尾标签，例如  "报告指挥所...（态势）"
        parsed: list[tuple[str, str]] = []
        for i, line in enumerate(lines):
            m = re.search(r'（(\S+)）\s*$', line)
            if m:
                label = m.group(1)
                text = re.sub(r'（\S+）\s*$', '', line)
                parsed.append((text, label))
            else:
                parsed.append((line, "未分类"))
        logger.info("Extract %s: parsed %d lines, %d with labels",
                     task_id, len(lines), len([p for p in parsed if p[1] != "未分类"]))

        # 过滤选中标签
        filtered = [(i+1, t) for i, (t, l) in enumerate(parsed) if l in selected_labels and l != "噪声"]
        if not filtered:
            with _extract_lock:
                _extract_tasks[task_id] = {"status": "failed", "error": "选中的标签下无匹配文本"}
            db.save_extract_record(task_id, labels_str, file_name, 0, llm_mode,
                                   api_model, "failed", "", "选中的标签下无匹配文本",
                                   _now_iso(), _now_iso())
            return
        filtered_text = "\n".join(f"[行{n}] {t}" for n, t in filtered)
        labels_str = "、".join(selected_labels)

        if llm_mode == "local" and DIFY_EXTRACT_API_KEY and DIFY_EXTRACT_API_KEY != "app-extract-placeholder":
            # 本地模式：调用 Dify 提取工作流（工作流内部处理提取+检验）
            with _extract_lock:
                _extract_tasks[task_id] = {"status": "extracting", "stage": "调用提取工作流中"}
            wf_result = _call_dify_extract(filtered_text, labels_str)
            if wf_result.startswith("[错误]"):
                with _extract_lock:
                    _extract_tasks[task_id] = {"status": "failed", "error": wf_result}
                db.save_extract_record(task_id, labels_str, file_name, 0, llm_mode,
                                       api_model, "failed", "", wf_result,
                                       _now_iso(), _now_iso())
                return
            now = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            report = f"""## 关键要素提取报告

- 处理时间：{now}
- 选中标签：{labels_str}
- 匹配行数：{len(filtered)} 条
- 模式：本地 Dify (Ollama)

---

{wf_result}
"""
            with _extract_lock:
                _extract_tasks[task_id] = {"status": "completed", "report": report,
                                            "extract": wf_result, "verify": ""}
            db.save_extract_record(task_id, labels_str, file_name, len(filtered),
                                   llm_mode, "", "completed", report, "",
                                   _now_iso(), _now_iso())
            logger.info("Extract %s: completed via Dify workflow", task_id)
            return

        # API 模式（或本地模式无 Dify 工作流时的回退）：两阶段 API 调用
        with _extract_lock:
            _extract_tasks[task_id] = {"status": "extracting", "stage": "提取关键要素中"}
        is_local_fallback = llm_mode == "local"
        if is_local_fallback:
            extract_result = _call_ollama_raw(
                EXTRACT_PROMPT.replace("{labels}", labels_str).replace("{filtered_lines}", filtered_text)
            )
        else:
            extract_result = _call_api_simple(
                api_base, api_key, api_model,
                EXTRACT_PROMPT.replace("{labels}", labels_str).replace("{filtered_lines}", filtered_text)
            )
        if extract_result.startswith("[错误]"):
            with _extract_lock:
                _extract_tasks[task_id] = {"status": "failed", "error": extract_result}
            db.save_extract_record(task_id, labels_str, file_name, 0, llm_mode,
                                   api_model, "failed", "", extract_result,
                                   _now_iso(), _now_iso())
            return

        # 检验
        with _extract_lock:
            _extract_tasks[task_id] = {"status": "verifying", "stage": "检验遗漏中"}
        verify_prompt = VERIFY_PROMPT.replace("{extracted_summary}", extract_result).replace("{all_lines}", filtered_text).replace("{labels}", labels_str)
        if is_local_fallback:
            verify_result = _call_ollama_raw(verify_prompt)
        else:
            verify_result = _call_api_simple(api_base, api_key, api_model, verify_prompt)

        now = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        mode_label = "本地 Ollama (两阶段)" if is_local_fallback else f"API ({api_model})"
        final = verify_result if not verify_result.startswith('[错误]') else extract_result
        report = f"""## 关键要素提取报告

- 处理时间：{now}
- 选中标签：{labels_str}
- 匹配行数：{len(filtered)} 条
- 模式：{mode_label}

---

{final}
"""
        with _extract_lock:
            _extract_tasks[task_id] = {"status": "completed", "report": report,
                                        "extract": extract_result, "verify": verify_result}
        db.save_extract_record(task_id, labels_str, file_name, len(filtered),
                               llm_mode, api_model, "completed", report, "",
                               _now_iso(), _now_iso())
        logger.info("Extract %s: completed", task_id)

    except Exception as exc:
        logger.exception("Extract %s: failed", task_id)
        with _extract_lock:
            _extract_tasks[task_id] = {"status": "failed", "error": str(exc)}
        db.save_extract_record(task_id, "", file_name, 0, llm_mode,
                               api_model, "failed", "", str(exc),
                               _now_iso(), _now_iso())


def _call_api_simple(api_base: str, api_key: str, api_model: str, prompt: str) -> str:
    try:
        resp = requests.post(
            f"{api_base.rstrip('/')}/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={"model": api_model, "messages": [{"role": "user", "content": prompt}], "temperature": 0.3},
            timeout=600,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]
    except Exception as exc:
        logger.error("API call failed: %s", exc)
        return f"[错误] {exc}"


def _call_ollama_raw(prompt: str) -> str:
    try:
        resp = requests.post(
            f"{OLLAMA_API_URL.rstrip('/')}/api/generate",
            headers={"Content-Type": "application/json"},
            json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False, "options": {"temperature": 0.3}},
            timeout=600,
        )
        resp.raise_for_status()
        return resp.json().get("response", "")
    except Exception as exc:
        logger.error("Ollama call failed: %s", exc)
        return f"[错误] {exc}"


def _call_dify_extract(filtered_lines: str, labels: str) -> str:
    """调用 Dify 关键要素提取工作流"""
    try:
        headers = {"Authorization": f"Bearer {DIFY_EXTRACT_API_KEY}", "Content-Type": "application/json"}
        payload = {
            "inputs": {
                "filtered_lines": filtered_lines,
                "labels": labels,
            },
            "response_mode": "blocking",
            "user": "extract-bridge",
        }
        resp = requests.post(f"{DIFY_API_URL}/workflows/run", headers=headers, json=payload, timeout=600)
        resp.raise_for_status()
        data = resp.json()
        outputs = data.get("data", {}).get("outputs", {})
        return outputs.get("extraction_report", outputs.get("text", ""))
    except Exception as exc:
        logger.exception("Dify extract workflow failed")
        return f"[错误] {exc}"


@app.post("/api/extract/batch")
@app.post("/api/extract/extract")
def extract_run():
    """上传文件+选中标签 → 解析标签 → 过滤 → LLM提取 → LLM检验。"""
    files = request.files.getlist("files")
    labels_str = (request.form.get("labels") or "").strip()
    labels = [l.strip() for l in labels_str.split(",") if l.strip()] if labels_str else []
    llm_mode = (request.form.get("llm_mode") or "local").strip()
    api_base = (request.form.get("api_base") or "").strip()
    api_key = (request.form.get("api_key") or "").strip()
    api_model = (request.form.get("api_model") or "qwen-plus").strip()

    if not files:
        return jsonify({"error": "未上传文件"}), 400
    if not labels:
        return jsonify({"error": "未选择标签"}), 400

    # 读取文件、解析每行标签
    all_lines: list[str] = []
    for f in files:
        try:
            content = f.read().decode("utf-8", errors="replace")
            lines = split_text_to_lines(content)
            all_lines.extend(lines)
        except Exception as exc:
            return jsonify({"error": f"文件 {f.filename} 读取失败: {exc}"}), 400

    if not all_lines:
        return jsonify({"error": "文件中未检测到有效文本内容"}), 400

    task_id = str(uuid.uuid4())
    file_name = files[0].filename if files else ""
    threading.Thread(
        target=_run_extract_task,
        args=(task_id, all_lines, labels, llm_mode, api_base, api_key, api_model, file_name),
        daemon=True,
    ).start()

    return jsonify({"task_id": task_id, "total_lines": len(all_lines)})


@app.get("/api/extract/progress/<task_id>")
def extract_progress(task_id: str):
    with _extract_lock:
        info = _extract_tasks.get(task_id)
    if not info:
        return jsonify({"status": "not_found"}), 404
    resp = {"status": info["status"]}
    for k in ("total", "completed", "stage", "report", "results", "error"):
        if k in info:
            resp[k] = info[k]
    return jsonify(resp)


@app.get("/api/extract/download/<task_id>")
def extract_download(task_id: str):
    with _extract_lock:
        info = _extract_tasks.get(task_id)
    if not info or info.get("status") != "completed":
        return jsonify({"error": "提取未完成"}), 400
    report = info.get("report", "")
    return FlaskResponse(
        report, mimetype="text/plain; charset=utf-8",
        headers={"Content-Disposition": f"attachment; filename=extract_report_{task_id[:8]}.md"},
    )


@app.get("/api/extract/active")
def extract_active():
    with _extract_lock:
        for tid, info in _extract_tasks.items():
            if info.get("status") in ("classifying", "extracting", "verifying"):
                return jsonify({"active": True, "task_id": tid, "status": info["status"],
                                "total": info.get("total", 0), "completed": info.get("completed", 0)})
    return jsonify({"active": False})


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    db.init_db()
    logger.info("Bridge starting on port %d, Dify API: %s", PORT, DIFY_API_URL)
    app.run(host="0.0.0.0", port=PORT, debug=False)
