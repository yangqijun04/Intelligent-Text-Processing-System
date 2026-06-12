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

from preprocessing import clean_text, estimate_asr_noise_level, split_text_to_lines
from semantic_classifier import BatchClassifier, KeywordMatcher

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DIFY_API_URL = os.environ.get("DIFY_API_URL", "http://api:5001/v1")
DIFY_API_KEY = os.environ.get("DIFY_API_KEY", "")
DIFY_CLASSIFY_API_KEY = os.environ.get("DIFY_CLASSIFY_API_KEY", "")
PORT = int(os.environ.get("BRIDGE_PORT", "8088"))

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
    """Accept files + instruction, start workflow in background, return task_id."""
    files = request.files.getlist("files")
    instruction = (request.form.get("instruction") or "").strip()
    rag_switch_raw = request.form.get("rag_switch", "false")
    rag_switch = "true" if rag_switch_raw in ("true", "True", "1", True) else "false"
    folder_name = (request.form.get("folder_name") or "未命名任务").strip()

    if not files:
        return jsonify({"error": "未上传任何文件"}), 400
    if not instruction:
        return jsonify({"error": "未输入处理指令"}), 400

    # 1) Upload each file to Dify and collect file IDs
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

    # 2) Create task and start workflow in background thread
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

    threading.Thread(
        target=_run_workflow,
        args=(task_id, file_ids, instruction, rag_switch, folder_name),
        daemon=True,
    ).start()

    return jsonify({"task_id": task_id})


@app.get("/task_status/<task_id>")
def task_status(task_id: str):
    """Return task status and result."""
    with _lock:
        t = _tasks.get(task_id)
    if not t:
        return jsonify({"status": "not_found", "error": "任务不存在"}), 404
    return jsonify(t)


@app.get("/tasks_recent")
def tasks_recent():
    """Return recent tasks (up to limit)."""
    limit = request.args.get("limit", 10, type=int)
    with _lock:
        items = list(_tasks.values())
    items.sort(key=lambda x: x.get("created_at", ""), reverse=True)
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
            return jsonify({"status": "success"})
    return jsonify({"status": "error", "message": "任务不存在"}), 404


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
            _batch_tasks[task_id] = {
                "status": "processing", "total": len(lines), "cancelled": False,
                "results": results, "classifier": classifier,
                "folder_name": folder_name, "original_filename": original_filename,
            }

        for progress in classifier.classify_lines(lines, task_id):
            results.append(progress["result"])
            if progress["result"].get("method") == "error":
                error_count += 1

        with _batch_lock:
            if _batch_tasks.get(task_id, {}).get("cancelled"):
                _batch_tasks[task_id]["status"] = "cancelled"
                _batch_tasks[task_id]["results"] = results
                logger.info("Batch classify %s: cancelled, %d lines saved", task_id, len(results))
            else:
                _batch_tasks[task_id]["status"] = "completed"
                _batch_tasks[task_id]["results"] = results
                logger.info("Batch classify %s: completed, %d lines, %d errors", task_id, len(results), error_count)

    except Exception as exc:
        logger.exception("Batch classify %s: failed", task_id)
        with _batch_lock:
            _batch_tasks[task_id] = {"status": "failed", "results": [], "error": str(exc)}


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
    if classifier and hasattr(classifier, "generate_report"):
        report = classifier.generate_report(results, original_filename)
    else:
        report = _generate_report_simple(results)

    return FlaskResponse(
        report,
        mimetype="text/plain; charset=utf-8",
        headers={"Content-Disposition": f"attachment; filename=classification_report_{task_id[:8]}.md"},
    )


def _generate_report_simple(results: list[dict]) -> str:
    """Standalone report generator."""
    categorized: dict[str, list[dict]] = {}
    for r in results:
        if r.get("status") == "verified":
            cat = r.get("label", "未分类")
            categorized.setdefault(cat, []).append(r)
    now = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    parts = [f"## 语音文本分类报告\n\n", f"- 处理时间：{now}\n", f"- 总条目：{len(results)}\n"]
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
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logger.info("Bridge starting on port %d, Dify API: %s", PORT, DIFY_API_URL)
    app.run(host="0.0.0.0", port=PORT, debug=False)
