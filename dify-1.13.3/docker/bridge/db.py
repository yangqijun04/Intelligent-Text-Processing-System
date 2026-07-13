import json
import logging
import os
import sqlite3
import threading
from contextlib import contextmanager

DB_PATH = os.path.join(os.path.dirname(__file__), "records.db")

logger = logging.getLogger("db")

_conn = None
_conn_lock = threading.Lock()


def _get_conn():
    global _conn
    if _conn is None:
        _conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        _conn.row_factory = sqlite3.Row
        _conn.execute("PRAGMA journal_mode=DELETE")
        _conn.execute("PRAGMA busy_timeout=5000")
    return _conn


@contextmanager
def _cursor():
    with _conn_lock:
        conn = _get_conn()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise


def _json(val):
    return json.dumps(val, ensure_ascii=False) if val is not None else None


def init_db():
    """建表（幂等）"""
    with _cursor() as conn:
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS analyze_records (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            task_id     TEXT UNIQUE NOT NULL,
            folder_name TEXT,
            instruction TEXT,
            file_count  INTEGER DEFAULT 0,
            file_names  TEXT DEFAULT '[]',
            llm_mode    TEXT DEFAULT 'local',
            api_model   TEXT,
            rag_switch  INTEGER DEFAULT 0,
            status      TEXT DEFAULT 'processing',
            result      TEXT,
            error       TEXT,
            created_at  TEXT,
            finished_at TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_analyze_created
            ON analyze_records(created_at COLLATE NOCASE);

        CREATE TABLE IF NOT EXISTS classify_records (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            task_id     TEXT UNIQUE NOT NULL,
            folder_name TEXT,
            file_name   TEXT,
            total_lines INTEGER,
            llm_mode    TEXT DEFAULT 'local',
            api_model   TEXT,
            status      TEXT DEFAULT 'processing',
            stats       TEXT DEFAULT '{}',
            results     TEXT DEFAULT '[]',
            error       TEXT,
            created_at  TEXT,
            finished_at TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_classify_created
            ON classify_records(created_at COLLATE NOCASE);

        CREATE TABLE IF NOT EXISTS extract_records (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            task_id     TEXT UNIQUE NOT NULL,
            labels      TEXT,
            file_name   TEXT,
            line_count  INTEGER,
            llm_mode    TEXT DEFAULT 'local',
            api_model   TEXT,
            status      TEXT DEFAULT 'processing',
            report      TEXT,
            error       TEXT,
            created_at  TEXT,
            finished_at TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_extract_created
            ON extract_records(created_at COLLATE NOCASE);
        """)
    logger.info("SQLite tables initialized at %s", DB_PATH)


# ─── 写入 ───

def save_analyze_record(task_id: str, folder_name: str, instruction: str,
                        file_count: int, file_names: list, llm_mode: str,
                        api_model: str, rag_switch: bool,
                        status: str, result: str, error: str,
                        created_at: str, finished_at: str):
    try:
        with _cursor() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO analyze_records
                (task_id, folder_name, instruction, file_count, file_names,
                 llm_mode, api_model, rag_switch, status, result, error,
                 created_at, finished_at)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, (task_id, folder_name, instruction, file_count,
                  _json(file_names), llm_mode, api_model, 1 if rag_switch else 0,
                  status, result, error, created_at, finished_at))
    except Exception as exc:
        logger.error("save_analyze_record failed: %s", exc)


def save_classify_record(task_id: str, folder_name: str, file_name: str,
                         total_lines: int, llm_mode: str, api_model: str,
                         status: str, stats: dict, results: list,
                         error: str, created_at: str, finished_at: str):
    try:
        with _cursor() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO classify_records
                (task_id, folder_name, file_name, total_lines, llm_mode,
                 api_model, status, stats, results, error, created_at, finished_at)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
            """, (task_id, folder_name, file_name, total_lines, llm_mode,
                  api_model, status, _json(stats), _json(results),
                  error, created_at, finished_at))
    except Exception as exc:
        logger.error("save_classify_record failed: %s", exc)


def save_extract_record(task_id: str, labels: str, file_name: str,
                        line_count: int, llm_mode: str, api_model: str,
                        status: str, report: str, error: str,
                        created_at: str, finished_at: str):
    try:
        with _cursor() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO extract_records
                (task_id, labels, file_name, line_count, llm_mode,
                 api_model, status, report, error, created_at, finished_at)
                VALUES (?,?,?,?,?,?,?,?,?,?,?)
            """, (task_id, labels, file_name, line_count, llm_mode,
                  api_model, status, report, error, created_at, finished_at))
    except Exception as exc:
        logger.error("save_extract_record failed: %s", exc)


# ─── 查询 ───

def get_recent_records(limit: int = 10) -> list[dict]:
    """合并三表，取最近 N 条记录"""
    try:
        with _cursor() as conn:
            rows = conn.execute("""
                SELECT task_id, folder_name, status, 'analyze' as module,
                       NULL as labels, created_at, finished_at
                FROM analyze_records ORDER BY created_at DESC LIMIT ?
            """, (limit,)).fetchall()
            result = [dict(r) for r in rows]
            rows2 = conn.execute("""
                SELECT task_id, folder_name, status, 'classify' as module,
                       NULL as labels, created_at, finished_at
                FROM classify_records ORDER BY created_at DESC LIMIT ?
            """, (limit,)).fetchall()
            result.extend(dict(r) for r in rows2)
            rows3 = conn.execute("""
                SELECT task_id, labels as folder_name, status, 'extract' as module,
                       labels, created_at, finished_at
                FROM extract_records ORDER BY created_at DESC LIMIT ?
            """, (limit,)).fetchall()
            result.extend(dict(r) for r in rows3)
            result.sort(key=lambda x: str(x.get("created_at", "")), reverse=True)
            return result[:limit]
    except Exception as exc:
        logger.error("get_recent_records failed: %s", exc)
        return []


def get_record(task_id: str) -> dict | None:
    """跨三表查单条记录"""
    try:
        for table in ('analyze_records', 'classify_records', 'extract_records'):
            with _cursor() as conn:
                row = conn.execute(
                    f"SELECT * FROM {table} WHERE task_id = ?", (task_id,)
                ).fetchone()
                if row:
                    return dict(row)
    except Exception as exc:
        logger.error("get_record failed: %s", exc)
    return None


def delete_record(task_id: str):
    """跨三表删记录"""
    try:
        for table in ('analyze_records', 'classify_records', 'extract_records'):
            with _cursor() as conn:
                conn.execute(f"DELETE FROM {table} WHERE task_id = ?", (task_id,))
    except Exception as exc:
        logger.error("delete_record failed: %s", exc)


def get_history(module: str = "", limit: int = 20, offset: int = 0,
                search: str = "") -> list[dict]:
    """分页查询历史记录，按 created_at 倒序"""
    table_map = {
        "analyze": "analyze_records",
        "classify": "classify_records",
        "extract": "extract_records",
    }
    tables = {module: table_map[module]} if module in table_map else table_map

    # 列名对齐（三表取并集，缺的补 NULL）
    select_map = {
        "analyze": ("task_id, folder_name, instruction, file_count, file_names, rag_switch, result",
                    {"fields": ["folder_name", "instruction", "result"]}),
        "classify": ("task_id, folder_name, NULL as instruction, NULL as file_count, NULL as file_names, NULL as rag_switch, file_name||' '||stats as result",
                     {"fields": ["folder_name", "file_name", "stats"]}),
        "extract": ("task_id, labels as folder_name, NULL as instruction, NULL as file_count, NULL as file_names, NULL as rag_switch, labels||' '||substr(report,1,100) as result",
                    {"fields": ["labels", "report"]}),
    }

    try:
        with _cursor() as conn:
            all_rows = []
            for mod, tbl in tables.items():
                cols, srch = select_map[mod]
                sql = (f"SELECT {cols}, status, llm_mode, api_model, "
                       f"error, created_at, finished_at, '{mod}' as module "
                       f"FROM {tbl}")
                params = []
                if search:
                    like = f"%{search}%"
                    conditions = " OR ".join(f"{f} LIKE ?" for f in srch["fields"])
                    sql += f" WHERE ({conditions})"
                    params = [like] * len(srch["fields"])
                sql += " ORDER BY created_at DESC"
                rows = conn.execute(sql, params).fetchall()
                all_rows.extend(dict(r) for r in rows)
            all_rows.sort(key=lambda x: str(x.get("created_at", "")), reverse=True)
            return all_rows[offset:offset + limit]
    except Exception as exc:
        logger.error("get_history failed: %s", exc)
        return []


def get_history_stats() -> dict:
    """统计各模块记录数"""
    try:
        with _cursor() as conn:
            a = conn.execute("SELECT count(*) FROM analyze_records").fetchone()[0]
            c = conn.execute("SELECT count(*) FROM classify_records").fetchone()[0]
            e = conn.execute("SELECT count(*) FROM extract_records").fetchone()[0]
            return {"analyze": a, "classify": c, "extract": e, "total": a + c + e}
    except Exception as exc:
        logger.error("get_history_stats failed: %s", exc)
        return {"analyze": 0, "classify": 0, "extract": 0, "total": 0}
