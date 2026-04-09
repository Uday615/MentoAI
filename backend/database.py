import json
import os
import sys
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

try:
    import pymongo  # noqa: F401
except ImportError:
    backend_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(backend_dir)
    venv_site_packages = os.path.join(root_dir, "venv", "Lib", "site-packages")
    if os.path.exists(venv_site_packages):
        sys.path.insert(0, venv_site_packages)

try:
    from pymongo import DESCENDING, MongoClient  # type: ignore
except Exception:
    DESCENDING = -1
    MongoClient = None

MONGO_URI = os.environ.get(
    "MONGO_URI",
    "mongodb+srv://22501a4436_db_user:22501a4436@cluster0.5gqww92.mongodb.net/?appName=Cluster0",
)
DB_NAME = os.environ.get("MONGO_DB", "mentor_support_db")
LOCAL_STORE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "local_store.json")

COLLECTIONS = [
    "screening_sessions",
    "assessment_sessions",
    "feedback_sessions",
    "feedback_requests",
    "users",
]

_memory_store: Dict[str, List[Dict[str, Any]]] = {name: [] for name in COLLECTIONS}
_USE_LOCAL = False
_DB_INIT_ATTEMPTED = False
_client = None
_db = None


def _ensure_local_store() -> None:
    if not os.path.exists(LOCAL_STORE_PATH):
        _write_local_store({name: [] for name in COLLECTIONS})


def _read_local_store() -> Dict[str, List[Dict[str, Any]]]:
    _ensure_local_store()
    with open(LOCAL_STORE_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    for name in COLLECTIONS:
        data.setdefault(name, [])
    return data


def _write_local_store(data: Dict[str, List[Dict[str, Any]]]) -> None:
    with open(LOCAL_STORE_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def _sync_memory_from_local() -> None:
    global _memory_store
    _memory_store = _read_local_store()


def _persist_memory_to_local() -> None:
    _write_local_store(_memory_store)


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def get_db():
    global _client, _db, _USE_LOCAL, _DB_INIT_ATTEMPTED
    if _db is not None or (_DB_INIT_ATTEMPTED and _USE_LOCAL):
        return _db

    _DB_INIT_ATTEMPTED = True
    if MongoClient is None:
        _USE_LOCAL = True
        _sync_memory_from_local()
        print("[Storage] pymongo unavailable. Using local JSON persistence.")
        return None

    try:
        _client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
        _client.admin.command("ping")
        _db = _client[DB_NAME]
        _ensure_indexes(_db)
        _USE_LOCAL = False
        print(f"[MongoDB] Connected to database: '{DB_NAME}'")
    except Exception as e:
        print(f"[MongoDB] Connection failed: {e}")
        print(f"[Storage] Falling back to local JSON persistence at {LOCAL_STORE_PATH}")
        _client = None
        _db = None
        _USE_LOCAL = True
        _sync_memory_from_local()
    return _db


def _ensure_indexes(db):
    try:
        db["screening_sessions"].create_index([("created_at", DESCENDING)])
        db["screening_sessions"].create_index([("student_id", DESCENDING)])
        db["assessment_sessions"].create_index([("created_at", DESCENDING)])
        db["assessment_sessions"].create_index([("student_id", DESCENDING)])
        db["assessment_sessions"].create_index([("risk_label", DESCENDING)])
        db["feedback_sessions"].create_index([("created_at", DESCENDING)])
        db["feedback_sessions"].create_index([("student_id", DESCENDING)])
        db["feedback_requests"].create_index([("token", DESCENDING)], unique=True)
        db["users"].create_index([("email", DESCENDING)], unique=True)
        print("[MongoDB] Indexes verified.")
    except Exception as e:
        print(f"[MongoDB] Index creation warning: {e}")


def is_connected() -> bool:
    return get_db() is not None


def using_memory() -> bool:
    get_db()
    return _USE_LOCAL


def _serialize_docs(docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    serialized: List[Dict[str, Any]] = []
    for doc in docs:
        item = {k: v for k, v in doc.items() if k != "_id"}
        serialized.append(item)
    return serialized


def _local_insert(collection: str, doc: Dict[str, Any]) -> str:
    doc_id = str(uuid.uuid4())
    stored = dict(doc)
    stored["_id"] = doc_id
    _memory_store[collection].append(stored)
    _persist_memory_to_local()
    return doc_id


def _local_find_one(collection: str, predicate) -> Optional[Dict[str, Any]]:
    for item in _memory_store[collection]:
        if predicate(item):
            return item
    return None


def _local_find_many(collection: str, predicate) -> List[Dict[str, Any]]:
    return [item for item in _memory_store[collection] if predicate(item)]


def _local_update_one(collection: str, predicate, updates: Dict[str, Any]) -> bool:
    for item in _memory_store[collection]:
        if predicate(item):
            item.update(updates)
            _persist_memory_to_local()
            return True
    return False


def create_user(email: str, password_hash: str, name: str = "") -> Optional[str]:
    db = get_db()

    if db is None:
        if _local_find_one("users", lambda item: item["email"] == email):
            return None
        user_id = _local_insert(
            "users",
            {
                "email": email,
                "password_hash": password_hash,
                "name": name,
                "created_at": _utcnow_iso(),
            },
        )
        print(f"[LocalStorage] User created: {email}")
        return user_id

    try:
        if db["users"].find_one({"email": email}):
            return None
        res = db["users"].insert_one(
            {
                "email": email,
                "password_hash": password_hash,
                "name": name,
                "created_at": _utcnow_iso(),
            }
        )
        return str(res.inserted_id)
    except Exception as e:
        print(f"[MongoDB] create_user error: {e}")
        return None


def get_user_by_email(email: str) -> Optional[Dict[str, Any]]:
    db = get_db()

    if db is None:
        return _local_find_one("users", lambda item: item["email"] == email)

    try:
        return db["users"].find_one({"email": email})
    except Exception as e:
        print(f"[MongoDB] get_user_by_email error: {e}")
        return None


def save_screening_session(payload: Dict[str, Any], result: Dict[str, Any]) -> Optional[str]:
    doc = {
        "student_name": payload.get("student_name", "Anonymous"),
        "student_id": payload.get("student_id", ""),
        "screening_answers": payload.get("answers", []),
        "risk_label": result.get("risk_label"),
        "risk_class": result.get("risk_class"),
        "total_score": result.get("total_score"),
        "max_score": result.get("max_score"),
        "proceed_to_stage2": result.get("proceed_to_stage2"),
        "triage_message": result.get("triage_message"),
        "domain_summary": result.get("domain_summary", {}),
        "stage": "Stage 1 - Weekly Screening",
        "created_at": _utcnow_iso(),
    }

    db = get_db()
    if db is None:
        doc_id = _local_insert("screening_sessions", doc)
        print(f"[LocalStorage] Screening session saved: {doc_id}")
        return doc_id

    try:
        inserted = db["screening_sessions"].insert_one(doc)
        return str(inserted.inserted_id)
    except Exception as e:
        print(f"[MongoDB] Failed to save screening session: {e}")
        return None


def save_assessment_session(payload: Dict[str, Any], result: Dict[str, Any]) -> Optional[str]:
    doc = {
        "student_name": payload.get("student_name", "Anonymous"),
        "student_id": payload.get("student_id", ""),
        "programme": payload.get("programme", ""),
        "assessment_answers": payload.get("answers", []),
        "risk_label": result.get("risk_label"),
        "risk_class": result.get("risk_class"),
        "hybrid_decision": result.get("hybrid_decision", {}),
        "xai": {
            "top_indicators": result.get("xai", {}).get("top_indicators", []),
            "domain_scores": result.get("xai", {}).get("domain_scores", {}),
            "all_shap_values": result.get("xai", {}).get("all_shap_values", []),
            "consistency_check": result.get("xai", {}).get("consistency_check", {}),
        },
        "rag": {
            "suggested_action": result.get("rag", {}).get("suggested_action", ""),
            "monitoring_recommendation": result.get("rag", {}).get("monitoring_recommendation", ""),
            "knowledge_guidance": result.get("rag", {}).get("knowledge_guidance", ""),
            "llm_guidance": result.get("rag", {}).get("llm_guidance", ""),
            "retrieved_context": result.get("rag", {}).get("retrieved_context", ""),
            "verification": result.get("rag", {}).get("verification", {}),
        },
        "stage": "Stage 2 - Detailed ML Assessment",
        "created_at": _utcnow_iso(),
    }

    db = get_db()
    if db is None:
        doc_id = _local_insert("assessment_sessions", doc)
        print(f"[LocalStorage] Assessment session saved: {doc_id}")
        return doc_id

    try:
        inserted = db["assessment_sessions"].insert_one(doc)
        return str(inserted.inserted_id)
    except Exception as e:
        print(f"[MongoDB] Failed to save assessment session: {e}")
        return None


def save_feedback_session(payload: Dict[str, Any]) -> Optional[str]:
    doc = {
        "student_name": payload.get("student_name", "Anonymous"),
        "student_id": payload.get("student_id", ""),
        "programme": payload.get("programme", ""),
        "assessment_session_id": payload.get("assessment_session_id", ""),
        "feedback_request_token": payload.get("feedback_request_token", ""),
        "mentor_email": payload.get("mentor_email", ""),
        "improvement_status": payload.get("improvement_status", "unsure"),
        "support_helpfulness": payload.get("support_helpfulness", "unknown"),
        "follow_up_needed": bool(payload.get("follow_up_needed", False)),
        "student_feedback_summary": payload.get("student_feedback_summary", ""),
        "mentor_follow_up_notes": payload.get("mentor_follow_up_notes", ""),
        "submitted_by": payload.get("submitted_by", "student"),
        "stage": "Post-Intervention Feedback",
        "created_at": _utcnow_iso(),
    }

    db = get_db()
    if db is None:
        doc_id = _local_insert("feedback_sessions", doc)
        print(f"[LocalStorage] Feedback session saved: {doc_id}")
        return doc_id

    try:
        inserted = db["feedback_sessions"].insert_one(doc)
        return str(inserted.inserted_id)
    except Exception as e:
        print(f"[MongoDB] Failed to save feedback session: {e}")
        return None


def create_feedback_request(payload: Dict[str, Any]) -> Dict[str, Any]:
    token = uuid.uuid4().hex
    doc = {
        "token": token,
        "student_name": payload.get("student_name", "Anonymous"),
        "student_id": payload.get("student_id", ""),
        "student_email": payload.get("student_email", ""),
        "programme": payload.get("programme", ""),
        "assessment_session_id": payload.get("assessment_session_id", ""),
        "mentor_email": payload.get("mentor_email", ""),
        "status": "pending",
        "created_at": _utcnow_iso(),
        "completed_at": "",
    }

    db = get_db()
    if db is None:
        doc_id = _local_insert("feedback_requests", doc)
        return {**doc, "id": doc_id}

    inserted = db["feedback_requests"].insert_one(doc)
    return {**doc, "id": str(inserted.inserted_id)}


def get_feedback_request_by_token(token: str) -> Optional[Dict[str, Any]]:
    db = get_db()
    if db is None:
        item = _local_find_one("feedback_requests", lambda row: row.get("token") == token)
        if not item:
            return None
        return {k: v for k, v in item.items() if k != "_id"}

    try:
        item = db["feedback_requests"].find_one({"token": token}, {"_id": 0})
        return item
    except Exception as e:
        print(f"[MongoDB] Feedback request lookup error: {e}")
        return None


def complete_feedback_request(token: str) -> bool:
    db = get_db()
    updates = {"status": "completed", "completed_at": _utcnow_iso()}
    if db is None:
        return _local_update_one("feedback_requests", lambda row: row.get("token") == token, updates)

    try:
        result = db["feedback_requests"].update_one({"token": token}, {"$set": updates})
        return result.modified_count > 0
    except Exception as e:
        print(f"[MongoDB] Feedback request update error: {e}")
        return False


def get_recent_assessments(limit: int = 20) -> List[Dict[str, Any]]:
    db = get_db()
    if db is None:
        sessions = sorted(_memory_store["assessment_sessions"], key=lambda row: row.get("created_at", ""), reverse=True)[:limit]
        return _serialize_docs(sessions)

    try:
        cursor = db["assessment_sessions"].find({}, {"_id": 0}).sort("created_at", DESCENDING).limit(limit)
        return _serialize_docs(list(cursor))
    except Exception as e:
        print(f"[MongoDB] Query error: {e}")
        return []


def get_student_history(student_id: str) -> List[Dict[str, Any]]:
    db = get_db()
    if db is None:
        sessions = sorted(
            _local_find_many("assessment_sessions", lambda row: row.get("student_id") == student_id),
            key=lambda row: row.get("created_at", ""),
            reverse=True,
        )
        return _serialize_docs(sessions)

    try:
        cursor = db["assessment_sessions"].find({"student_id": student_id}, {"_id": 0}).sort("created_at", DESCENDING)
        return _serialize_docs(list(cursor))
    except Exception as e:
        print(f"[MongoDB] Student history query error: {e}")
        return []


def get_feedback_history(student_id: str) -> List[Dict[str, Any]]:
    db = get_db()
    if db is None:
        sessions = sorted(
            _local_find_many("feedback_sessions", lambda row: row.get("student_id") == student_id),
            key=lambda row: row.get("created_at", ""),
            reverse=True,
        )
        return _serialize_docs(sessions)

    try:
        cursor = db["feedback_sessions"].find({"student_id": student_id}, {"_id": 0}).sort("created_at", DESCENDING)
        return _serialize_docs(list(cursor))
    except Exception as e:
        print(f"[MongoDB] Feedback history query error: {e}")
        return []


def get_db_stats() -> Dict[str, Any]:
    db = get_db()
    if db is None:
        return {
            "connected": False,
            "storage": f"local-json ({LOCAL_STORE_PATH})",
            "screening_sessions": len(_memory_store["screening_sessions"]),
            "assessment_sessions": len(_memory_store["assessment_sessions"]),
            "feedback_sessions": len(_memory_store["feedback_sessions"]),
            "feedback_requests": len(_memory_store["feedback_requests"]),
            "users": len(_memory_store["users"]),
        }

    try:
        return {
            "connected": True,
            "storage": "MongoDB Atlas",
            "database": DB_NAME,
            "screening_sessions": db["screening_sessions"].count_documents({}),
            "assessment_sessions": db["assessment_sessions"].count_documents({}),
            "feedback_sessions": db["feedback_sessions"].count_documents({}),
            "feedback_requests": db["feedback_requests"].count_documents({}),
            "users": db["users"].count_documents({}),
        }
    except Exception as e:
        return {"connected": False, "error": str(e)}
