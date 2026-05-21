from __future__ import annotations

import hashlib
import secrets
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def get_db_path(base_dir: Path | None = None) -> Path:
    base = Path(base_dir or Path(__file__).resolve().parent)
    return base / "archvision.db"


def get_connection(base_dir: Path | None = None) -> sqlite3.Connection:
    conn = sqlite3.connect(get_db_path(base_dir))
    conn.row_factory = sqlite3.Row
    return conn


def init_db(base_dir: Path | None = None) -> None:
    conn = get_connection(base_dir)
    cur = conn.cursor()

    # Users
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            salt BLOB NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )

    # Query history
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS query_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            image_name TEXT,
            image_thumbnail TEXT,
            architectural_style TEXT,
            confidence REAL,
            ai_analysis TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
        """
    )

    # Architectural preferences
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS architectural_preferences (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            style_name TEXT,
            preference_score REAL DEFAULT 1.0,
            last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
        """
    )

    conn.commit()
    conn.close()


# Auth helpers
def hash_password(password: str, salt: bytes) -> str:
    return hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 100_000).hex()


def _normalize_salt(salt: Any) -> bytes:
    if isinstance(salt, bytes):
        return salt
    if isinstance(salt, str):
        try:
            return bytes.fromhex(salt)
        except ValueError:
            return salt.encode("utf-8")
    raise TypeError("Unsupported salt type")


def verify_password(password: str, salt: Any, hashed: str) -> bool:
    salt_bytes = _normalize_salt(salt)
    return hash_password(password, salt_bytes) == hashed


def create_user(username: str, password: str, base_dir: Path | None = None) -> Tuple[bool, str]:
    conn = get_connection(base_dir)
    cur = conn.cursor()

    cur.execute("SELECT id FROM users WHERE username = ?", (username,))
    if cur.fetchone():
        conn.close()
        return False, "Користувач уже існує"

    salt = secrets.token_bytes(32)
    pwd_hash = hash_password(password, salt)

    cur.execute(
        "INSERT INTO users (username, password_hash, salt) VALUES (?, ?, ?)",
        (username, pwd_hash, salt),
    )
    conn.commit()
    conn.close()
    return True, "Реєстрація успішна"


def authenticate_user(username: str, password: str, base_dir: Path | None = None) -> Optional[Dict[str, Any]]:
    conn = get_connection(base_dir)
    cur = conn.cursor()

    cur.execute(
        "SELECT id, username, password_hash, salt FROM users WHERE username = ?",
        (username,),
    )
    row = cur.fetchone()
    conn.close()

    if not row:
        return None

    if not verify_password(password, row["salt"], row["password_hash"]):
        return None

    return {"id": row["id"], "username": row["username"]}


# Query history / analytics
def save_query_history(
    user_id: int,
    image_name: str,
    image_thumbnail: str,
    architectural_style: str,
    confidence: float,
    ai_analysis: str,
    base_dir: Path | None = None,
) -> int:
    conn = get_connection(base_dir)
    cur = conn.cursor()

    cur.execute(
        """
        INSERT INTO query_history
        (user_id, image_name, image_thumbnail, architectural_style, confidence, ai_analysis)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (user_id, image_name, image_thumbnail, architectural_style, confidence, ai_analysis),
    )

    history_id = int(cur.lastrowid)

    cur.execute(
        "SELECT id, preference_score FROM architectural_preferences WHERE user_id = ? AND style_name = ?",
        (user_id, architectural_style),
    )
    pref = cur.fetchone()

    if pref:
        cur.execute(
            """
            UPDATE architectural_preferences
            SET preference_score = ?, last_updated = CURRENT_TIMESTAMP
            WHERE id = ?
            """,
            (float(pref["preference_score"]) + 1.0, pref["id"]),
        )
    else:
        cur.execute(
            """
            INSERT INTO architectural_preferences (user_id, style_name, preference_score)
            VALUES (?, ?, 1.0)
            """,
            (user_id, architectural_style),
        )

    conn.commit()
    conn.close()

    return history_id

def update_query_history_ai_analysis(
    user_id: int,
    history_id: int,
    ai_analysis: str,
    base_dir: Path | None = None,
) -> bool:
    conn = get_connection(base_dir)
    cur = conn.cursor()

    cur.execute(
        """
        UPDATE query_history
        SET ai_analysis = ?
        WHERE id = ? AND user_id = ?
        """,
        (ai_analysis, history_id, user_id),
    )

    updated = cur.rowcount > 0

    conn.commit()
    conn.close()

    return updated


def get_user_history(user_id: int, base_dir: Path | None = None, limit: int = 20) -> List[Dict[str, Any]]:
    conn = get_connection(base_dir)
    cur = conn.cursor()

    cur.execute(
        """
        SELECT id, image_name, image_thumbnail, architectural_style, confidence, ai_analysis, created_at
        FROM query_history
        WHERE user_id = ?
        ORDER BY created_at DESC
        LIMIT ?
        """,
        (user_id, limit),
    )
    rows = cur.fetchall()
    conn.close()

    return [dict(r) for r in rows]


def get_user_stats(user_id: int, base_dir: Path | None = None) -> Dict[str, Any]:
    conn = get_connection(base_dir)
    cur = conn.cursor()

    cur.execute("SELECT COUNT(*) AS total FROM query_history WHERE user_id = ?", (user_id,))
    total = cur.fetchone()["total"]

    cur.execute(
        """
        SELECT architectural_style, COUNT(*) AS count
        FROM query_history
        WHERE user_id = ?
        GROUP BY architectural_style
        ORDER BY count DESC
        LIMIT 5
        """,
        (user_id,),
    )
    rows = [dict(r) for r in cur.fetchall()]

    cur.execute(
        """
        SELECT AVG(confidence) AS avg_confidence
        FROM query_history
        WHERE user_id = ?
        """,
        (user_id,),
    )
    avg_conf = cur.fetchone()["avg_confidence"] or 0

    conn.close()

    popular_styles = [
        {"style": r["architectural_style"], "count": r["count"]}
        for r in rows
        if r.get("architectural_style")
    ]

    return {
        "total_analyses": total,
        "popular_styles": popular_styles,
        "favorite_style": popular_styles[0]["style"] if popular_styles else None,
        "avg_confidence": round(float(avg_conf), 3) if avg_conf else 0.0,
    }


def get_user_preferences(user_id: int, base_dir: Path | None = None) -> List[Dict[str, Any]]:
    conn = get_connection(base_dir)
    cur = conn.cursor()

    cur.execute(
        """
        SELECT style_name, preference_score, last_updated
        FROM architectural_preferences
        WHERE user_id = ?
        ORDER BY preference_score DESC, last_updated DESC
        LIMIT 20
        """,
        (user_id,),
    )
    rows = cur.fetchall()
    conn.close()

    return [dict(r) for r in rows]