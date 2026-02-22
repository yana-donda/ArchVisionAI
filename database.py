import os
import sqlite3
import hashlib
from functools import wraps

from flask import session, jsonify

DB_PATH = "archvision.db"

def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()

    # Users
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash BLOB,
            salt BLOB,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Analysis history
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS analysis_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            image_name TEXT,
            predicted_style TEXT,
            confidence REAL,
            top_predictions TEXT,
            gemini_analysis TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    """)

    # Architectural preferences
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS user_preferences (
            user_id INTEGER PRIMARY KEY,
            theme TEXT DEFAULT 'light',
            language TEXT DEFAULT 'uk',
            notifications INTEGER DEFAULT 1,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    """)

    conn.commit()
    conn.close()
    
# Auth helpers
def hash_password(password: str, salt: bytes) -> bytes:
    return hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 100000)


def verify_password(password: str, salt: bytes, stored_hash: bytes) -> bool:
    return hash_password(password, salt) == stored_hash


def create_user(username: str, password: str):
    if not username or not password:
        return False, "Username and password are required"

    username = username.strip()

    if len(username) < 3:
        return False, "Username must be at least 3 characters"

    if len(password) < 6:
        return False, "Password must be at least 6 characters"

    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT id FROM users WHERE username = ?", (username,))
    if cursor.fetchone():
        conn.close()
        return False, "User already exists"

    salt = os.urandom(32)
    password_hash = hash_password(password, salt)

    cursor.execute(
        "INSERT INTO users (username, password_hash, salt) VALUES (?, ?, ?)",
        (username, password_hash, salt)
    )

    conn.commit()
    user_id = cursor.lastrowid
    conn.close()

    return True, {
        "id": user_id,
        "username": username
    }


def authenticate_user(username: str, password: str):
    if not username or not password:
        return None

    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute(
        "SELECT id, username, password_hash, salt FROM users WHERE username = ?",
        (username.strip(),)
    )
    row = cursor.fetchone()
    conn.close()

    if not row:
        return None

    if not row["password_hash"] or not row["salt"]:
        return None

    if not verify_password(password, row["salt"], row["password_hash"]):
        return None

    return {
        "id": row["id"],
        "username": row["username"]
    }


def create_test_users():
    """Тестовий користувач для швидкого входу"""
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT id FROM users WHERE username = ?", ("demo",))
    exists = cursor.fetchone()
    conn.close()

    if not exists:
        create_user("demo", "demo123")


def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if "user_id" not in session:
            return jsonify({"error": "Authentication required"}), 401
        return f(*args, **kwargs)
    return decorated