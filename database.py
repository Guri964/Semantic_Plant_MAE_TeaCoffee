import sqlite3
import bcrypt
from config import DATABASE_PATH


def get_connection():
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            username      TEXT    NOT NULL UNIQUE,
            password_hash TEXT    NOT NULL,
            created_at    DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS prediction_logs (
            id                INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id           INTEGER NOT NULL,
            image_path        TEXT,
            predicted_crop    TEXT,
            predicted_health  TEXT,
            predicted_pattern TEXT,
            detected_colors   TEXT,
            predicted_disease TEXT,
            severity          TEXT,
            severity_pct      REAL,
            match_score       REAL,
            recommendation    TEXT,
            created_at        DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    ''')

    conn.commit()
    conn.close()


def get_user_by_username(username):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM users WHERE username = ?', (username,))
    user = cursor.fetchone()
    conn.close()
    return user


def create_user(username, password):
    password_hash = bcrypt.hashpw(
        password.encode('utf-8'), bcrypt.gensalt()
    ).decode('utf-8')
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        'INSERT INTO users (username, password_hash) VALUES (?, ?)',
        (username, password_hash)
    )
    conn.commit()
    conn.close()


def verify_password(password, password_hash):
    return bcrypt.checkpw(
        password.encode('utf-8'),
        password_hash.encode('utf-8')
    )


def log_prediction(user_id, image_path, predicted_crop, predicted_health,
                   predicted_pattern, detected_colors, predicted_disease,
                   severity, severity_pct, match_score, recommendation):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO prediction_logs
        (user_id, image_path, predicted_crop, predicted_health,
         predicted_pattern, detected_colors, predicted_disease,
         severity, severity_pct, match_score, recommendation)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (user_id, image_path, predicted_crop, predicted_health,
          predicted_pattern, ','.join(detected_colors), predicted_disease,
          severity, severity_pct, match_score, recommendation))
    conn.commit()
    conn.close()