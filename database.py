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

    # The knowledge base was originally created only inside the training
    # notebook's Kaggle database (gitignored), so a fresh checkout had no
    # knowledge_base table and /predict crashed with "no such table".
    # Create it here with the same schema and seed it with the notebook's
    # final content — but only when empty, so existing deployments keep
    # their data untouched.
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS knowledge_base (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            crop        TEXT NOT NULL,
            disease     TEXT NOT NULL,
            health      TEXT NOT NULL,
            pattern     TEXT,
            colors      TEXT NOT NULL,
            description TEXT,
            treatment   TEXT
        )
    ''')

    cursor.execute('SELECT COUNT(*) FROM knowledge_base')
    if cursor.fetchone()[0] == 0:
        cursor.executemany('''
            INSERT INTO knowledge_base
            (crop, disease, health, pattern, colors, description, treatment)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', _KNOWLEDGE_BASE_SEED)

    conn.commit()
    conn.close()


# Exact knowledge-base content from the final version of the training
# notebook (models/plant-me.ipynb, "Visual Check of Unhealthy Images Before
# Pattern Assignment" cell). The 'pattern' column is retained for schema
# compatibility; the live API now derives the displayed pattern from image
# evidence (services/pattern_detection.py) rather than this static column.
_KNOWLEDGE_BASE_SEED = [
    ('coffee', 'Coffee Leaf Rust', 'unhealthy', 'spots',
     'brown,yellow',
     'Fungal disease causing circular yellow-brown spots on upper leaf surface with orange pustules underneath.',
     'Apply copper-based fungicides. Remove infected leaves. Improve air circulation.'),

    ('coffee', 'Coffee Phoma Blight', 'unhealthy', 'spots',
     'brown,black',
     'Fungal disease causing dark brown to black circular necrotic lesions with defined margins.',
     'Apply mancozeb or carbendazim. Prune infected branches. Avoid waterlogging.'),

    ('coffee', 'Coffee Chlorosis', 'unhealthy', 'mosaic',
     'yellow,white',
     'Nutrient deficiency causing irregular yellow-white mottled patches between leaf veins.',
     'Apply chelated iron and magnesium fertilizer. Test soil pH. Improve drainage.'),

    ('coffee', 'Coffee Cercospora Leaf Spot', 'unhealthy', 'spots',
     'brown,gray',
     'Fungal disease causing circular spots with brown borders and gray centers on leaves.',
     'Apply copper fungicides. Remove infected leaves. Maintain proper plant spacing.'),

    ('coffee', 'Coffee Sooty Mold', 'unhealthy', 'powdery',
     'black,gray',
     'Fungal growth forming black powdery coating across leaf surface due to insect honeydew.',
     'Control insect populations. Wash leaves with soapy water. Improve ventilation.'),

    ('coffee', 'Coffee Bacterial Blight', 'unhealthy', 'spots',
     'brown,black,gray',
     'Bacterial infection causing water-soaked spots that turn brown and black scattered across leaf.',
     'Apply copper bactericides. Avoid overhead irrigation. Remove infected material.'),

    ('coffee', 'Healthy Coffee Leaf', 'healthy', 'none',
     '',
     'Leaf shows no signs of disease. Plant is in good health.',
     'Continue current care. Maintain regular fertilization and irrigation.'),

    ('tea', 'Tea Blister Blight', 'unhealthy', 'powdery',
     'white,gray',
     'Fungal disease causing raised white blister pustules on young tea leaves — powdery white patches on surface.',
     'Apply copper oxychloride. Avoid working in wet conditions. Improve air circulation.'),

    ('tea', 'Tea Red Spot', 'unhealthy', 'spots',
     'brown,black',
     'Fungal disease causing small reddish-brown circular spots scattered across mature tea leaves.',
     'Apply mancozeb fungicides. Remove infected leaves. Improve plantation drainage.'),

    ('tea', 'Tea Algal Leaf Spot', 'unhealthy', 'spots',
     'gray,white',
     'Algal infection causing raised circular crusty gray-white spots on tea leaf surface.',
     'Apply copper fungicides. Improve air circulation. Reduce shade and humidity.'),

    ('tea', 'Tea Gray Blight', 'unhealthy', 'spots',
     'gray,brown',
     'Fungal disease causing gray-brown necrotic spots that expand and merge across tea leaves.',
     'Apply carbendazim or mancozeb. Remove infected leaves. Maintain plantation hygiene.'),

    ('tea', 'Tea Brown Blight', 'unhealthy', 'spots',
     'brown,black',
     'Fungal disease causing brown water-soaked circular lesions that darken and spread rapidly.',
     'Apply copper-based fungicides immediately. Remove infected branches. Avoid wet conditions.'),

    ('tea', 'Tea White Spot', 'unhealthy', 'spots',
     'white,black',
     'Fungal disease causing circular white spots with dark borders on tea leaves.',
     'Apply hexaconazole fungicide. Remove infected debris. Improve drainage.'),

    ('tea', 'Tea Anthracnose', 'unhealthy', 'spots',
     'brown,black,gray',
     'Fungal disease causing dark sunken circular lesions with yellow halos on tea leaves.',
     'Apply mancozeb or copper fungicides. Prune infected parts. Avoid waterlogging.'),

    ('tea', 'Tea Chlorosis', 'unhealthy', 'mosaic',
     'yellow,white',
     'Nutrient deficiency causing irregular yellow-white mottled discoloration between leaf veins.',
     'Apply chelated iron or magnesium sulfate. Adjust soil pH. Improve drainage.'),

    ('tea', 'Healthy Tea Leaf', 'healthy', 'none',
     '',
     'Leaf shows no signs of disease. Plant is in good health.',
     'Continue current care. Monitor regularly for early signs of disease.'),
]


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