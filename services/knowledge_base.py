import sqlite3
from config import DATABASE_PATH, KB_WEIGHTS, KB_MIN_SCORE


def match_disease(predicted_crop, predicted_health, detected_colors):
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM knowledge_base')
    rows = cursor.fetchall()
    conn.close()

    best_score = 0.0
    best_match = None

    for row in rows:
        score = 0.0

        if predicted_crop.lower() == row['crop'].lower():
            score += KB_WEIGHTS['crop']

        if predicted_health.lower() == row['health'].lower():
            score += KB_WEIGHTS['health']

        row_colors = [c.strip().lower() for c in row['colors'].split(',') if c.strip()]
        if detected_colors and row_colors:
            overlap     = len(set(detected_colors) & set(row_colors))
            color_score = overlap / len(row_colors)
            score += KB_WEIGHTS['color'] * color_score
        elif not detected_colors and not row_colors:
            score += KB_WEIGHTS['color']

        if score > best_score:
            best_score = score
            best_match = row

    if best_score < KB_MIN_SCORE or best_match is None:
        return {
            'disease':     'Condition Not Identified',
            'pattern':     'unknown',
            'description': 'No matching condition found in the knowledge base.',
            'treatment':   'Please consult an agricultural expert.',
            'match_score': round(best_score, 4)
        }

    return {
        'disease':     best_match['disease'],
        'pattern':     best_match['pattern'],
        'description': best_match['description'],
        'treatment':   best_match['treatment'],
        'match_score': round(best_score, 4)
    }