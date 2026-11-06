import sqlite3

from config import DATABASE_PATH, KB_WEIGHTS, KB_MIN_SCORE


def _inconclusive_result(best_score, reason=None):
    return {
        'disease':     'Condition Not Identified',
        'pattern':     'inconclusive',
        'description': reason or 'No matching condition found in the knowledge base.',
        'treatment':   'Please consult an agricultural expert.',
        'match_score': round(best_score, 4),
    }


def match_disease(predicted_crop, predicted_health, detected_colors):
    """Match crop/health/color evidence against the knowledge base.

    Fixes vs the legacy version:
      * Healthy predictions only ever match healthy rows (a healthy leaf can
        no longer be assigned a disease because of stray color hits).
      * Unhealthy predictions require at least one supporting color overlap
        to name a *specific* disease; otherwise the result is inconclusive
        rather than the first row that happens to win a crop+health tie.
      * A missing knowledge_base table degrades to an inconclusive result
        instead of raising sqlite3.OperationalError.

    The returned ``pattern`` is the KB row's static pattern, kept for schema
    compatibility. Callers should prefer the evidence-based pattern from
    services.pattern_detection.detect_pattern for display.
    """
    try:
        conn = sqlite3.connect(DATABASE_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM knowledge_base')
        rows = cursor.fetchall()
        conn.close()
    except sqlite3.OperationalError:
        return _inconclusive_result(
            0.0, 'Knowledge base is unavailable; cannot identify a condition.'
        )

    detected = {c.strip().lower() for c in (detected_colors or []) if c.strip()}

    best_score = 0.0
    best_match = None
    best_color_overlap = 0

    for row in rows:
        row_crop = row['crop'].lower()
        row_health = row['health'].lower()

        # Health consistency: never cross-match healthy <-> unhealthy.
        if row_health != predicted_health.lower():
            continue
        if row_crop != predicted_crop.lower():
            continue

        score = KB_WEIGHTS['crop'] + KB_WEIGHTS['health']

        row_colors = {c.strip().lower() for c in row['colors'].split(',') if c.strip()}
        overlap = len(detected & row_colors)
        if detected and row_colors:
            score += KB_WEIGHTS['color'] * (overlap / len(row_colors))
        elif not detected and not row_colors:
            score += KB_WEIGHTS['color']

        if score > best_score:
            best_score = score
            best_match = row
            best_color_overlap = overlap

    if best_match is None or best_score < KB_MIN_SCORE:
        return _inconclusive_result(best_score)

    # Naming a specific disease requires visual support. For unhealthy leaves
    # with no color evidence, every disease row ties on crop+health alone;
    # picking one would invent a diagnosis.
    if predicted_health.lower() == 'unhealthy' and best_color_overlap == 0:
        return _inconclusive_result(
            best_score,
            'Leaf shows signs of poor health, but no specific disease pattern '
            'has enough color evidence to identify the condition.',
        )

    return {
        'disease':     best_match['disease'],
        'pattern':     best_match['pattern'],
        'description': best_match['description'],
        'treatment':   best_match['treatment'],
        'match_score': round(best_score, 4),
    }
