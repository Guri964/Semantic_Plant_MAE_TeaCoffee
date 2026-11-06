from groq import Groq
from config import GROQ_API_KEY, GROQ_MODEL


def get_recommendation(crop, health, disease, severity, colors):
    if not GROQ_API_KEY:
        return 'LLM service not configured. Please add GROQ_API_KEY to .env file.'

    client = Groq(api_key=GROQ_API_KEY)

    if health == 'healthy':
        prompt = f"""A {crop} leaf has been identified as healthy with no signs of disease.
Provide a brief 2-3 sentence practical recommendation for maintaining the health of this {crop} plant.
Use simple farmer-friendly language."""
    elif disease in ('Condition Not Identified', 'No Disease Detected'):
        prompt = f"""A {crop} leaf looks unhealthy, but the specific disease could not be identified from the photo.
Provide a brief 2-3 sentence practical recommendation: general care steps and advice to consult a local agricultural expert with a clearer photo.
Use simple farmer-friendly language. Do not guess a specific disease."""
    else:
        color_str = ', '.join(colors) if colors else 'none detected'
        prompt = f"""A {crop} leaf has been diagnosed with {disease}.
Severity level: {severity}
Colors detected in the diseased area: {color_str}

Provide a practical 3-4 sentence treatment recommendation for a farmer.
Be specific, actionable, and use simple language. No bullet points."""

    try:
        completion = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{'role': 'user', 'content': prompt}],
            temperature=0.7,
            max_completion_tokens=200,
            top_p=1,
            stream=False,
            stop=None
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        return f'Recommendation unavailable: {str(e)}'