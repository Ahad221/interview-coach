import os
from groq import Groq, BadRequestError
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")


def generate_question(context, role, weak_areas, difficulty):
    prompt = f"""
You are an expert technical interviewer.

Generate ONE interview question based on:

Candidate Context: {context}
Job Role: {role}
Weak Areas: {weak_areas}
Difficulty: {difficulty}

Rules:
- Return only the question text, nothing else.
- Make it relevant, clear and professional.
"""

    messages = [{"role": "user", "content": prompt}]

    # gpt-oss is a reasoning model: thinking tokens count toward max_tokens,
    # so keep the limit generous and the reasoning effort low.
    try:
        res = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=0.7,
            max_tokens=2000,
            reasoning_effort="low",
        )
    except BadRequestError:
        # Fallback if the chosen model doesn't support reasoning_effort
        res = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=0.7,
            max_tokens=2000,
        )

    text = (res.choices[0].message.content or "").strip()
    if not text:
        raise ValueError("Model returned an empty question")
    return text