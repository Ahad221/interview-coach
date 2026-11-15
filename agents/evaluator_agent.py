import os
import json
from groq import Groq, BadRequestError
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")


def _call_model(prompt: str) -> str:
    messages = [{"role": "user", "content": prompt}]
    try:
        res = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=0.3,
            max_tokens=4000,
            reasoning_effort="low",
            response_format={"type": "json_object"},
        )
    except BadRequestError:
        # Fallback if the model rejects reasoning_effort / response_format
        res = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=0.3,
            max_tokens=4000,
        )
    return (res.choices[0].message.content or "").strip()


def _extract_json(raw: str) -> dict:
    raw = raw.replace("```json", "").replace("```", "").strip()
    start, end = raw.find("{"), raw.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("Model returned no JSON")
    return json.loads(raw[start:end + 1])


def _normalize(data: dict) -> dict:
    """Make sure every field the UI/memory expects exists."""
    data.setdefault("score", 0)
    data.setdefault("star_score", {})
    data.setdefault("soft_skills", {})
    data.setdefault("strengths", [])
    data.setdefault("weaknesses", [])
    data.setdefault("improved_answer", "")
    data.setdefault("filler_word_tip", "")
    return data


def evaluate_answer(question: str, answer: str, context: str) -> dict:
    prompt = f"""
You are a strict but fair interview coach.

Question: {question}
Candidate's Answer: {answer}
Job Context: {context}

Evaluate the answer. Return ONLY a JSON object with exactly these fields:

{{
    "score": <number 1-10>,
    "star_score": {{
        "situation": <number 1-10>,
        "task": <number 1-10>,
        "action": <number 1-10>,
        "result": <number 1-10>
    }},
    "soft_skills": {{
        "communication": <number 1-10>,
        "problem_solving": <number 1-10>,
        "leadership": <number 1-10>,
        "teamwork": <number 1-10>,
        "adaptability": <number 1-10>
    }},
    "strengths": ["<strength 1>", "<strength 2>"],
    "weaknesses": ["<weakness 1>", "<weakness 2>"],
    "improved_answer": "<a better version of their answer>",
    "filler_word_tip": "<tip about communication clarity>"
}}

Return ONLY the JSON. No markdown. No explanation.
"""

    last_error = None
    for _ in range(2):  # one automatic retry if the JSON is malformed
        try:
            return _normalize(_extract_json(_call_model(prompt)))
        except (ValueError, json.JSONDecodeError) as e:
            last_error = e
    raise last_error