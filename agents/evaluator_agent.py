import os
import json
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

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
    res = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}]
    )
    raw = res.choices[0].message.content.strip()
    raw = raw.replace("```json", "").replace("```", "").strip()
    return json.loads(raw)