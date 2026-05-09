import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def generate_question(context: str, role: str, weak_areas: list, difficulty: str = "medium") -> str:
    weak_str = ", ".join(weak_areas) if weak_areas else "none identified yet"
    
    difficulty_instruction = {
        "easy": "Ask a simple, straightforward question suitable for a junior candidate.",
        "medium": "Ask a moderately challenging question requiring specific examples.",
        "hard": "Ask a complex, multi-part question requiring deep technical knowledge and leadership examples."
    }.get(difficulty, "medium")

    prompt = f"""
You are an expert technical interviewer for a {role} position.
Job context: {context}
Candidate's weak areas to target: {weak_str}
Difficulty level: {difficulty.upper()} — {difficulty_instruction}

Generate ONE targeted interview question based on the weak areas and difficulty.
Be concise. Return only the question, nothing else.
"""
    res = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}]
    )
    return res.choices[0].message.content.strip()