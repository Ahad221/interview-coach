---
title: Interview Coach AI
emoji: 🎯
colorFrom: purple
colorTo: indigo
sdk: streamlit
sdk_version: 1.32.0
app_file: app.py
pinned: false
license: mit
---

<div align="center">

# 🎯 InterviewAI — Multi-Agent Interview Coach

**AI-generated questions · STAR-format feedback · Session memory · Adaptive difficulty**

[![Hugging Face](https://img.shields.io/badge/🤗%20Live%20Demo-HuggingFace-yellow)](https://huggingface.co/spaces/abdulahad101/interview-coach)
[![GitHub](https://img.shields.io/badge/GitHub-interview--coach-black?logo=github)](https://github.com/Ahad221/interview-coach)
[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-red?logo=streamlit)](https://streamlit.io)

</div>

---

## 🚀 Overview

**InterviewAI** is a multi-agent coaching system that helps job seekers prepare for interviews with precision. Paste a job description, get role-specific questions tailored to your weak areas, answer them, and receive structured STAR-format feedback — all tracked across sessions so you improve over time.

### ✨ Key Features

| Feature | Description |
|--------|-------------|
| 📋 **Job Description Analysis** | Paste any job posting; AI extracts key requirements and tailors every question to that role |
| 🤖 **AI Question Generation** | `question_agent.py` generates role-specific questions that adapt to your difficulty level |
| ⭐ **STAR Feedback** | `evaluator_agent.py` scores your answers 1–10 and gives structured feedback |
| 🧠 **Session Memory** | `memory_agent.py` tracks score history, weak areas, and adjusts difficulty automatically |
| 🎙️ **Voice Input** | Whisper integration via `whisper_utils.py` — speak your answers instead of typing |
| 📈 **Progress Dashboard** | Sidebar shows sessions done, average score, difficulty level, and flagged weak areas |

---

## 🧠 Architecture

```
interview-coach/
├── app.py                    # Streamlit UI — main entry point
├── agents/
│   ├── question_agent.py     # Generates role-specific interview questions
│   ├── evaluator_agent.py    # Scores answers using STAR framework (1–10)
│   └── memory_agent.py       # Tracks history, weak areas, adaptive difficulty
├── utils/
│   ├── rag_utils.py          # RAG pipeline — embeds & retrieves JD context
│   └── whisper_utils.py      # OpenAI Whisper voice-to-text transcription
├── data/                     # Session data & vector store
├── .env                      # API keys (never commit this)
├── requirements.txt
└── README.md
```

### How It Works

```
Job Description (text)
        │
        ▼
  rag_utils.py ──► Embed & store JD context
        │
        ▼
question_agent.py ──► Generate targeted question
        │             (uses JD + weak areas + difficulty)
        ▼
  User answers (text or 🎙️ Whisper voice)
        │
        ▼
evaluator_agent.py ──► Score (1–10) + STAR feedback
        │
        ▼
 memory_agent.py ──► Update history, flag weak areas,
                     adjust difficulty for next round
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| UI | Streamlit |
| AI Agents | LangChain / OpenAI GPT |
| Voice Input | OpenAI Whisper |
| RAG Pipeline | FAISS + Sentence Transformers |
| Memory | JSON-based session persistence |
| Deployment | Hugging Face Spaces |

---

## 🖥️ Run Locally

```bash
# 1. Clone the repository
git clone https://github.com/Ahad221/interview-coach.git
cd interview-coach

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up environment variables
cp .env.example .env
# Edit .env and add your API keys

# 5. Run the app
streamlit run app.py
```

---

## 🔑 Environment Variables

Create a `.env` file in the root directory:

```env
OPENAI_API_KEY=your_openai_api_key_here
```

> ⚠️ Never commit your `.env` file. It is already in `.gitignore`.

**For Hugging Face Spaces:** Go to your Space → Settings → Repository secrets → add `OPENAI_API_KEY`.

---

## 📸 Screenshots

| Progress Sidebar | Question Generation |
|-----------------|-------------------|
| Sessions, score, difficulty, weak areas tracked per session | Role-specific questions adapt to your performance |

---

## 🤝 Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss what you'd like to change.

---

## 📄 License

[MIT](LICENSE)

---

<div align="center">
Made with ❤️ by <a href="https://github.com/Ahad221">Abdul Ahad</a>
</div>
