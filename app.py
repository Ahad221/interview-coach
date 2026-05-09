import streamlit as st
import plotly.graph_objects as go
from dotenv import load_dotenv
from utils.rag_utils import embed_jd, retrieve_context
from utils.whisper_utils import record_audio, transcribe, count_filler_words
from agents.question_agent import generate_question
from agents.evaluator_agent import evaluate_answer
from agents.memory_agent import MemoryAgent

load_dotenv()

st.set_page_config(
    page_title="InterviewAI",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=Inter:ital,wght@0,300;0,400;0,500;0,600;1,400&display=swap');

/* ── Reset ── */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [class*="css"], .stApp {
    font-family: 'Inter', sans-serif;
    background-color: #f5f4fe !important;
    color: #1a1a2e;
}

#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2.5rem 3rem 5rem 3rem !important; max-width: 1080px; }

/* ── Keyframe Animations ── */
@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(18px); }
    to   { opacity: 1; transform: translateY(0); }
}
@keyframes fadeInLeft {
    from { opacity: 0; transform: translateX(-16px); }
    to   { opacity: 1; transform: translateX(0); }
}
@keyframes fadeIn {
    from { opacity: 0; }
    to   { opacity: 1; }
}
@keyframes scaleIn {
    from { opacity: 0; transform: scale(0.94); }
    to   { opacity: 1; transform: scale(1); }
}
@keyframes shimmer {
    0%   { background-position: -400px 0; }
    100% { background-position: 400px 0; }
}
@keyframes pulseGlow {
    0%, 100% { box-shadow: 0 0 0 0 rgba(124,58,237,0.15); }
    50%       { box-shadow: 0 0 0 8px rgba(124,58,237,0); }
}
@keyframes dotBounce {
    0%, 80%, 100% { transform: translateY(0); opacity: 0.4; }
    40%            { transform: translateY(-5px); opacity: 1; }
}
@keyframes slideInRight {
    from { opacity: 0; transform: translateX(20px); }
    to   { opacity: 1; transform: translateX(0); }
}
@keyframes borderGrow {
    from { width: 0; }
    to   { width: 100%; }
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background-color: #ffffff !important;
    border-right: 1px solid #ebe8ff !important;
}
[data-testid="stSidebar"] > div:first-child { background-color: #ffffff !important; }
[data-testid="stSidebar"] .block-container { padding: 2rem 1.5rem !important; }

.sidebar-brand {
    font-family: 'Syne', sans-serif;
    font-size: 1.05rem;
    font-weight: 800;
    color: #1a1a2e;
    margin-bottom: 0.15rem;
    animation: fadeInLeft 0.5s ease both;
}
.sidebar-sub {
    font-size: 0.72rem;
    color: #a09cc0;
    margin-bottom: 1.75rem;
    font-style: italic;
    animation: fadeInLeft 0.5s 0.1s ease both;
}

.s-metric {
    background: #f8f7ff;
    border: 1px solid #ebe8ff;
    border-radius: 12px;
    padding: 0.9rem 1.1rem;
    margin-bottom: 0.65rem;
    animation: fadeInLeft 0.5s ease both;
    transition: border-color 0.2s, box-shadow 0.2s;
}
.s-metric:hover {
    border-color: #c4b8f8;
    box-shadow: 0 2px 12px rgba(124,58,237,0.08);
}
.s-metric-label {
    font-size: 0.6rem;
    font-weight: 700;
    color: #a09cc0;
    text-transform: uppercase;
    letter-spacing: 2px;
    margin-bottom: 0.25rem;
}
.s-metric-value {
    font-family: 'Syne', sans-serif;
    font-size: 1.65rem;
    font-weight: 800;
    color: #1a1a2e;
    line-height: 1;
}
.s-metric-value span { font-size: 0.85rem; color: #a09cc0; font-weight: 400; }

/* ── Badges ── */
.badge {
    display: inline-block;
    padding: 0.25rem 0.85rem;
    border-radius: 20px;
    font-size: 0.6rem;
    font-weight: 800;
    letter-spacing: 2px;
    text-transform: uppercase;
    transition: transform 0.2s;
}
.badge:hover { transform: scale(1.05); }
.badge-easy   { background:#e6faf3; color:#0f7a55; border:1px solid #a8e6ce; }
.badge-medium { background:#fff7e6; color:#b45309; border:1px solid #fcd99a; }
.badge-hard   { background:#fff0f0; color:#c0392b; border:1px solid #f8b4b4; }

/* ── Weak tags ── */
.weak-tag {
    display: inline-block;
    background: #fff0f7;
    border: 1px solid #f0b8d8;
    color: #9d2060;
    border-radius: 6px;
    padding: 0.18rem 0.6rem;
    font-size: 0.68rem;
    font-weight: 500;
    margin: 0.18rem 0.18rem 0.18rem 0;
    animation: scaleIn 0.3s ease both;
    transition: transform 0.15s;
}
.weak-tag:hover { transform: scale(1.04); }

/* ── App Header ── */
.app-header {
    display: flex;
    align-items: center;
    gap: 1rem;
    margin-bottom: 2.75rem;
    padding-bottom: 1.75rem;
    border-bottom: 1px solid #ebe8ff;
    animation: fadeInUp 0.6s ease both;
}
.app-logo {
    width: 48px; height: 48px;
    background: linear-gradient(135deg, #7c3aed, #a855f7);
    border-radius: 13px;
    display: flex; align-items: center; justify-content: center;
    font-size: 21px;
    box-shadow: 0 4px 18px rgba(124,58,237,0.28);
    animation: pulseGlow 3s ease-in-out infinite;
    flex-shrink: 0;
}
.app-name {
    font-family: 'Syne', sans-serif;
    font-size: 1.7rem;
    font-weight: 800;
    color: #1a1a2e;
    letter-spacing: -0.8px;
    line-height: 1.1;
}
.app-name span { color: #7c3aed; }
.app-tagline {
    font-size: 0.78rem;
    color: #a09cc0;
    font-style: italic;
    margin-top: 0.1rem;
}

/* ── Section anatomy ── */
.step-chip {
    display: inline-block;
    background: linear-gradient(135deg, #ede9ff, #e4dcff);
    color: #6d28d9;
    border-radius: 20px;
    padding: 0.22rem 0.8rem;
    font-size: 0.58rem;
    font-weight: 800;
    letter-spacing: 2.5px;
    text-transform: uppercase;
    margin-bottom: 0.45rem;
    animation: fadeIn 0.4s ease both;
}
.section-title {
    font-family: 'Syne', sans-serif;
    font-size: 1.3rem;
    font-weight: 800;
    color: #1a1a2e;
    margin-bottom: 0.35rem;
    letter-spacing: -0.3px;
    animation: fadeInUp 0.5s ease both;
}
.section-desc {
    font-size: 0.82rem;
    color: #a09cc0;
    margin-bottom: 1.25rem;
    font-style: italic;
    animation: fadeIn 0.6s 0.1s ease both;
}
.divider { border: none; border-top: 1px solid #ebe8ff; margin: 2.25rem 0; }

/* ── Question box ── */
.question-box {
    background: #ffffff;
    border: 1px solid #ebe8ff;
    border-left: 4px solid #7c3aed;
    border-radius: 14px;
    padding: 1.3rem 1.6rem;
    margin-top: 1rem;
    font-size: 1.02rem;
    line-height: 1.7;
    color: #1a1a2e;
    font-weight: 500;
    box-shadow: 0 3px 16px rgba(124,58,237,0.08);
    animation: scaleIn 0.4s ease both;
    position: relative;
    overflow: hidden;
}
.question-box::after {
    content: '';
    position: absolute;
    bottom: 0; left: 0;
    height: 2px;
    background: linear-gradient(90deg, #7c3aed, #a855f7, #7c3aed);
    animation: borderGrow 0.6s 0.2s ease both;
    width: 100%;
}

/* ── Metric grid ── */
.metric-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 1rem;
    margin: 1.5rem 0;
}
.metric-card {
    background: #ffffff;
    border: 1px solid #ebe8ff;
    border-radius: 14px;
    padding: 1.3rem 1.5rem;
    text-align: center;
    box-shadow: 0 2px 10px rgba(124,58,237,0.06);
    animation: fadeInUp 0.5s ease both;
    transition: transform 0.2s, box-shadow 0.2s;
}
.metric-card:hover { transform: translateY(-3px); box-shadow: 0 8px 24px rgba(124,58,237,0.12); }
.metric-card:nth-child(2) { animation-delay: 0.08s; }
.metric-card:nth-child(3) { animation-delay: 0.16s; }
.metric-value {
    font-family: 'Syne', sans-serif;
    font-size: 2.3rem;
    font-weight: 800;
    color: #7c3aed;
    line-height: 1;
    margin-bottom: 0.3rem;
}
.metric-value span { font-size: 1rem; color: #c4b8f8; font-weight: 400; }
.metric-label {
    font-size: 0.65rem;
    font-weight: 700;
    color: #a09cc0;
    text-transform: uppercase;
    letter-spacing: 1.5px;
}

/* ── STAR grid ── */
.star-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 0.75rem;
    margin: 0.75rem 0 1.75rem;
}
.star-card {
    background: #ffffff;
    border: 1px solid #ebe8ff;
    border-radius: 12px;
    padding: 1rem;
    text-align: center;
    box-shadow: 0 2px 8px rgba(124,58,237,0.05);
    animation: fadeInUp 0.5s ease both;
    transition: transform 0.2s, box-shadow 0.2s;
}
.star-card:hover { transform: translateY(-2px); box-shadow: 0 6px 18px rgba(124,58,237,0.1); }
.star-card:nth-child(2) { animation-delay: 0.06s; }
.star-card:nth-child(3) { animation-delay: 0.12s; }
.star-card:nth-child(4) { animation-delay: 0.18s; }
.star-letter {
    width: 28px; height: 28px;
    background: linear-gradient(135deg, #ede9ff, #ddd5ff);
    border-radius: 8px;
    display: flex; align-items: center; justify-content: center;
    font-family: 'Syne', sans-serif;
    font-size: 0.75rem;
    font-weight: 800;
    color: #6d28d9;
    margin: 0 auto 0.5rem;
}
.star-label {
    font-size: 0.6rem;
    font-weight: 700;
    color: #a09cc0;
    text-transform: uppercase;
    letter-spacing: 2px;
    margin-bottom: 0.35rem;
}
.star-value {
    font-family: 'Syne', sans-serif;
    font-size: 1.7rem;
    font-weight: 800;
    color: #1a1a2e;
}
.star-value span { font-size: 0.78rem; color: #c4b8f8; font-weight: 400; }

/* ── Feedback rows ── */
.fb-section-title {
    font-family: 'Syne', sans-serif;
    font-size: 0.9rem;
    font-weight: 700;
    color: #1a1a2e;
    margin-bottom: 0.6rem;
    padding-bottom: 0.5rem;
    border-bottom: 2px solid #ebe8ff;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.fb-icon {
    width: 20px; height: 20px;
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 0.7rem;
    flex-shrink: 0;
}
.fb-icon-green { background: #dcfce7; color: #16a34a; }
.fb-icon-red   { background: #fee2e2; color: #dc2626; }
.fb-row {
    display: flex;
    align-items: flex-start;
    gap: 0.75rem;
    padding: 0.65rem 0;
    border-bottom: 1px solid #f5f3ff;
    font-size: 0.855rem;
    line-height: 1.55;
    color: #3a3a5c;
    animation: fadeInUp 0.4s ease both;
}
.dot-green { width:7px;height:7px;min-width:7px;background:#16a34a;border-radius:50%;margin-top:6px; }
.dot-red   { width:7px;height:7px;min-width:7px;background:#dc2626;border-radius:50%;margin-top:6px; }

/* ── Improved answer ── */
.improved-label {
    font-family: 'Syne', sans-serif;
    font-size: 0.9rem;
    font-weight: 700;
    color: #1a1a2e;
    margin: 1.75rem 0 0.65rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.improved-box {
    background: linear-gradient(135deg, #f0fdf8 0%, #e8fdf2 100%);
    border: 1px solid #bbf0d6;
    border-left: 4px solid #16a34a;
    border-radius: 14px;
    padding: 1.3rem 1.6rem;
    font-size: 0.88rem;
    line-height: 1.75;
    color: #14532d;
    animation: scaleIn 0.4s ease both;
    font-style: italic;
}
.tip-box {
    background: linear-gradient(135deg, #fffbeb 0%, #fef9e0 100%);
    border: 1px solid #fcd34d;
    border-radius: 12px;
    padding: 0.9rem 1.25rem;
    font-size: 0.845rem;
    color: #78350f;
    margin-top: 1rem;
    animation: fadeInUp 0.4s ease both;
}
.next-box {
    background: linear-gradient(135deg, #f5f0ff 0%, #ede9ff 100%);
    border: 1px solid #d8d0ff;
    border-radius: 12px;
    padding: 0.9rem 1.25rem;
    font-size: 0.845rem;
    color: #5b21b6;
    margin-top: 1rem;
    animation: fadeInUp 0.4s 0.1s ease both;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #7c3aed 0%, #9333ea 100%) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.6rem 1.6rem !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.865rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.2px !important;
    box-shadow: 0 4px 16px rgba(124,58,237,0.32) !important;
    transition: all 0.22s cubic-bezier(0.4,0,0.2,1) !important;
    position: relative !important;
    overflow: hidden !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 24px rgba(124,58,237,0.42) !important;
}
.stButton > button:active {
    transform: translateY(0px) !important;
}

/* ── Inputs ── */
.stTextArea textarea, .stTextInput input {
    background: #ffffff !important;
    border: 1.5px solid #e2dcff !important;
    border-radius: 11px !important;
    color: #1a1a2e !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.88rem !important;
    transition: border-color 0.2s, box-shadow 0.2s !important;
}
.stTextArea textarea:focus, .stTextInput input:focus {
    border-color: #7c3aed !important;
    box-shadow: 0 0 0 3px rgba(124,58,237,0.13) !important;
}
textarea::placeholder, input::placeholder {
    color: #c4b8f8 !important;
    font-style: italic !important;
}

/* ── Radio ── */
.stRadio > div { gap: 0.5rem; }
.stRadio label {
    font-size: 0.855rem !important;
    color: #3a3a5c !important;
    font-weight: 500 !important;
}

/* ── Spinner ── */
.stSpinner > div { border-top-color: #7c3aed !important; }

/* ── Alerts ── */
div[data-testid="stAlert"] {
    border-radius: 11px !important;
    font-size: 0.855rem !important;
    animation: fadeInUp 0.3s ease both !important;
}

/* ── Adaptive row ── */
.adaptive-row {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    margin-bottom: 1.25rem;
    animation: fadeIn 0.5s ease both;
}
.adaptive-label { font-size: 0.8rem; color: #a09cc0; font-style: italic; }
.adaptive-note  { font-size: 0.75rem; color: #c4b8f8; }

/* ── Section separator with label ── */
.sub-label {
    font-family: 'Syne', sans-serif;
    font-size: 0.88rem;
    font-weight: 700;
    color: #1a1a2e;
    margin: 1.75rem 0 0.65rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.sub-label-line {
    flex: 1;
    height: 1px;
    background: #ebe8ff;
}
</style>
""", unsafe_allow_html=True)

# ── Session State ─────────────────────────────────────────────
for key, val in [("memory", None), ("db", None), ("question", None), ("answer", ""), ("last_result", None)]:
    if key not in st.session_state:
        st.session_state[key] = MemoryAgent() if key == "memory" else val

memory     = st.session_state.memory
difficulty = memory.get_difficulty()
badge_map  = {"easy": "badge-easy", "medium": "badge-medium", "hard": "badge-hard"}

# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"""
        <div class='sidebar-brand'>Your Progress</div>
        <div class='sidebar-sub'>Tracked automatically across every session</div>
        <div class='s-metric'>
            <div class='s-metric-label'>Sessions Done</div>
            <div class='s-metric-value'>{memory.session_count}</div>
        </div>
        <div class='s-metric'>
            <div class='s-metric-label'>Average Score</div>
            <div class='s-metric-value'>{memory.get_avg_score()}<span>&thinsp;/10</span></div>
        </div>
        <div class='s-metric'>
            <div class='s-metric-label'>Current Difficulty</div>
            <div style='margin-top:0.5rem;'>
                <span class='badge {badge_map[difficulty]}'>{difficulty}</span>
            </div>
        </div>
    """, unsafe_allow_html=True)

    if memory.get_weak_areas():
        st.markdown("""
            <div style='margin-top:1.5rem;margin-bottom:0.55rem;
                        font-size:0.6rem;font-weight:700;color:#a09cc0;
                        text-transform:uppercase;letter-spacing:2px;'>
                Flagged Weak Areas
            </div>
        """, unsafe_allow_html=True)
        st.markdown(
            "".join([f"<span class='weak-tag'>{w}</span>" for w in memory.get_weak_areas()]),
            unsafe_allow_html=True
        )

    if memory.history:
        st.markdown("""
            <div style='margin-top:1.5rem;margin-bottom:0.4rem;
                        font-size:0.6rem;font-weight:700;color:#a09cc0;
                        text-transform:uppercase;letter-spacing:2px;'>
                Score History
            </div>
        """, unsafe_allow_html=True)
        scores = [h["score"] for h in memory.history]
        fig_s = go.Figure(go.Scatter(
            y=scores, mode='lines+markers',
            line=dict(color='#7c3aed', width=2.5),
            marker=dict(size=7, color='#a855f7',
                        line=dict(color='white', width=1.5)),
            fill='tozeroy',
            fillcolor='rgba(124,58,237,0.07)'
        ))
        fig_s.update_layout(
            height=155, margin=dict(l=0,r=0,t=8,b=0),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            yaxis=dict(range=[0,10], gridcolor='#f0ecff',
                       color='#a09cc0', tickfont=dict(size=9)),
            xaxis=dict(gridcolor='#f0ecff', color='#a09cc0',
                       tickfont=dict(size=9))
        )
        st.plotly_chart(fig_s, use_container_width=True)

    st.markdown("<div style='margin-top:1.25rem;'>", unsafe_allow_html=True)
    if st.button("Reset All Progress", use_container_width=True):
        st.session_state.memory.reset()
        st.session_state.memory   = MemoryAgent()
        st.session_state.question = None
        st.session_state.last_result = None
        st.rerun()

# ── Main: Header ──────────────────────────────────────────────
st.markdown("""
<div class='app-header'>
    <div class='app-logo'>💼</div>
    <div>
        <div class='app-name'>Interview<span>AI</span></div>
        <div class='app-tagline'>
            Multi-agent coaching — generate role-specific questions, get STAR-format feedback,
            and track your growth session by session.
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Step 1 ────────────────────────────────────────────────────
st.markdown("<div class='step-chip'>Step 01</div>", unsafe_allow_html=True)
st.markdown("<div class='section-title'>Upload Job Description</div>", unsafe_allow_html=True)
st.markdown("<div class='section-desc'>Paste the job posting below. The AI reads it, extracts key requirements, and uses it to generate targeted questions throughout your session.</div>", unsafe_allow_html=True)

col_jd, col_role = st.columns([3, 1])
with col_jd:
    jd_text = st.text_area(
        "jd", height=135,
        placeholder="e.g. We are looking for a Senior Backend Engineer with 3+ years of experience in Python, FastAPI, and PostgreSQL. You will design scalable REST APIs, collaborate with cross-functional teams, and own infrastructure on AWS...",
        label_visibility="collapsed"
    )
with col_role:
    role = st.text_input("Job Role", value="Software Engineer",
                         placeholder="e.g. Backend Engineer")
    st.markdown("<div style='height:0.4rem'></div>", unsafe_allow_html=True)
    if st.button("Load Job Description", use_container_width=True):
        if jd_text:
            with st.spinner("Reading and embedding job description..."):
                st.session_state.db = embed_jd(jd_text)
            st.success("Job description embedded — you're ready to generate questions.")
        else:
            st.warning("Paste a job description before loading.")

st.markdown("<hr class='divider'>", unsafe_allow_html=True)

# ── Step 2 ────────────────────────────────────────────────────
st.markdown("<div class='step-chip'>Step 02</div>", unsafe_allow_html=True)
st.markdown("<div class='section-title'>Generate a Question</div>", unsafe_allow_html=True)
st.markdown("<div class='section-desc'>Each question is tailored to the job description and your weakest areas from previous rounds. Difficulty adapts as your scores improve.</div>", unsafe_allow_html=True)

st.markdown(f"""
<div class='adaptive-row'>
    <span class='adaptive-label'>Adaptive difficulty</span>
    <span class='badge {badge_map[difficulty]}'>{difficulty}</span>
    <span class='adaptive-note'>— adjusts automatically as your average score changes</span>
</div>
""", unsafe_allow_html=True)

if st.button("Generate Question"):
    if st.session_state.db is None:
        st.warning("Load a job description first — the AI needs it to generate relevant questions.")
    else:
        with st.spinner("Crafting a targeted question for you..."):
            context    = retrieve_context(role, st.session_state.db)
            weak_areas = memory.get_weak_areas()
            q = generate_question(context, role, weak_areas, difficulty)
            st.session_state.question = q

if st.session_state.question:
    st.markdown(f"<div class='question-box'>{st.session_state.question}</div>",
                unsafe_allow_html=True)

st.markdown("<hr class='divider'>", unsafe_allow_html=True)

# ── Step 3 ────────────────────────────────────────────────────
st.markdown("<div class='step-chip'>Step 03</div>", unsafe_allow_html=True)
st.markdown("<div class='section-title'>Your Answer</div>", unsafe_allow_html=True)
st.markdown("<div class='section-desc'>Strong answers follow the STAR format — describe the <em>Situation</em>, your <em>Task</em>, the <em>Action</em> you took, and the measurable <em>Result</em>. Be specific with numbers.</div>", unsafe_allow_html=True)

mode = st.radio("Input method", ["Type my answer", "Record voice (10 seconds)"], horizontal=True)

if mode == "Type my answer":
    st.session_state.answer = st.text_area(
        "ans", height=155,
        placeholder="Walk through a real example. Mention what the situation was, what you were responsible for, what steps you took, and what the outcome was — ideally with a metric (e.g. reduced latency by 40%, increased retention by 12%).",
        label_visibility="collapsed"
    )
else:
    if st.button("Start Recording"):
        with st.spinner("Recording for 10 seconds — speak clearly and at a steady pace..."):
            filepath = record_audio(duration=10)
        if filepath:
            with st.spinner("Transcribing your answer..."):
                text = transcribe(filepath)
            st.session_state.answer = text
            st.success(f"Transcription complete: {text}")
        else:
            st.warning("Recording failed. Please check your microphone settings and try again.")

st.markdown("<hr class='divider'>", unsafe_allow_html=True)

# ── Step 4 ────────────────────────────────────────────────────
st.markdown("<div class='step-chip'>Step 04</div>", unsafe_allow_html=True)
st.markdown("<div class='section-title'>Get Your Evaluation</div>", unsafe_allow_html=True)
st.markdown("<div class='section-desc'>The evaluation agent scores your answer across four STAR dimensions and five soft skills, identifies patterns in your weak areas, and suggests a stronger version of your answer.</div>", unsafe_allow_html=True)

if st.button("Evaluate My Answer"):
    if not st.session_state.question:
        st.warning("Generate a question first before submitting an answer.")
    elif not st.session_state.answer:
        st.warning("Write or record your answer before requesting an evaluation.")
    else:
        with st.spinner("Evaluating your response — this takes a moment..."):
            context = retrieve_context(st.session_state.question, st.session_state.db)
            result  = evaluate_answer(st.session_state.question, st.session_state.answer, context)
            memory.update(result)
            st.session_state.last_result = result

if st.session_state.last_result:
    result = st.session_state.last_result
    score  = result.get("score", 0)
    star   = result.get("star_score", {})
    soft   = result.get("soft_skills", {})

    # Score cards
    st.markdown(f"""
    <div class='metric-grid'>
        <div class='metric-card'>
            <div class='metric-value'>{score}<span>/10</span></div>
            <div class='metric-label'>Overall Score</div>
        </div>
        <div class='metric-card'>
            <div class='metric-value'>{memory.session_count}</div>
            <div class='metric-label'>Sessions Done</div>
        </div>
        <div class='metric-card'>
            <div class='metric-value'>{memory.get_avg_score()}<span>/10</span></div>
            <div class='metric-label'>Running Average</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # STAR breakdown
    st.markdown("""
    <div class='sub-label'>
        STAR Format Breakdown
        <div class='sub-label-line'></div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown(f"""
    <div class='star-grid'>
        <div class='star-card'>
            <div class='star-letter'>S</div>
            <div class='star-label'>Situation</div>
            <div class='star-value'>{star.get('situation',0)}<span>/10</span></div>
        </div>
        <div class='star-card'>
            <div class='star-letter'>T</div>
            <div class='star-label'>Task</div>
            <div class='star-value'>{star.get('task',0)}<span>/10</span></div>
        </div>
        <div class='star-card'>
            <div class='star-letter'>A</div>
            <div class='star-label'>Action</div>
            <div class='star-value'>{star.get('action',0)}<span>/10</span></div>
        </div>
        <div class='star-card'>
            <div class='star-letter'>R</div>
            <div class='star-label'>Result</div>
            <div class='star-value'>{star.get('result',0)}<span>/10</span></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Radar chart
    st.markdown("""
    <div class='sub-label'>
        Soft Skills Radar
        <div class='sub-label-line'></div>
    </div>
    """, unsafe_allow_html=True)
    cats = ["Communication", "Problem Solving", "Leadership", "Teamwork", "Adaptability"]
    vals = [soft.get(k, 0) for k in ["communication","problem_solving","leadership","teamwork","adaptability"]]
    fig = go.Figure(data=go.Scatterpolar(
        r=vals + vals[:1],
        theta=cats + [cats[0]],
        fill='toself',
        fillcolor='rgba(124,58,237,0.1)',
        line=dict(color='#7c3aed', width=2.5),
        marker=dict(color='#a855f7', size=7,
                    line=dict(color='white', width=1.5))
    ))
    fig.update_layout(
        polar=dict(
            bgcolor='rgba(0,0,0,0)',
            radialaxis=dict(visible=True, range=[0,10],
                            gridcolor='#ebe8ff', color='#a09cc0',
                            tickfont=dict(size=9)),
            angularaxis=dict(gridcolor='#ebe8ff', color='#3a3a5c',
                             tickfont=dict(size=11, family='Inter'))
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        showlegend=False,
        height=380,
        margin=dict(l=60, r=60, t=40, b=40)
    )
    st.plotly_chart(fig, use_container_width=True)

    # Strengths & Weaknesses
    st.markdown("""
    <div class='sub-label'>
        Detailed Feedback
        <div class='sub-label-line'></div>
    </div>
    """, unsafe_allow_html=True)
    col_s, col_w = st.columns(2)
    with col_s:
        st.markdown("<div class='fb-section-title'><div class='fb-icon fb-icon-green'>✓</div> What you did well</div>", unsafe_allow_html=True)
        for s in result.get("strengths", []):
            st.markdown(f"<div class='fb-row'><div class='dot-green'></div><div>{s}</div></div>", unsafe_allow_html=True)
    with col_w:
        st.markdown("<div class='fb-section-title'><div class='fb-icon fb-icon-red'>!</div> Where to improve</div>", unsafe_allow_html=True)
        for w in result.get("weaknesses", []):
            st.markdown(f"<div class='fb-row'><div class='dot-red'></div><div>{w}</div></div>", unsafe_allow_html=True)

    # Improved answer
    st.markdown("<div class='improved-label'>✦ How a stronger answer would sound</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='improved-box'>{result.get('improved_answer','')}</div>", unsafe_allow_html=True)

    if result.get("filler_word_tip"):
        st.markdown(f"""
        <div class='tip-box'>
            <strong>Communication note —</strong> {result.get('filler_word_tip')}
        </div>
        """, unsafe_allow_html=True)

    new_diff = memory.get_difficulty()
    st.markdown(f"""
    <div class='next-box'>
        &#8594;&nbsp; Your next question will be at <strong>{new_diff.upper()}</strong> difficulty.
        Click <em>Generate Question</em> above to keep going.
    </div>
    """, unsafe_allow_html=True)