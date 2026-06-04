import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer
import sklearn
import time

ps = PorterStemmer()
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)

# ─── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SpamShield — SMS Classifier",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=JetBrains+Mono:wght@400;500&family=DM+Sans:wght@300;400;500&display=swap');

/* ── Reset & Base ── */
*, *::before, *::after { box-sizing: border-box; }

html, body, [data-testid="stAppViewContainer"] {
    background: #080c14 !important;
    color: #e2e8f0 !important;
    font-family: 'DM Sans', sans-serif !important;
}

[data-testid="stAppViewContainer"] {
    background:
        radial-gradient(ellipse 80% 50% at 20% -10%, rgba(0,210,255,0.07) 0%, transparent 60%),
        radial-gradient(ellipse 60% 40% at 80% 110%, rgba(255,60,100,0.06) 0%, transparent 60%),
        #080c14 !important;
}

[data-testid="stHeader"] { background: transparent !important; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #0d1420 !important;
    border-right: 1px solid rgba(255,255,255,0.06) !important;
}

[data-testid="stSidebar"] .stMarkdown p,
[data-testid="stSidebar"] .stMarkdown li,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stSelectbox label {
    color: #94a3b8 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.84rem !important;
}

[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #e2e8f0 !important;
    font-family: 'Syne', sans-serif !important;
}

/* ── Main headings ── */
h1, h2, h3 { font-family: 'Syne', sans-serif !important; }

/* ── Hero Banner ── */
.hero-banner {
    background: linear-gradient(135deg, #0d1f35 0%, #0d1420 50%, #1a0d20 100%);
    border: 1px solid rgba(0,210,255,0.15);
    border-radius: 20px;
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero-banner::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 220px; height: 220px;
    background: radial-gradient(circle, rgba(0,210,255,0.12) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-banner::after {
    content: '';
    position: absolute;
    bottom: -40px; left: -40px;
    width: 180px; height: 180px;
    background: radial-gradient(circle, rgba(255,60,100,0.09) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-title {
    font-family: 'Syne', sans-serif !important;
    font-size: 2.6rem !important;
    font-weight: 800 !important;
    background: linear-gradient(120deg, #00d2ff, #7b5ea7, #ff3c64);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.1 !important;
    margin-bottom: 0.5rem;
}
.hero-subtitle {
    font-family: 'DM Sans', sans-serif;
    color: #64748b;
    font-size: 1rem;
    font-weight: 300;
    letter-spacing: 0.04em;
}
.hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(0,210,255,0.08);
    border: 1px solid rgba(0,210,255,0.2);
    border-radius: 20px;
    padding: 4px 14px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    color: #00d2ff;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 1rem;
}

/* ── Metric Cards ── */
.metric-row {
    display: flex;
    gap: 1rem;
    margin-bottom: 1.5rem;
}
.metric-card {
    flex: 1;
    background: #0d1420;
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 14px;
    padding: 1.2rem 1.4rem;
    position: relative;
    overflow: hidden;
    transition: border-color 0.3s;
}
.metric-card:hover { border-color: rgba(0,210,255,0.25); }
.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
}
.metric-card.cyan::before  { background: linear-gradient(90deg, transparent, #00d2ff, transparent); }
.metric-card.green::before { background: linear-gradient(90deg, transparent, #00e96e, transparent); }
.metric-card.amber::before { background: linear-gradient(90deg, transparent, #f59e0b, transparent); }
.metric-card.red::before   { background: linear-gradient(90deg, transparent, #ff3c64, transparent); }

.metric-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #475569;
    margin-bottom: 0.5rem;
}
.metric-value {
    font-family: 'Syne', sans-serif;
    font-size: 1.8rem;
    font-weight: 700;
    color: #e2e8f0;
    line-height: 1;
}
.metric-value.cyan  { color: #00d2ff; }
.metric-value.green { color: #00e96e; }
.metric-value.amber { color: #f59e0b; }
.metric-value.red   { color: #ff3c64; }
.metric-delta {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.75rem;
    color: #475569;
    margin-top: 0.3rem;
}

/* ── Input Area ── */
[data-testid="stTextArea"] textarea {
    background: #0d1420 !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 12px !important;
    color: #e2e8f0 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.95rem !important;
    padding: 1rem !important;
    resize: none !important;
    transition: border-color 0.25s !important;
}
[data-testid="stTextArea"] textarea:focus {
    border-color: rgba(0,210,255,0.4) !important;
    box-shadow: 0 0 0 3px rgba(0,210,255,0.08) !important;
    outline: none !important;
}
[data-testid="stTextArea"] label {
    color: #94a3b8 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #0099cc, #0066aa) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'Syne', sans-serif !important;
    font-size: 0.95rem !important;
    font-weight: 600 !important;
    padding: 0.7rem 2.2rem !important;
    letter-spacing: 0.04em !important;
    transition: all 0.25s !important;
    box-shadow: 0 4px 20px rgba(0,153,204,0.25) !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #00b8f0, #0080cc) !important;
    box-shadow: 0 6px 28px rgba(0,153,204,0.4) !important;
    transform: translateY(-1px) !important;
}
.stButton > button:active { transform: translateY(0) !important; }

/* ── Result Boxes ── */
.result-spam {
    background: linear-gradient(135deg, rgba(255,60,100,0.12), rgba(180,0,50,0.08));
    border: 1px solid rgba(255,60,100,0.35);
    border-left: 4px solid #ff3c64;
    border-radius: 14px;
    padding: 1.5rem 2rem;
    margin: 1.5rem 0;
}
.result-ham {
    background: linear-gradient(135deg, rgba(0,233,110,0.10), rgba(0,150,80,0.06));
    border: 1px solid rgba(0,233,110,0.3);
    border-left: 4px solid #00e96e;
    border-radius: 14px;
    padding: 1.5rem 2rem;
    margin: 1.5rem 0;
}
.result-title {
    font-family: 'Syne', sans-serif;
    font-size: 1.4rem;
    font-weight: 700;
    margin-bottom: 0.3rem;
}
.result-subtitle {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.88rem;
    opacity: 0.7;
}

/* ── Probability Bar ── */
.prob-bar-wrap {
    background: rgba(255,255,255,0.05);
    border-radius: 8px;
    height: 10px;
    overflow: hidden;
    margin: 0.6rem 0 0.3rem;
}
.prob-bar-fill-spam {
    height: 100%;
    border-radius: 8px;
    background: linear-gradient(90deg, #ff7b54, #ff3c64);
    transition: width 0.8s cubic-bezier(0.4,0,0.2,1);
}
.prob-bar-fill-ham {
    height: 100%;
    border-radius: 8px;
    background: linear-gradient(90deg, #00c87a, #00e96e);
    transition: width 0.8s cubic-bezier(0.4,0,0.2,1);
}
.prob-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
    display: flex;
    justify-content: space-between;
    color: #64748b;
}

/* ── Divider ── */
.custom-divider {
    border: none;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.07), transparent);
    margin: 2rem 0;
}

/* ── Example Cards ── */
.example-section-title {
    font-family: 'Syne', sans-serif;
    font-size: 1rem;
    font-weight: 700;
    color: #e2e8f0;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    gap: 10px;
}
.example-section-title span {
    display: inline-block;
    width: 28px; height: 2px;
    background: #00d2ff;
}
.example-card {
    background: #0d1420;
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 12px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.7rem;
    cursor: pointer;
    transition: all 0.2s;
    position: relative;
}
.example-card:hover {
    border-color: rgba(0,210,255,0.25);
    background: #111827;
    transform: translateX(3px);
}
.example-card .tag {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    padding: 2px 8px;
    border-radius: 20px;
    margin-bottom: 0.4rem;
    display: inline-block;
}
.example-card .tag.spam {
    background: rgba(255,60,100,0.12);
    color: #ff6b8a;
    border: 1px solid rgba(255,60,100,0.2);
}
.example-card .tag.ham {
    background: rgba(0,233,110,0.1);
    color: #4ade80;
    border: 1px solid rgba(0,233,110,0.2);
}
.example-card p {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.85rem;
    color: #94a3b8;
    margin: 0;
    line-height: 1.5;
}

/* ── Sidebar Nav ── */
.nav-item {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 0.65rem 0.9rem;
    border-radius: 10px;
    margin-bottom: 0.3rem;
    cursor: pointer;
    transition: all 0.2s;
    font-family: 'DM Sans', sans-serif;
    font-size: 0.88rem;
    color: #64748b;
    border: 1px solid transparent;
}
.nav-item.active {
    background: rgba(0,210,255,0.08);
    border-color: rgba(0,210,255,0.15);
    color: #00d2ff;
}
.nav-item:hover:not(.active) {
    background: rgba(255,255,255,0.04);
    color: #94a3b8;
}

/* ── Sidebar stat badges ── */
.sidebar-stat {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 10px;
    padding: 0.8rem 1rem;
    margin-bottom: 0.6rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
}
.sidebar-stat-label {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.8rem;
    color: #475569;
}
.sidebar-stat-val {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.85rem;
    color: #00d2ff;
    font-weight: 500;
}

/* ── Footer ── */
.footer {
    text-align: center;
    padding: 2rem 0 1rem;
    margin-top: 3rem;
    border-top: 1px solid rgba(255,255,255,0.05);
}
.footer p {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    color: #334155;
    letter-spacing: 0.05em;
    margin: 0;
}
.footer .tech-tags {
    display: flex;
    justify-content: center;
    gap: 0.5rem;
    flex-wrap: wrap;
    margin-top: 0.6rem;
}
.tech-tag {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem;
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 20px;
    padding: 3px 10px;
    color: #475569;
}

/* ── Analysis Panel ── */
.analysis-panel {
    background: #0d1420;
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 14px;
    padding: 1.4rem 1.6rem;
    margin-top: 1rem;
}
.analysis-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.45rem 0;
    border-bottom: 1px solid rgba(255,255,255,0.04);
}
.analysis-row:last-child { border-bottom: none; }
.analysis-key {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.82rem;
    color: #475569;
}
.analysis-val {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.82rem;
    color: #94a3b8;
}

/* ── Selectbox ── */
[data-testid="stSelectbox"] > div > div {
    background: #0d1420 !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 10px !important;
    color: #e2e8f0 !important;
    font-family: 'DM Sans', sans-serif !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #080c14; }
::-webkit-scrollbar-thumb { background: #1e2d40; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #2d4a63; }

/* ── Expander ── */
[data-testid="stExpander"] {
    background: #0d1420 !important;
    border: 1px solid rgba(255,255,255,0.07) !important;
    border-radius: 12px !important;
}
[data-testid="stExpander"] summary {
    color: #94a3b8 !important;
    font-family: 'DM Sans', sans-serif !important;
}
</style>
""", unsafe_allow_html=True)


# ─── Data ───────────────────────────────────────────────────────────────────────
SPAM_EXAMPLES = [
    "Congratulations! You have won ₹50,000. Claim your prize now by clicking the link.",
    "URGENT! Your account has been suspended. Verify your details immediately.",
    "You have been selected for a free iPhone 16. Reply YES to claim.",
    "Earn ₹10,000 per day working from home. Limited spots available!",
    "Exclusive offer! Get 90% discount on all products. Shop now.",
    "Winner! You have won a vacation package. Call now to redeem.",
    "Get a personal loan instantly with zero documentation. Apply today.",
    "Your ATM card will be blocked. Update KYC immediately.",
    "Claim your lottery winnings before midnight.",
    "You have won a brand new car! Click the link to claim now.",
]

HAM_EXAMPLES = [
    "Hey, are we still meeting at 5 PM today?",
    "Don't forget to bring the assignment tomorrow.",
    "Happy Birthday! Have a wonderful day ahead.",
    "Your Amazon order has been delivered successfully.",
    "Let's have lunch together after class.",
    "The meeting has been rescheduled to Monday morning.",
    "Mom said dinner will be ready by 8 PM.",
    "Good luck for your exam tomorrow!",
    "Your train will arrive at platform number 3.",
    "Thank you for your payment. Transaction completed successfully.",
]


# ─── NLP ────────────────────────────────────────────────────────────────────────
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    y = [i for i in text if i.isalnum()]
    y = [i for i in y if i not in stopwords.words('english') and i not in string.punctuation]
    y = [ps.stem(i) for i in y]
    return " ".join(y)


# ─── Load Models ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    tfidf  = pickle.load(open('vectoizer.pkl', 'rb'))
    model  = pickle.load(open('model.pkl', 'rb'))
    return tfidf, model

tfidf, model = load_models()


# ─── Session State ──────────────────────────────────────────────────────────────
if 'history' not in st.session_state:
    st.session_state.history = []
if 'total_scanned' not in st.session_state:
    st.session_state.total_scanned = 0
if 'spam_caught' not in st.session_state:
    st.session_state.spam_caught = 0
if 'selected_example' not in st.session_state:
    st.session_state.selected_example = ""
if 'active_page' not in st.session_state:
    st.session_state.active_page = "Classifier"


# ─── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding: 0.5rem 0 1.5rem;'>
        <div style='font-family:Syne,sans-serif; font-size:1.3rem; font-weight:800;
                    background:linear-gradient(120deg,#00d2ff,#7b5ea7);
                    -webkit-background-clip:text; -webkit-text-fill-color:transparent;
                    background-clip:text; margin-bottom:4px;'>
            SpamShield
        </div>
        <div style='font-family:JetBrains Mono,monospace; font-size:0.65rem;
                    color:#334155; letter-spacing:0.1em; text-transform:uppercase;'>
            v2.0 · NLP Engine
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Navigation
    pages = [
        ("🎯", "Classifier",  "Classifier"),
        ("📊", "Analytics",   "Analytics"),
        ("📚", "Examples",    "Examples"),
        ("⚙️", "About",       "About"),
    ]
    for icon, label, key in pages:
        active = "active" if st.session_state.active_page == key else ""
        if st.button(f"{icon}  {label}", key=f"nav_{key}", use_container_width=True):
            st.session_state.active_page = key
            st.rerun()

    st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)

    # Session Stats
    st.markdown("<div style='font-family:Syne,sans-serif; font-size:0.75rem; font-weight:700; color:#334155; text-transform:uppercase; letter-spacing:0.1em; margin-bottom:0.8rem;'>Session Stats</div>", unsafe_allow_html=True)

    accuracy_rate = (
        round((st.session_state.spam_caught / st.session_state.total_scanned) * 100, 1)
        if st.session_state.total_scanned > 0 else 0.0
    )
    ham_count = st.session_state.total_scanned - st.session_state.spam_caught

    stats = [
        ("Messages Scanned", st.session_state.total_scanned),
        ("Spam Detected",    st.session_state.spam_caught),
        ("Ham Passed",       ham_count),
        ("Detection Rate",   f"{accuracy_rate}%"),
    ]
    for label, val in stats:
        st.markdown(f"""
        <div class='sidebar-stat'>
            <span class='sidebar-stat-label'>{label}</span>
            <span class='sidebar-stat-val'>{val}</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)
    st.markdown("""
    <div style='font-family:DM Sans,sans-serif; font-size:0.78rem; color:#334155; line-height:1.6;'>
        🤖 Powered by Multinomial Naive Bayes + TF-IDF Vectorization
    </div>
    """, unsafe_allow_html=True)


# ─── Page: Classifier ───────────────────────────────────────────────────────────
if st.session_state.active_page == "Classifier":

    # Hero
    st.markdown("""
    <div class='hero-banner'>
        <div class='hero-badge'>⚡ Real-time NLP Classification</div>
        <div class='hero-title'>SMS Spam Detector</div>
        <div class='hero-subtitle'>Intelligent message filtering using machine learning & natural language processing</div>
    </div>
    """, unsafe_allow_html=True)

    # Metric Cards
    ham_count_m = st.session_state.total_scanned - st.session_state.spam_caught
    st.markdown(f"""
    <div class='metric-row'>
        <div class='metric-card cyan'>
            <div class='metric-label'>Total Scanned</div>
            <div class='metric-value cyan'>{st.session_state.total_scanned}</div>
            <div class='metric-delta'>Messages analyzed</div>
        </div>
        <div class='metric-card red'>
            <div class='metric-label'>Spam Caught</div>
            <div class='metric-value red'>{st.session_state.spam_caught}</div>
            <div class='metric-delta'>Threats blocked</div>
        </div>
        <div class='metric-card green'>
            <div class='metric-label'>Safe Messages</div>
            <div class='metric-value green'>{ham_count_m}</div>
            <div class='metric-delta'>Legitimate messages</div>
        </div>
        <div class='metric-card amber'>
            <div class='metric-label'>Detection Rate</div>
            <div class='metric-value amber'>{accuracy_rate}%</div>
            <div class='metric-delta'>Spam ratio this session</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Input & Controls
    col_input, col_controls = st.columns([3, 1])

    with col_input:
        default_val = st.session_state.selected_example
        input_sms = st.text_area(
            "✉️  Paste or type an SMS message",
            value=default_val,
            height=160,
            placeholder="Enter any SMS message to check whether it's spam or legitimate…",
            key="sms_input",
        )

    with col_controls:
        st.markdown("<div style='height:1.9rem'></div>", unsafe_allow_html=True)
        analyze_btn = st.button("🔍  Analyze", use_container_width=True, key="analyze")
        clear_btn   = st.button("🗑  Clear",   use_container_width=True, key="clear")

        if clear_btn:
            st.session_state.selected_example = ""
            st.rerun()

        char_count = len(input_sms) if input_sms else 0
        word_count = len(input_sms.split()) if input_sms.strip() else 0
        st.markdown(f"""
        <div style='margin-top:1rem; padding:0.8rem; background:#0d1420;
                    border:1px solid rgba(255,255,255,0.06); border-radius:10px;'>
            <div style='font-family:JetBrains Mono,monospace; font-size:0.7rem; color:#334155; margin-bottom:0.4rem; text-transform:uppercase; letter-spacing:0.08em;'>Message Stats</div>
            <div style='font-family:JetBrains Mono,monospace; font-size:0.78rem; color:#475569;'>
                {char_count} chars<br>{word_count} words
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Analysis ──
    if analyze_btn and input_sms.strip():
        with st.spinner(""):
            time.sleep(0.3)  # brief visual pause
            transformed = transform_text(input_sms)
            vector_input = tfidf.transform([transformed])
            result = model.predict(vector_input)[0]
            prob   = model.predict_proba(vector_input)[0]
            spam_prob = prob[1] * 100
            ham_prob  = prob[0] * 100

        # Update stats
        st.session_state.total_scanned += 1
        if result == 1:
            st.session_state.spam_caught += 1
        st.session_state.history.append({
            "message": input_sms[:60] + ("…" if len(input_sms) > 60 else ""),
            "result":  "SPAM" if result == 1 else "HAM",
            "prob":    round(spam_prob, 1),
        })

        # Result box
        if result == 1:
            st.markdown(f"""
            <div class='result-spam'>
                <div class='result-title' style='color:#ff6b8a;'>🚨 SPAM DETECTED</div>
                <div class='result-subtitle'>This message has been flagged as potentially harmful or unsolicited.</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class='result-ham'>
                <div class='result-title' style='color:#4ade80;'>✅ SAFE MESSAGE</div>
                <div class='result-subtitle'>This message appears to be legitimate and safe.</div>
            </div>
            """, unsafe_allow_html=True)

        # Probability bars
        col_s, col_h = st.columns(2)
        with col_s:
            bar_class = "prob-bar-fill-spam"
            st.markdown(f"""
            <div style='padding:1rem; background:#0d1420; border:1px solid rgba(255,60,100,0.15); border-radius:12px;'>
                <div style='font-family:JetBrains Mono,monospace; font-size:0.68rem; text-transform:uppercase;
                            letter-spacing:0.1em; color:#475569; margin-bottom:0.4rem;'>Spam Probability</div>
                <div style='font-family:Syne,sans-serif; font-size:1.8rem; font-weight:700; color:#ff6b8a;'>{spam_prob:.1f}%</div>
                <div class='prob-bar-wrap'><div class='{bar_class}' style='width:{spam_prob:.1f}%'></div></div>
                <div class='prob-label'><span>0%</span><span>100%</span></div>
            </div>
            """, unsafe_allow_html=True)
        with col_h:
            bar_class2 = "prob-bar-fill-ham"
            st.markdown(f"""
            <div style='padding:1rem; background:#0d1420; border:1px solid rgba(0,233,110,0.15); border-radius:12px;'>
                <div style='font-family:JetBrains Mono,monospace; font-size:0.68rem; text-transform:uppercase;
                            letter-spacing:0.1em; color:#475569; margin-bottom:0.4rem;'>Ham Probability</div>
                <div style='font-family:Syne,sans-serif; font-size:1.8rem; font-weight:700; color:#4ade80;'>{ham_prob:.1f}%</div>
                <div class='prob-bar-wrap'><div class='{bar_class2}' style='width:{ham_prob:.1f}%'></div></div>
                <div class='prob-label'><span>0%</span><span>100%</span></div>
            </div>
            """, unsafe_allow_html=True)

        # Processed token analysis
        with st.expander("🔬 NLP Token Analysis", expanded=False):
            tokens = transformed.split() if transformed else []
            st.markdown(f"""
            <div class='analysis-panel'>
                <div class='analysis-row'>
                    <span class='analysis-key'>Processed tokens</span>
                    <span class='analysis-val'>{len(tokens)}</span>
                </div>
                <div class='analysis-row'>
                    <span class='analysis-key'>Stemmed text</span>
                    <span class='analysis-val' style='max-width:55%; text-align:right; word-break:break-all;'>{transformed[:80]}{'…' if len(transformed)>80 else ''}</span>
                </div>
                <div class='analysis-row'>
                    <span class='analysis-key'>Confidence level</span>
                    <span class='analysis-val'>{'HIGH' if max(spam_prob,ham_prob)>80 else 'MEDIUM' if max(spam_prob,ham_prob)>60 else 'LOW'}</span>
                </div>
                <div class='analysis-row'>
                    <span class='analysis-key'>Original chars</span>
                    <span class='analysis-val'>{len(input_sms)}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

    elif analyze_btn:
        st.warning("⚠️  Please enter a message before analyzing.")

    # Recent history (last 5)
    if st.session_state.history:
        st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)
        st.markdown("<div class='example-section-title'><span></span> Recent Scans</div>", unsafe_allow_html=True)
        for item in reversed(st.session_state.history[-5:]):
            tag_class = "spam" if item["result"] == "SPAM" else "ham"
            icon = "🚨" if item["result"] == "SPAM" else "✅"
            st.markdown(f"""
            <div class='example-card'>
                <div class='tag {tag_class}'>{icon} {item["result"]}</div>
                <p>{item["message"]}</p>
                <div style='font-family:JetBrains Mono,monospace; font-size:0.68rem; color:#334155; margin-top:0.3rem;'>
                    Spam prob: {item["prob"]}%
                </div>
            </div>
            """, unsafe_allow_html=True)


# ─── Page: Analytics ────────────────────────────────────────────────────────────
elif st.session_state.active_page == "Analytics":
    st.markdown("""
    <div style='font-family:Syne,sans-serif; font-size:2rem; font-weight:800; color:#e2e8f0; margin-bottom:0.3rem;'>
        Analytics Dashboard
    </div>
    <div style='font-family:DM Sans,sans-serif; color:#475569; margin-bottom:2rem;'>
        Session-level classification metrics and model information
    </div>
    """, unsafe_allow_html=True)

    total = st.session_state.total_scanned
    spam  = st.session_state.spam_caught
    ham   = total - spam

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        <div style='background:#0d1420; border:1px solid rgba(255,255,255,0.07); border-radius:16px; padding:1.8rem;'>
            <div style='font-family:Syne,sans-serif; font-size:1rem; font-weight:700; color:#94a3b8; margin-bottom:1.5rem; text-transform:uppercase; letter-spacing:0.08em;'>
                Classification Breakdown
            </div>
            <div style='margin-bottom:1.5rem;'>
                <div style='display:flex; justify-content:space-between; margin-bottom:0.5rem;'>
                    <span style='font-family:DM Sans,sans-serif; font-size:0.85rem; color:#ff6b8a;'>🚨 Spam</span>
                    <span style='font-family:JetBrains Mono,monospace; font-size:0.85rem; color:#ff6b8a;'>{spam}</span>
                </div>
                <div class='prob-bar-wrap'>
                    <div class='prob-bar-fill-spam' style='width:{(spam/total*100) if total>0 else 0:.1f}%'></div>
                </div>
            </div>
            <div>
                <div style='display:flex; justify-content:space-between; margin-bottom:0.5rem;'>
                    <span style='font-family:DM Sans,sans-serif; font-size:0.85rem; color:#4ade80;'>✅ Ham</span>
                    <span style='font-family:JetBrains Mono,monospace; font-size:0.85rem; color:#4ade80;'>{ham}</span>
                </div>
                <div class='prob-bar-wrap'>
                    <div class='prob-bar-fill-ham' style='width:{(ham/total*100) if total>0 else 0:.1f}%'></div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div style='background:#0d1420; border:1px solid rgba(255,255,255,0.07); border-radius:16px; padding:1.8rem;'>
            <div style='font-family:Syne,sans-serif; font-size:1rem; font-weight:700; color:#94a3b8; margin-bottom:1.5rem; text-transform:uppercase; letter-spacing:0.08em;'>
                Model Information
            </div>
        """, unsafe_allow_html=True)

        model_info = [
            ("Algorithm",    "Multinomial Naive Bayes"),
            ("Vectorizer",   "TF-IDF"),
            ("Preprocessing","NLTK + PorterStemmer"),
            ("Languages",    "English"),
            ("Input Type",   "SMS Text"),
            ("Output",       "Binary (Spam / Ham)"),
        ]
        rows = "".join([f"""
        <div class='analysis-row'>
            <span class='analysis-key'>{k}</span>
            <span class='analysis-val'>{v}</span>
        </div>""" for k, v in model_info])
        st.markdown(f"<div class='analysis-panel' style='margin-top:0;'>{rows}</div></div>", unsafe_allow_html=True)

    # Scan history table
    if st.session_state.history:
        st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)
        st.markdown("<div class='example-section-title'><span></span> Full Scan History</div>", unsafe_allow_html=True)
        for i, item in enumerate(reversed(st.session_state.history), 1):
            tag_class = "spam" if item["result"] == "SPAM" else "ham"
            st.markdown(f"""
            <div class='example-card'>
                <div style='display:flex; align-items:center; justify-content:space-between;'>
                    <div class='tag {tag_class}'>{item["result"]}</div>
                    <span style='font-family:JetBrains Mono,monospace; font-size:0.72rem; color:#334155;'>#{len(st.session_state.history)-i+1}</span>
                </div>
                <p style='margin-top:0.4rem;'>{item["message"]}</p>
                <div style='font-family:JetBrains Mono,monospace; font-size:0.68rem; color:#334155; margin-top:0.3rem;'>
                    Spam probability: {item["prob"]}%
                </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style='text-align:center; padding:3rem; color:#334155; font-family:DM Sans,sans-serif;'>
            No scan history yet. Go to Classifier to analyze messages.
        </div>
        """, unsafe_allow_html=True)


# ─── Page: Examples ─────────────────────────────────────────────────────────────
elif st.session_state.active_page == "Examples":
    st.markdown("""
    <div style='font-family:Syne,sans-serif; font-size:2rem; font-weight:800; color:#e2e8f0; margin-bottom:0.3rem;'>
        Example Messages
    </div>
    <div style='font-family:DM Sans,sans-serif; color:#475569; margin-bottom:2rem;'>
        Click any message to load it in the classifier
    </div>
    """, unsafe_allow_html=True)

    col_spam, col_ham = st.columns(2)

    with col_spam:
        st.markdown("<div class='example-section-title'><span></span> Spam Examples</div>", unsafe_allow_html=True)
        for msg in SPAM_EXAMPLES:
            if st.button(msg[:55] + "…" if len(msg) > 55 else msg, key=f"spam_{msg[:20]}", use_container_width=True):
                st.session_state.selected_example = msg
                st.session_state.active_page = "Classifier"
                st.rerun()

    with col_ham:
        st.markdown("<div class='example-section-title'><span></span> Legitimate Examples</div>", unsafe_allow_html=True)
        for msg in HAM_EXAMPLES:
            if st.button(msg[:55] + "…" if len(msg) > 55 else msg, key=f"ham_{msg[:20]}", use_container_width=True):
                st.session_state.selected_example = msg
                st.session_state.active_page = "Classifier"
                st.rerun()


# ─── Page: About ────────────────────────────────────────────────────────────────
elif st.session_state.active_page == "About":
    st.markdown("""
    <div style='font-family:Syne,sans-serif; font-size:2rem; font-weight:800; color:#e2e8f0; margin-bottom:0.3rem;'>
        About SpamShield
    </div>
    <div style='font-family:DM Sans,sans-serif; color:#475569; margin-bottom:2rem;'>
        Technical overview and project documentation
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style='background:#0d1420; border:1px solid rgba(255,255,255,0.07); border-radius:16px; padding:2rem; margin-bottom:1.5rem;'>
        <div style='font-family:Syne,sans-serif; font-size:1.1rem; font-weight:700; color:#00d2ff; margin-bottom:1rem;'>
            🔬 How It Works
        </div>
        <div style='font-family:DM Sans,sans-serif; font-size:0.9rem; color:#94a3b8; line-height:1.8;'>
            SpamShield uses a classic NLP pipeline combining TF-IDF vectorization with a Multinomial Naive Bayes classifier.
            Each message goes through tokenization, stopword removal, and Porter stemming before being converted into a
            numerical feature vector. The model then predicts class probabilities, providing both a binary label and
            a confidence score.
        </div>
    </div>
    """, unsafe_allow_html=True)

    pipeline_steps = [
        ("01", "Text Preprocessing", "Lowercase conversion, tokenization, alphanumeric filtering"),
        ("02", "Stopword Removal",   "Removes common words (the, is, at…) using NLTK corpus"),
        ("03", "Stemming",           "Porter Stemmer reduces words to their root form"),
        ("04", "TF-IDF Vectorization","Converts text into weighted numerical feature vectors"),
        ("05", "Classification",     "Multinomial Naive Bayes predicts spam probability"),
        ("06", "Result Output",      "Binary label + confidence percentage returned to UI"),
    ]

    for num, title, desc in pipeline_steps:
        st.markdown(f"""
        <div style='display:flex; gap:1.2rem; align-items:flex-start; padding:1rem 0;
                    border-bottom:1px solid rgba(255,255,255,0.04);'>
            <div style='font-family:JetBrains Mono,monospace; font-size:0.7rem; color:#00d2ff;
                        background:rgba(0,210,255,0.08); border:1px solid rgba(0,210,255,0.15);
                        border-radius:6px; padding:4px 8px; flex-shrink:0; margin-top:2px;'>
                {num}
            </div>
            <div>
                <div style='font-family:Syne,sans-serif; font-size:0.9rem; font-weight:600; color:#e2e8f0;'>{title}</div>
                <div style='font-family:DM Sans,sans-serif; font-size:0.82rem; color:#475569; margin-top:2px;'>{desc}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)


# ─── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class='footer'>
    <p>SPAMSHIELD — SMS SPAM CLASSIFIER</p>
    <div class='tech-tags'>
        <span class='tech-tag'>Streamlit</span>
        <span class='tech-tag'>Scikit-Learn</span>
        <span class='tech-tag'>NLTK</span>
        <span class='tech-tag'>TF-IDF</span>
        <span class='tech-tag'>Naive Bayes</span>
        <span class='tech-tag'>Python 3</span>
    </div>
    <p style='margin-top:0.8rem; font-size:0.62rem;'>Built with Streamlit · Scikit-Learn · NLP</p>
</div>
""", unsafe_allow_html=True)