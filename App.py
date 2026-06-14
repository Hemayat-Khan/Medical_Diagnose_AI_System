import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import deque
import heapq
import re

# ─────────────────────────────────────────────
# PAGE CONFIG  (must be first Streamlit call)
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Medical Diagnosis AI System",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# GLOBAL CSS  –  Black × Red dark theme
# ─────────────────────────────────────────────
st.markdown("""
<style>
/* ── Base ── */
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Inter:wght@300;400;500;600&display=swap');

html, body, [data-testid="stAppViewContainer"] {
    background-color: #0a0a0a !important;
    color: #e8e8e8 !important;
    font-family: 'Inter', sans-serif;
}
[data-testid="stSidebar"] {
    background: #0f0f0f !important;
    border-right: 1px solid #c0000040;
}
[data-testid="stSidebar"] * { color: #e8e8e8 !important; }
[data-testid="stHeader"] { background: transparent !important; }

/* ── Headings ── */
h1, h2, h3 { font-family: 'Orbitron', monospace !important; }
h1 { color: #ff1a1a !important; letter-spacing: 2px; }
h2 { color: #cc0000 !important; }
h3 { color: #ff4444 !important; }

/* ── Cards ── */
.card {
    background: #111111;
    border: 1px solid #cc000033;
    border-radius: 8px;
    padding: 18px 22px;
    margin-bottom: 14px;
    box-shadow: 0 0 12px #cc000015;
}
.card-red {
    background: linear-gradient(135deg, #1a0000 0%, #0f0f0f 100%);
    border-left: 3px solid #cc0000;
    border-radius: 6px;
    padding: 14px 18px;
    margin-bottom: 10px;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #8b0000, #cc0000) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 6px !important;
    font-weight: 600 !important;
    letter-spacing: 0.5px !important;
    transition: all 0.2s ease !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #cc0000, #ff1a1a) !important;
    box-shadow: 0 0 16px #cc000060 !important;
}

/* ── Inputs ── */
.stTextInput > div > div > input,
.stNumberInput > div > div > input,
.stTextArea textarea,
.stSelectbox > div > div {
    background-color: #1a1a1a !important;
    color: #e8e8e8 !important;
    border: 1px solid #cc000050 !important;
    border-radius: 6px !important;
}
.stMultiSelect > div > div {
    background-color: #1a1a1a !important;
    border: 1px solid #cc000050 !important;
}
.stSlider > div > div > div { background-color: #cc0000 !important; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background-color: #111111 !important;
    border-bottom: 1px solid #cc000040 !important;
}
.stTabs [data-baseweb="tab"] {
    color: #888 !important;
    font-family: 'Orbitron', monospace !important;
    font-size: 11px !important;
}
.stTabs [aria-selected="true"] {
    color: #ff1a1a !important;
    border-bottom: 2px solid #ff1a1a !important;
    background: transparent !important;
}

/* ── Metrics ── */
[data-testid="stMetricValue"] { color: #ff4444 !important; font-size: 28px !important; }
[data-testid="stMetricLabel"] { color: #888 !important; }

/* ── DataFrame ── */
[data-testid="stDataFrame"] { border: 1px solid #cc000030 !important; }

/* ── Expander ── */
.streamlit-expanderHeader {
    background-color: #1a0000 !important;
    color: #ff4444 !important;
    border: 1px solid #cc000040 !important;
    border-radius: 6px !important;
}

/* ── Alerts ── */
.stSuccess { background-color: #001a00 !important; border-left: 3px solid #00aa44 !important; }
.stWarning { background-color: #1a0e00 !important; border-left: 3px solid #ff8800 !important; }
.stError   { background-color: #1a0000 !important; border-left: 3px solid #cc0000 !important; }
.stInfo    { background-color: #00001a !important; border-left: 3px solid #0066cc !important; }

/* ── Sidebar radio ── */
.stRadio > div { gap: 6px !important; }
.stRadio label { color: #ccc !important; }

/* ── Hero banner ── */
.hero-banner {
    background: linear-gradient(135deg, #1a0000 0%, #0a0a0a 50%, #0f0000 100%);
    border: 1px solid #cc000050;
    border-radius: 10px;
    padding: 30px 36px;
    margin-bottom: 28px;
    text-align: center;
    box-shadow: 0 0 40px #cc000020;
}
.hero-title {
    font-family: 'Orbitron', monospace;
    font-size: 2.2rem;
    font-weight: 900;
    color: #ff1a1a;
    letter-spacing: 3px;
    margin-bottom: 6px;
}
.hero-sub {
    color: #888;
    font-size: 0.95rem;
    letter-spacing: 1px;
}
.pulse-dot {
    display: inline-block;
    width: 10px; height: 10px;
    background: #cc0000;
    border-radius: 50%;
    margin-right: 8px;
    animation: pulse 1.5s infinite;
}
@keyframes pulse {
    0%   { box-shadow: 0 0 0 0 #cc000060; }
    70%  { box-shadow: 0 0 0 10px #cc000000; }
    100% { box-shadow: 0 0 0 0 #cc000000; }
}

/* ── Result badge ── */
.badge-positive {
    display: inline-block;
    background: #8b000033;
    color: #ff4444;
    border: 1px solid #cc0000;
    border-radius: 20px;
    padding: 4px 14px;
    font-size: 0.85rem;
    font-weight: 600;
}
.badge-negative {
    display: inline-block;
    background: #00330033;
    color: #44cc44;
    border: 1px solid #00aa44;
    border-radius: 20px;
    padding: 4px 14px;
    font-size: 0.85rem;
    font-weight: 600;
}

/* ── Plot background ── */
.stPlotlyChart, .stPyplot { background: transparent !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# MATPLOTLIB DARK THEME
# ─────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "#111111",
    "axes.facecolor": "#0f0f0f",
    "axes.edgecolor": "#333333",
    "axes.labelcolor": "#cccccc",
    "xtick.color": "#888888",
    "ytick.color": "#888888",
    "text.color": "#cccccc",
    "grid.color": "#1a1a1a",
    "grid.linewidth": 0.6,
})

# ═══════════════════════════════════════════════════════════════
#  BACK-END LOGIC
# ═══════════════════════════════════════════════════════════════

# ── Symptom / Disease KB ──────────────────────────────────────
SYMPTOM_DISEASE_MAP = {
    "fever": ["Flu", "Malaria", "Typhoid", "COVID-19"],
    "cough": ["Flu", "Bronchitis", "COVID-19", "Tuberculosis"],
    "headache": ["Migraine", "Flu", "Hypertension", "Meningitis"],
    "chest pain": ["Heart Disease", "Angina", "Pneumonia", "Anxiety"],
    "fatigue": ["Anemia", "Diabetes", "Flu", "Thyroid Disorder"],
    "shortness of breath": ["Asthma", "Heart Disease", "Pneumonia", "COVID-19"],
    "nausea": ["Gastritis", "Food Poisoning", "Hepatitis", "Pregnancy"],
    "joint pain": ["Arthritis", "Gout", "Lupus", "Flu"],
    "rash": ["Allergy", "Eczema", "Psoriasis", "Measles"],
    "dizziness": ["Vertigo", "Hypertension", "Anemia", "Migraine"],
    "abdominal pain": ["Appendicitis", "Gastritis", "IBS", "Kidney Stone"],
    "sore throat": ["Flu", "Strep Throat", "Tonsillitis", "COVID-19"],
}

DISEASE_INFO = {
    "Flu": {"severity": "Moderate", "specialist": "General Physician"},
    "Malaria": {"severity": "High", "specialist": "Infectious Disease"},
    "Typhoid": {"severity": "High", "specialist": "Infectious Disease"},
    "COVID-19": {"severity": "Variable", "specialist": "Pulmonologist"},
    "Migraine": {"severity": "Moderate", "specialist": "Neurologist"},
    "Heart Disease": {"severity": "Critical", "specialist": "Cardiologist"},
    "Diabetes": {"severity": "High", "specialist": "Endocrinologist"},
    "Asthma": {"severity": "Moderate", "specialist": "Pulmonologist"},
    "Anemia": {"severity": "Moderate", "specialist": "Hematologist"},
    "Hypertension": {"severity": "High", "specialist": "Cardiologist"},
    "Arthritis": {"severity": "Moderate", "specialist": "Rheumatologist"},
    "Gastritis": {"severity": "Low", "specialist": "Gastroenterologist"},
    "Appendicitis": {"severity": "Critical", "specialist": "Surgeon"},
    "Pneumonia": {"severity": "High", "specialist": "Pulmonologist"},
    "Tuberculosis": {"severity": "High", "specialist": "Pulmonologist"},
    "Allergy": {"severity": "Low", "specialist": "Allergist"},
}

SYMPTOM_GRAPH = {
    "fever": ["cough", "headache", "fatigue", "sore throat"],
    "cough": ["fever", "shortness of breath", "sore throat"],
    "headache": ["fever", "dizziness", "nausea"],
    "chest pain": ["shortness of breath", "dizziness", "fatigue"],
    "fatigue": ["fever", "joint pain", "headache"],
    "shortness of breath": ["chest pain", "cough", "fatigue"],
    "nausea": ["abdominal pain", "dizziness", "headache"],
    "joint pain": ["rash", "fatigue", "fever"],
    "rash": ["joint pain", "fever"],
    "dizziness": ["headache", "nausea", "chest pain"],
    "abdominal pain": ["nausea", "fever"],
    "sore throat": ["fever", "cough"],
}


def bfs_search(start_symptom, target_symptom):
    visited, queue, path = set(), deque([[start_symptom]]), []
    while queue:
        path = queue.popleft()
        node = path[-1]
        if node == target_symptom:
            return path
        if node not in visited:
            visited.add(node)
            for nb in SYMPTOM_GRAPH.get(node, []):
                if nb not in visited:
                    queue.append(path + [nb])
    return None


def dfs_search(start_symptom, target_symptom, visited=None, path=None):
    if visited is None: visited = set()
    if path is None: path = [start_symptom]
    if start_symptom == target_symptom:
        return path
    visited.add(start_symptom)
    for nb in SYMPTOM_GRAPH.get(start_symptom, []):
        if nb not in visited:
            result = dfs_search(nb, target_symptom, visited, path + [nb])
            if result:
                return result
    return None


def heuristic(symptom, target):
    s1 = set(SYMPTOM_GRAPH.get(symptom, []))
    s2 = set(SYMPTOM_GRAPH.get(target, []))
    return max(0, 3 - len(s1 & s2))


def astar_search(start, goal):
    heap = [(0, 0, [start])]
    visited = set()
    while heap:
        f, g, path = heapq.heappop(heap)
        node = path[-1]
        if node == goal:
            return path
        if node in visited:
            continue
        visited.add(node)
        for nb in SYMPTOM_GRAPH.get(node, []):
            if nb not in visited:
                ng = g + 1
                h = heuristic(nb, goal)
                heapq.heappush(heap, (ng + h, ng, path + [nb]))
    return None


def diagnose(symptoms):
    disease_count = {}
    for sym in symptoms:
        for dis in SYMPTOM_DISEASE_MAP.get(sym.lower(), []):
            disease_count[dis] = disease_count.get(dis, 0) + 1
    ranked = sorted(disease_count.items(), key=lambda x: x[1], reverse=True)
    return ranked


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def logistic_regression_train(X, y, lr=0.05, epochs=200):
    m, n = X.shape
    theta = np.zeros(n)
    losses = []
    for _ in range(epochs):
        pred = sigmoid(X @ theta)
        pred = np.clip(pred, 1e-9, 1 - 1e-9)
        loss = -np.mean(y * np.log(pred) + (1 - y) * np.log(1 - pred))
        losses.append(loss)
        theta -= lr * (X.T @ (pred - y)) / m
    return theta, losses



def kmeans(X, k=3, iters=50):
    np.random.seed(42)
    idx = np.random.choice(len(X), k, replace=False)
    centers = X[idx].copy()
    labels = np.zeros(len(X), dtype=int)
    for _ in range(iters):
        dists = np.linalg.norm(X[:, None] - centers[None, :], axis=2)
        labels = np.argmin(dists, axis=1)
        for ki in range(k):
            pts = X[labels == ki]
            if len(pts): centers[ki] = pts.mean(axis=0)
    return labels, centers


def ac3_solve():
    """Mini CSP: assign 3 symptoms with mutual-exclusion constraint."""
    domain = {0: [0, 1], 1: [0, 1], 2: [0, 1]}
    solutions = []
    for v0 in domain[0]:
        for v1 in domain[1]:
            if v0 != v1 or v0 == 0:  # arc: var0 ≠ var1 unless both 0
                for v2 in domain[2]:
                    if v1 + v2 <= 1:  # arc: at most one of var1,var2 is 1
                        solutions.append((v0, v1, v2))
    return solutions


# ═══════════════════════════════════════════════════════════════
#  SIDEBAR
# ═══════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding:16px 0 8px'>
        <span style='font-family:Orbitron; font-size:1.2rem; color:#ff1a1a; font-weight:900;'>
            🩺 MED-AI
        </span><br>
        <span style='color:#555; font-size:0.75rem;'>Diagnosis System v1.0</span>
    </div>
    <hr style='border-color:#cc000030; margin:8px 0 16px'>
    """, unsafe_allow_html=True)

    module = st.radio(
        "NAVIGATION",
        ["🏠 Dashboard",
         "🔍 Symptom Checker",
         "🗺️ Graph Search (BFS/DFS/A*)",
         "📈 Logistic Regression",
         "🔵 K-Means Clustering",
         "🧩 CSP / AC-3",
         "🧠 FOL Knowledge Base",
         "📋 Patient Report"],
    )

    st.markdown("<hr style='border-color:#cc000030; margin:16px 0'>", unsafe_allow_html=True)
    st.markdown("""
    <div class='card-red' style='font-size:0.78rem; color:#888;'>
        <b style='color:#ff4444;'>Project:</b> Medical Diagnosis AI<br>
        <b style='color:#ff4444;'>Student:</b> Hemayat — 2430-0198<br>
        <b style='color:#ff4444;'>Institute:</b> CASE, Islamabad<br>
        <b style='color:#ff4444;'>Course:</b> AI 2101
    </div>
    """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
#  HERO BANNER
# ═══════════════════════════════════════════════════════════════

st.markdown("""
<div class='hero-banner'>
    <div class='hero-title'>🩺 MEDICAL DIAGNOSIS AI SYSTEM</div>
    <div class='hero-sub'>
        <span class='pulse-dot'></span>
        BFS · DFS · A* · Logistic Regression · K-Means · CSP/AC-3 · FOL · Agent Types
    </div>
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
#  MODULE:  DASHBOARD
# ═══════════════════════════════════════════════════════════════

if module == "🏠 Dashboard":
    col1, col2, col3, col4 = st.columns(4)
    metrics = [
        ("Symptoms in KB", len(SYMPTOM_DISEASE_MAP)),
        ("Diseases Tracked", len(DISEASE_INFO)),
        ("AI Modules", 7),
        ("Agent Types", 4),
    ]
    for col, (label, val) in zip([col1, col2, col3, col4], metrics):
        col.metric(label, val)

    st.markdown("---")

    st.markdown("### 🧱 System Architecture")
    cols = st.columns(3)
    modules_info = [
        ("🗺️ Search Algorithms", "BFS, DFS, A* used to traverse symptom graphs and find related conditions."),
        ("📈 ML Classifiers", "Logistic Regression trained from scratch with gradient descent for disease prediction."),
        ("🔵 Clustering", "K-Means clusters patients by symptom profile for risk-group identification."),
        ("🧩 CSP / AC-3",
         "Constraint Satisfaction Problem solver with Arc Consistency to handle diagnosis constraints."),
        ("🧠 FOL Knowledge Base",
         "First-Order Logic rules encode medical knowledge: ∀x Symptom(x,fever) → Possible(x,flu)."),
        ("🤖 Agent Types",
         "Simple Reflex, Model-based, Goal-based, and Utility-based agents for clinical decision-making."),
    ]
    for i, (title, desc) in enumerate(modules_info):
        with cols[i % 3]:
            st.markdown(f"""
            <div class='card'>
                <b style='color:#ff4444'>{title}</b>
                <p style='color:#888; font-size:0.85rem; margin-top:8px'>{desc}</p>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("### 🤖 Agent Types in Use")
    agents = {
        "Simple Reflex Agent": "Responds to current symptom only. Rule: IF fever → possible flu.",
        "Model-Based Agent": "Maintains patient history to refine diagnosis over multiple visits.",
        "Goal-Based Agent": "Plans diagnostic steps toward confirming or ruling out a target disease.",
        "Utility-Based Agent": "Maximises expected utility (accuracy × cost) when recommending tests.",
    }
    a_cols = st.columns(2)
    for i, (name, desc) in enumerate(agents.items()):
        with a_cols[i % 2]:
            st.markdown(f"""
            <div class='card-red'>
                <b style='color:#ff4444'>{name}</b>
                <p style='color:#aaa; font-size:0.83rem; margin:6px 0 0'>{desc}</p>
            </div>
            """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
#  MODULE:  SYMPTOM CHECKER
# ═══════════════════════════════════════════════════════════════

elif module == "🔍 Symptom Checker":
    st.markdown("## 🔍 Symptom Checker")
    st.markdown("Select all symptoms the patient is experiencing:")

    all_symptoms = list(SYMPTOM_DISEASE_MAP.keys())
    selected = st.multiselect("Symptoms", options=all_symptoms,
                              placeholder="Choose one or more symptoms…")

    age = st.slider("Patient Age", 1, 100, 30)
    temp = st.slider("Body Temperature (°C)", 35.0, 42.0, 37.0, 0.1)

    if st.button("🔬 Run Diagnosis"):
        if not selected:
            st.warning("Please select at least one symptom.")
        else:
            results = diagnose(selected)
            st.markdown("### 📊 Diagnosis Results")

            if results:
                top_disease, top_score = results[0]
                info = DISEASE_INFO.get(top_disease, {})
                sev = info.get("severity", "Unknown")
                spec = info.get("specialist", "General Physician")

                c1, c2, c3 = st.columns(3)
                c1.metric("Top Diagnosis", top_disease)
                c2.metric("Severity", sev)
                c3.metric("See Specialist", spec)

                with st.expander("📋 All Ranked Diagnoses"):
                    df = pd.DataFrame(results, columns=["Disease", "Symptom Matches"])
                    df["Severity"] = df["Disease"].apply(lambda d: DISEASE_INFO.get(d, {}).get("severity", "—"))
                    df["Specialist"] = df["Disease"].apply(lambda d: DISEASE_INFO.get(d, {}).get("specialist", "—"))
                    st.dataframe(df, use_container_width=True)

                # Bar chart
                fig, ax = plt.subplots(figsize=(8, 3))
                diseases = [r[0] for r in results[:6]]
                scores = [r[1] for r in results[:6]]
                colors = ["#cc0000" if i == 0 else "#660000" for i in range(len(diseases))]
                ax.barh(diseases[::-1], scores[::-1], color=colors[::-1])
                ax.set_xlabel("Symptom Match Score")
                ax.set_title("Diagnosis Ranking", color="#ff4444", fontsize=12)
                fig.tight_layout()
                st.pyplot(fig)

                # Fever alert
                if temp >= 38.5:
                    st.error(f"⚠️ High temperature detected ({temp}°C) — urgent evaluation advised.")
                elif temp >= 37.5:
                    st.warning(f"🌡 Slightly elevated temperature ({temp}°C).")
            else:
                st.info("No diseases matched the selected symptoms.")

# ═══════════════════════════════════════════════════════════════
#  MODULE:  GRAPH SEARCH
# ═══════════════════════════════════════════════════════════════

elif module == "🗺️ Graph Search (BFS/DFS/A*)":
    st.markdown("## 🗺️ Symptom Graph Search")
    st.markdown("Explore how symptoms are connected using classic search algorithms.")

    all_syms = list(SYMPTOM_GRAPH.keys())
    c1, c2 = st.columns(2)
    start = c1.selectbox("Start Symptom", all_syms, index=0)
    target = c2.selectbox("Target Symptom", all_syms, index=3)
    algo = st.radio("Algorithm", ["BFS", "DFS", "A*"], horizontal=True)

    if st.button("🔍 Find Path"):
        if start == target:
            st.info("Start and target are the same node.")
        else:
            if algo == "BFS":
                path = bfs_search(start, target)
            elif algo == "DFS":
                path = dfs_search(start, target)
            else:
                path = astar_search(start, target)

            if path:
                st.success(f"✅ Path found using {algo}  ({len(path) - 1} edges)")
                steps_html = " → ".join(
                    [f"<b style='color:#ff4444'>{s}</b>" for s in path]
                )
                st.markdown(f"<div class='card'>{steps_html}</div>", unsafe_allow_html=True)

                # Visualise path
                fig, ax = plt.subplots(figsize=(max(6, len(path) * 1.8), 2.5))
                for i, node in enumerate(path):
                    color = "#cc0000" if node in (path[0], path[-1]) else "#440000"
                    ax.add_patch(plt.Circle((i * 2, 0), 0.55, color=color, zorder=3))
                    ax.text(i * 2, 0, node[:8], ha='center', va='center',
                            fontsize=7.5, color='white', zorder=4)
                    if i < len(path) - 1:
                        ax.annotate("", xy=((i + 1) * 2 - 0.6, 0), xytext=(i * 2 + 0.6, 0),
                                    arrowprops=dict(arrowstyle="->", color="#ff4444", lw=1.5))
                ax.set_xlim(-1, (len(path) - 1) * 2 + 1)
                ax.set_ylim(-1, 1)
                ax.axis("off")
                ax.set_title(f"{algo} Path: {path[0]} → {path[-1]}", color="#ff4444")
                fig.tight_layout()
                st.pyplot(fig)
            else:
                st.error(f"No path found between '{start}' and '{target}' using {algo}.")

    with st.expander("📌 Symptom Graph Connections"):
        rows = [{"Symptom": k, "Connected To": ", ".join(v)}
                for k, v in SYMPTOM_GRAPH.items()]
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

# ═══════════════════════════════════════════════════════════════
#  MODULE:  LOGISTIC REGRESSION
# ═══════════════════════════════════════════════════════════════

elif module == "📈 Logistic Regression":
    st.markdown("## 📈 Logistic Regression — Disease Predictor")
    st.markdown("Train a binary classifier (Heart Disease: Yes/No) from scratch using gradient descent.")

    n_samples = st.slider("Training samples", 100, 1000, 300)
    lr = st.select_slider("Learning rate", [0.001, 0.005, 0.01, 0.05, 0.1], value=0.05)
    epochs = st.slider("Epochs", 50, 500, 200)

    if st.button("🏋️ Train Model"):
        np.random.seed(42)
        # Synthetic: features = [age_norm, chol_norm, bp_norm]
        X_raw = np.random.randn(n_samples, 3)
        y = ((X_raw[:, 0] * 0.5 + X_raw[:, 1] * 0.4 + X_raw[:, 2] * 0.3 + np.random.randn(n_samples) * 0.3) > 0).astype(
            float)
        X = np.c_[np.ones(n_samples), X_raw]

        theta, losses = logistic_regression_train(X, y, lr=lr, epochs=epochs)

        preds = (sigmoid(X @ theta) >= 0.5).astype(float)
        accuracy = np.mean(preds == y) * 100

        m1, m2, m3 = st.columns(3)
        m1.metric("Accuracy", f"{accuracy:.1f}%")
        m2.metric("Epochs", epochs)
        m3.metric("Final Loss", f"{losses[-1]:.4f}")

        fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
        axes[0].plot(losses, color="#cc0000", lw=2)
        axes[0].set_title("Training Loss", color="#ff4444")
        axes[0].set_xlabel("Epoch");
        axes[0].set_ylabel("BCE Loss")
        axes[0].grid(True, alpha=0.3)

        probs = sigmoid(X @ theta)
        axes[1].hist(probs[y == 0], bins=20, alpha=0.7, color="#004400", label="No Disease")
        axes[1].hist(probs[y == 1], bins=20, alpha=0.7, color="#cc0000", label="Disease")
        axes[1].axvline(0.5, color="#ff8800", linestyle="--", label="Threshold")
        axes[1].set_title("Predicted Probability Distribution", color="#ff4444")
        axes[1].legend(fontsize=8)
        fig.tight_layout()
        st.pyplot(fig)

        st.markdown("### 🧪 Try a New Patient")
        nc1, nc2, nc3 = st.columns(3)
        age_n = nc1.slider("Age (normalised)", -2.0, 2.0, 0.5, 0.1)
        chol_n = nc2.slider("Cholesterol (normalised)", -2.0, 2.0, 0.3, 0.1)
        bp_n = nc3.slider("Blood Pressure (normalised)", -2.0, 2.0, 0.0, 0.1)

        x_new = np.array([1, age_n, chol_n, bp_n])
        prob = sigmoid(np.dot(theta, x_new))

        if prob >= 0.5:
            st.markdown(f"**Prediction:** <span class='badge-positive'>Heart Disease ({prob * 100:.1f}%)</span>",
                        unsafe_allow_html=True)
        else:
            st.markdown(f"**Prediction:** <span class='badge-negative'>No Disease ({(1 - prob) * 100:.1f}%)</span>",
                        unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
#  MODULE:  K-MEANS CLUSTERING
# ═══════════════════════════════════════════════════════════════

elif module == "🔵 K-Means Clustering":
    st.markdown("## 🔵 K-Means Patient Clustering")
    st.markdown("Cluster patients by their symptom-feature profile.")

    n_pts = st.slider("Number of patients", 50, 400, 150)
    k = st.slider("Number of clusters (K)", 2, 6, 3)

    if st.button("🔵 Run K-Means"):
        np.random.seed(7)
        X = np.vstack([
            np.random.randn(n_pts // 3, 2) + [0, 0],
            np.random.randn(n_pts // 3, 2) + [4, 4],
            np.random.randn(n_pts // 3, 2) + [-3, 4],
        ])
        labels, centers = kmeans(X, k=k)

        cluster_names = ["Low Risk", "Moderate Risk", "High Risk",
                         "Critical", "Acute", "Chronic"][:k]

        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        reds = ["#cc0000", "#8b0000", "#ff4444", "#440000", "#ff8888", "#660000"]
        for ki in range(k):
            mask = labels == ki
            axes[0].scatter(X[mask, 0], X[mask, 1], c=reds[ki], s=25,
                            alpha=0.7, label=cluster_names[ki])
        axes[0].scatter(centers[:, 0], centers[:, 1], c="white", marker="X",
                        s=150, zorder=5, edgecolors="#ff4444", linewidths=1.5)
        axes[0].set_title("Patient Clusters", color="#ff4444")
        axes[0].legend(fontsize=8)
        axes[0].grid(True, alpha=0.2)

        counts = [np.sum(labels == ki) for ki in range(k)]
        axes[1].bar(cluster_names, counts, color=reds[:k])
        axes[1].set_title("Cluster Distribution", color="#ff4444")
        axes[1].set_ylabel("Patients")
        fig.tight_layout()
        st.pyplot(fig)

        df = pd.DataFrame({
            "Cluster": cluster_names,
            "Patients": counts,
            "Centroid X": [f"{c[0]:.2f}" for c in centers],
            "Centroid Y": [f"{c[1]:.2f}" for c in centers],
        })
        st.dataframe(df, use_container_width=True)

# ═══════════════════════════════════════════════════════════════
#  MODULE:  CSP / AC-3
# ═══════════════════════════════════════════════════════════════

elif module == "🧩 CSP / AC-3":
    st.markdown("## 🧩 Constraint Satisfaction Problem — AC-3 Solver")
    st.markdown("""
    **Diagnosis CSP:** Three binary variables represent whether a symptom is *present (1)* or *absent (0)*.

    **Constraints:**
    - Arc(Var₀, Var₁): `Var₀ ≠ Var₁`  unless both are 0
    - Arc(Var₁, Var₂): `Var₁ + Var₂ ≤ 1`  (at most one active)
    """)

    variables = {
        "Var₀ (Fever)": st.selectbox("Var₀ (Fever)", [0, 1], index=0),
        "Var₁ (Cough)": st.selectbox("Var₁ (Cough)", [0, 1], index=0),
        "Var₂ (ChestPain)": st.selectbox("Var₂ (Chest Pain)", [0, 1], index=0),
    }

    if st.button("⚙️ Run AC-3"):
        solutions = ac3_solve()
        st.success(f"✅ {len(solutions)} consistent assignment(s) found by AC-3.")

        df = pd.DataFrame(solutions, columns=["Var₀ (Fever)", "Var₁ (Cough)", "Var₂ (Chest Pain)"])
        df["Valid"] = "✅"
        st.dataframe(df, use_container_width=True)

        v0 = variables["Var₀ (Fever)"]
        v1 = variables["Var₁ (Cough)"]
        v2 = variables["Var₂ (ChestPain)"]
        chosen = (v0, v1, v2)
        if chosen in solutions:
            st.success(f"Your selection {chosen} is **consistent** with all constraints.")
        else:
            st.error(f"Your selection {chosen} **violates** one or more constraints.")

    with st.expander("📖 What is AC-3?"):
        st.markdown("""
        **AC-3 (Arc Consistency Algorithm 3)** enforces arc consistency by
        removing values from variable domains that have no valid support in
        the neighbouring variable's domain.  

        Each *arc* `(Xᵢ, Xⱼ)` is checked: for every value in `domain(Xᵢ)`,
        there must exist at least one value in `domain(Xⱼ)` that satisfies the
        constraint. If a value is removed, all arcs pointing to `Xᵢ` are
        re-added to the queue. The algorithm runs in **O(cd³)** time.
        """)

# ═══════════════════════════════════════════════════════════════
#  MODULE:  FOL KNOWLEDGE BASE
# ═══════════════════════════════════════════════════════════════

elif module == "🧠 FOL Knowledge Base":
    st.markdown("## 🧠 First-Order Logic Knowledge Base")

    fol_rules = [
        ("R1", "∀x  HasSymptom(x, fever) ∧ HasSymptom(x, cough)  →  Possible(x, Flu)"),
        ("R2", "∀x  HasSymptom(x, chest_pain) ∧ HasSymptom(x, shortness_of_breath)  →  Possible(x, HeartDisease)"),
        ("R3", "∀x  HasSymptom(x, fever) ∧ HasSymptom(x, joint_pain)  →  Possible(x, Malaria)"),
        ("R4", "∀x  HasSymptom(x, cough) ∧ HasSymptom(x, shortness_of_breath)  →  Possible(x, Asthma)"),
        ("R5", "∀x  HasSymptom(x, fatigue) ∧ HasSymptom(x, dizziness)  →  Possible(x, Anemia)"),
        ("R6", "∀x  Possible(x, HeartDisease) ∧ Age(x) > 50  →  HighRisk(x)"),
        ("R7", "∀x  HighRisk(x)  →  Recommend(x, CardiologistConsult)"),
        ("R8", "∀x  Possible(x, Flu)  →  Recommend(x, RestAndHydration)"),
    ]

    st.markdown("### 📜 Medical FOL Rules")
    for rule_id, rule in fol_rules:
        st.markdown(f"""
        <div class='card-red'>
            <code style='color:#ff4444'>[{rule_id}]</code>
            <span style='color:#ccc; margin-left:10px; font-family:monospace'>{rule}</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🔎 FOL Inference Engine")
    patient_syms = st.multiselect("Patient Symptoms",
                                  ["fever", "cough", "chest pain", "shortness of breath",
                                   "joint pain", "fatigue", "dizziness"], [])
    age_fol = st.number_input("Patient Age", 1, 100, 35)

    if st.button("🧠 Run FOL Inference"):
        facts = set(patient_syms)
        inferred = []

        if "fever" in facts and "cough" in facts:
            inferred.append(("Possible(patient, Flu)", "R1"))
            inferred.append(("Recommend(patient, RestAndHydration)", "R8"))
        if "chest pain" in facts and "shortness of breath" in facts:
            inferred.append(("Possible(patient, HeartDisease)", "R2"))
            if age_fol > 50:
                inferred.append(("HighRisk(patient)", "R6"))
                inferred.append(("Recommend(patient, CardiologistConsult)", "R7"))
        if "fever" in facts and "joint pain" in facts:
            inferred.append(("Possible(patient, Malaria)", "R3"))
        if "cough" in facts and "shortness of breath" in facts:
            inferred.append(("Possible(patient, Asthma)", "R4"))
        if "fatigue" in facts and "dizziness" in facts:
            inferred.append(("Possible(patient, Anemia)", "R5"))

        if inferred:
            st.markdown("#### ✅ Inferred Facts")
            for fact, rule in inferred:
                st.markdown(f"""
                <div class='card'>
                    <code style='color:#ff4444'>[{rule}]</code>
                    <span style='color:#e8e8e8; margin-left:10px'>{fact}</span>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No rules fired. Try adding more symptoms.")

# ═══════════════════════════════════════════════════════════════
#  MODULE:  PATIENT REPORT
# ═══════════════════════════════════════════════════════════════

elif module == "📋 Patient Report":
    st.markdown("## 📋 Patient Report Generator")

    with st.form("patient_form"):
        c1, c2 = st.columns(2)
        name = c1.text_input("Patient Name", "Ali Hassan")
        age = c2.number_input("Age", 1, 100, 40)
        gender = c1.selectbox("Gender", ["Male", "Female", "Other"])
        btemp = c2.number_input("Temperature (°C)", 35.0, 42.0, 37.2)
        syms = st.multiselect("Symptoms", list(SYMPTOM_DISEASE_MAP.keys()))
        notes = st.text_area("Doctor Notes", "Patient presented with…")
        submitted = st.form_submit_button("📄 Generate Report")

    if submitted:
        results = diagnose(syms) if syms else []
        top = results[0][0] if results else "Inconclusive"
        info = DISEASE_INFO.get(top, {"severity": "—", "specialist": "—"})

        st.markdown(f"""
        <div class='card' style='border-left:3px solid #cc0000'>
            <div style='font-family:Orbitron; color:#ff4444; font-size:1.1rem; margin-bottom:14px'>
                🩺 DIAGNOSTIC REPORT
            </div>
            <table style='width:100%; border-collapse:collapse; font-size:0.9rem'>
                <tr>
                    <td style='color:#888; padding:5px 12px 5px 0'>Patient</td>
                    <td style='color:#fff; font-weight:600'>{name}</td>
                    <td style='color:#888; padding:5px 12px 5px 20px'>Age / Gender</td>
                    <td style='color:#fff'>{age} / {gender}</td>
                </tr>
                <tr>
                    <td style='color:#888; padding:5px 12px 5px 0'>Temperature</td>
                    <td style='color:{"#ff4444" if btemp >= 38.5 else "#fff"}'>{btemp}°C</td>
                    <td style='color:#888; padding:5px 12px 5px 20px'>Symptoms</td>
                    <td style='color:#fff'>{", ".join(syms) if syms else "—"}</td>
                </tr>
                <tr>
                    <td style='color:#888; padding:5px 12px 5px 0'>Primary Diagnosis</td>
                    <td style='color:#ff4444; font-weight:700'>{top}</td>
                    <td style='color:#888; padding:5px 12px 5px 20px'>Severity</td>
                    <td style='color:#fff'>{info["severity"]}</td>
                </tr>
                <tr>
                    <td style='color:#888; padding:5px 12px 5px 0'>Refer To</td>
                    <td style='color:#fff'>{info["specialist"]}</td>
                    <td style='color:#888; padding:5px 12px 5px 20px'>Confidence</td>
                    <td style='color:#fff'>{results[0][1] if results else "—"} symptom match(es)</td>
                </tr>
            </table>
            <div style='margin-top:14px; border-top:1px solid #cc000030; padding-top:12px'>
                <span style='color:#888'>Doctor Notes:</span>
                <p style='color:#ccc; margin-top:4px'>{notes}</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

        if results:
            st.markdown("#### All Ranked Diagnoses")
            df = pd.DataFrame(results, columns=["Disease", "Score"])
            df["Severity"] = df["Disease"].apply(lambda d: DISEASE_INFO.get(d, {}).get("severity", "—"))
            df["Specialist"] = df["Disease"].apply(lambda d: DISEASE_INFO.get(d, {}).get("specialist", "—"))
            st.dataframe(df, use_container_width=True)

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("""
<hr style='border-color:#cc000020; margin-top:40px'>
<div style='text-align:center; color:#333; font-size:0.75rem; padding-bottom:12px'>
    Medical Diagnosis AI System · CASE Islamabad · AI 2101 · Built with Streamlit
</div>
""", unsafe_allow_html=True)


