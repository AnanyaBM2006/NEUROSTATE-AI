import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import tempfile
import os
import time
import json
import smtplib
from email.message import EmailMessage

import sys
from pathlib import Path

# Make dashboard/ available for local module imports
DASHBOARD_DIR = Path(__file__).resolve().parent
if str(DASHBOARD_DIR) not in sys.path:
    sys.path.insert(0, str(DASHBOARD_DIR))

# MindLink is available for the local live demo.
# The deployed Streamlit app can still run without the headset.
try:
    from live.mindlink import MindLinkReader
    MINDLINK_AVAILABLE = True
except Exception:
    MindLinkReader = None
    MINDLINK_AVAILABLE = False
from utils.eeg_features import extract_features
from utils.csv_pipeline import preprocess_csv
from utils.model_pipeline import run_models


st.set_page_config(
    page_title="NeuroState AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================================
# CONFIG
# ============================================================

DURATIONS = {
    "1 Minute": 60,
    "5 Minutes": 300,
    "10 Minutes": 600,
    "20 Minutes": 1200,
    "30 Minutes": 1800,
    "45 Minutes": 2700,
    "1 Hour": 3600,
    "1.5 Hours": 5400,
    "2 Hours": 7200,
}

WINDOW_SIZE = 1024
EEG_FS = 512.0
NO_SIGNAL_TIMEOUT = 3.0
CONNECTION_TIMEOUT = 20.0
AGE_CLASSES = ["18-30", "31-50"]

# ============================================================
# SESSION STATE
# ============================================================

state_defaults = {
    "running": False,
    "reader": None,
    "duration_label": "1 Minute",
    "duration_seconds": 60,
    "session_started": None,
    "session_raw": [],
    "processed_raw_count": 0,
    "session_features": [],
    "session_attention": [],
    "session_meditation": [],
    "last_sample_count": 0,
    "last_sample_time": None,
    "connection_wait_started": None,
    "signal_lost": False,
    "session_completed": False,
    "final_results": None,
    "live_model_results": None,
    "last_model_feature_count": 0,
    "manual_results": None,
    "manual_features": None,
    "uploaded_name": None,
}

for key, value in state_defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value

# ============================================================
# CSS
# ============================================================

css = (
    "<style>"
    ".stApp{background:linear-gradient(180deg,#FFFFFF 0%,#F8FAFC 100%);color:#0F172A;}"
    ".block-container{max-width:1500px;padding-top:1.2rem;padding-bottom:3rem;}"

    ".title{font-size:46px;font-weight:900;color:#0F172A;letter-spacing:-1.5px;}"
    ".sub{color:#475569;font-size:15px;font-weight:600;}"

    ".section{font-size:26px;font-weight:900;margin-top:30px;margin-bottom:14px;color:#0F172A;}"

    ".status-card{border-radius:16px;padding:15px 20px;"
    "border:1px solid #BFDBFE;background:linear-gradient(90deg,#EFF6FF,#F8FAFC);"
    "color:#0F172A;font-weight:800;box-shadow:0 4px 14px rgba(37,99,235,.08);}"

    ".live-card{border-radius:18px;padding:18px 20px;background:#FFFFFF;"
    "border:1px solid #E2E8F0;box-shadow:0 6px 22px rgba(15,23,42,.06);}"

    ".metric-card{border-radius:16px;padding:17px 18px;background:#FFFFFF;"
    "border:1px solid #E2E8F0;box-shadow:0 5px 18px rgba(15,23,42,.05);}"
    ".metric-label{font-size:12px;font-weight:900;letter-spacing:1px;color:#64748B;text-transform:uppercase;}"
    ".metric-value{font-size:32px;font-weight:900;color:#0F172A;margin-top:5px;}"

    ".state{font-size:42px;font-weight:900;color:#0F172A;}"
    ".conf{color:#475569;font-size:13px;font-weight:700;}"

    ".ai-panel{border-radius:18px;padding:22px;background:#FFFFFF;"
    "border:1px solid #E2E8F0;box-shadow:0 7px 24px rgba(15,23,42,.06);min-height:220px;}"
    ".coach-title{color:#166534;font-size:24px;font-weight:900;}"
    ".scientist-title{color:#1D4ED8;font-size:24px;font-weight:900;}"

    ".badge{display:inline-block;margin-top:12px;padding:7px 13px;border-radius:20px;"
    "background:#ECFDF5;border:1px solid #86EFAC;color:#166534;font-size:11px;font-weight:900;}"
    ".advice{font-size:25px;font-weight:900;margin-top:14px;color:#0F172A;line-height:1.2;}"
    ".reason{color:#475569;font-size:14px;line-height:1.55;margin-top:9px;}"

    ".wave-card{border-radius:15px;padding:15px;background:#FFFFFF;"
    "border:1px solid #E2E8F0;box-shadow:0 4px 15px rgba(15,23,42,.04);}"
    ".wave-name{color:#64748B;font-size:11px;font-weight:900;letter-spacing:1px;}"
    ".wave-value{font-size:24px;font-weight:900;color:#0F172A;margin:4px 0 8px;}"

    ".age-card{border-radius:18px;padding:22px;background:linear-gradient(135deg,#EFF6FF,#FFFFFF);"
    "border:1px solid #BFDBFE;box-shadow:0 6px 22px rgba(37,99,235,.07);}"
    ".age-number{font-size:46px;font-weight:900;color:#1D4ED8;}"
    ".age-label{color:#475569;font-size:13px;margin-top:3px;font-weight:700;}"
    ".prob-label{font-size:12px;font-weight:800;color:#475569;}"

    ".summary-card{border-radius:18px;padding:20px;"
    "border:1px solid #E2E8F0;background:#FFFFFF;color:#0F172A;"
    "box-shadow:0 6px 20px rgba(15,23,42,.05);}"

    ".stMetric{background:#FFFFFF!important;border:1px solid #E2E8F0!important;"
    "border-radius:15px!important;padding:14px!important;box-shadow:0 4px 14px rgba(15,23,42,.04)!important;}"
    "[data-testid='stMetricLabel']{color:#475569!important;font-weight:800!important;}"
    "[data-testid='stMetricValue']{color:#0F172A!important;font-weight:900!important;}"
    "[data-testid='stMetricDelta']{color:#475569!important;}"

    ".stProgress>div>div>div>div{background:#2563EB!important;}"
    ".stButton>button{background:linear-gradient(135deg,#2563EB,#4F46E5)!important;"
    "color:#FFFFFF!important;border:1px solid #1D4ED8!important;border-radius:12px;"
    "font-weight:900;min-height:44px;box-shadow:0 5px 14px rgba(37,99,235,.18);}"
    ".stButton>button:hover{background:linear-gradient(135deg,#1D4ED8,#4338CA)!important;"
    "color:#FFFFFF!important;}"

    ".stSelectbox label,.stFileUploader label{color:#0F172A!important;font-weight:800!important;}"
    ".stMarkdown,.stText{color:#0F172A;}"
    "h1,h2,h3,h4,h5,h6,p,label{color:#0F172A;}"

    "</style>"
)

st.markdown(css, unsafe_allow_html=True)
# Only the uploader button gets this color override.
st.markdown('<style>\n[data-testid="stFileUploader"] button {\n    background: #2563EB !important;\n    color: #FFFFFF !important;\n    border: 1px solid #2563EB !important;\n}\n[data-testid="stFileUploader"] button p,\n[data-testid="stFileUploader"] button span {\n    color: #FFFFFF !important;\n}\n[data-testid="stFileUploader"] button:hover {\n    background: #1D4ED8 !important;\n    color: #FFFFFF !important;\n}\n</style>', unsafe_allow_html=True)


# ============================================================
# HELPERS
# ============================================================

def reset_live_session():
    st.session_state.session_raw = []
    st.session_state.processed_raw_count = 0
    st.session_state.session_features = []
    st.session_state.session_attention = []
    st.session_state.session_meditation = []
    st.session_state.last_sample_count = 0
    st.session_state.last_sample_time = time.time()
    st.session_state.connection_wait_started = time.time()
    st.session_state.signal_lost = False
    st.session_state.session_completed = False
    st.session_state.final_results = None
    st.session_state.live_model_results = None
    st.session_state.last_model_feature_count = 0


def stop_reader():
    reader = st.session_state.reader
    if reader is not None:
        try:
            reader.stop()
        except Exception:
            pass
    st.session_state.reader = None


def live_feature_df():
    if not st.session_state.session_features:
        return pd.DataFrame()
    return pd.DataFrame(st.session_state.session_features)


def live_models():
    """
    Live AI inference.

    The dashboard refreshes every second, so the latest EEG feature
    windows are passed through the age and cognitive-state models on
    every refresh. This keeps the displayed prediction and confidence
    synchronized with the live EEG stream.
    """
    features = live_feature_df()

    if features.empty:
        return None, features

    try:
        results = run_models(features)
        st.session_state.live_model_results = results
        return results, features
    except Exception:
        # Keep the previous valid result visible if one inference cycle
        # temporarily fails.
        return st.session_state.get("live_model_results"), features


def finalize_session():
    results, _ = live_models()
    st.session_state.final_results = results if results is not None else st.session_state.get("live_model_results")
    st.session_state.running = False
    st.session_state.session_completed = True
    stop_reader()


def get_wave_values(features):
    if features is None or features.empty:
        return {
            "delta_norm": 0.0,
            "theta_norm": 0.0,
            "alpha_norm": 0.0,
            "beta_norm": 0.0,
            "gamma_norm": 0.0,
        }
    row = features.iloc[-1]
    return {
        "delta_norm": float(row.get("delta_norm", 0.0)),
        "theta_norm": float(row.get("theta_norm", 0.0)),
        "alpha_norm": float(row.get("alpha_norm", 0.0)),
        "beta_norm": float(row.get("beta_norm", 0.0)),
        "gamma_norm": float(row.get("gamma_norm", 0.0)),
    }


def render_graph(values):
    if values is None or len(values) == 0:
        st.info("Waiting for EEG samples...")
        return

    arr = np.asarray(values, dtype=float)
    x = np.arange(len(arr))

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=x,
            y=arr,
            mode="lines",
            name="EEG",
            line=dict(color="#2563EB", width=2.2),
            hovertemplate="Sample: %{x}<br>EEG: %{y:.2f}<extra></extra>",
        )
    )

    fig.add_hline(
        y=0,
        line_width=1,
        line_color="#64748B",
        opacity=0.75,
    )

    fig.update_layout(
        height=380,
        margin=dict(l=65, r=25, t=25, b=55),
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        font=dict(
            color="#334155",
            family="Arial, sans-serif",
            size=13,
        ),
        showlegend=False,
        hovermode="x unified",
        xaxis=dict(
            title=dict(
                text="Recent EEG Samples",
                font=dict(color="#334155", size=14),
            ),
            tickfont=dict(color="#475569", size=12),
            gridcolor="#E2E8F0",
            gridwidth=1,
            zeroline=False,
            showline=True,
            linecolor="#94A3B8",
            mirror=False,
        ),
        yaxis=dict(
            title=dict(
                text="EEG Amplitude (µV)",
                font=dict(color="#334155", size=14),
            ),
            tickfont=dict(color="#475569", size=12),
            gridcolor="#E2E8F0",
            gridwidth=1,
            zeroline=False,
            showline=True,
            linecolor="#94A3B8",
            nticks=8,
        ),
    )

    st.plotly_chart(
        fig,
        width="stretch",
        config={
            "displayModeBar": False,
            "responsive": True,
        },
    )


def speak_text(text):
    safe = json.dumps(" ".join(str(text).split()))
    st.components.v1.html(
        "<script>"
        f"const m=new SpeechSynthesisUtterance({safe});"
        "m.rate=.95;m.pitch=1;m.volume=1;"
        "window.speechSynthesis.cancel();"
        "window.speechSynthesis.speak(m);"
        "</script>",
        height=0
    )


# ============================================================
# AI DATA VISUALS
# ============================================================

def render_coach_visual(state_result):
    """Small visual representation of the cognitive-state model output."""
    labels = ["Fatigued", "Focused", "Moderate"]
    values = []

    if state_result and state_result.get("available"):
        probabilities = state_result.get("probabilities")
        if probabilities is not None:
            values = [float(x) * 100 for x in probabilities]

    if not values:
        attention = (
            float(st.session_state.session_attention[-1])
            if st.session_state.session_attention
            else 0.0
        )
        meditation = (
            float(st.session_state.session_meditation[-1])
            if st.session_state.session_meditation
            else 0.0
        )
        labels = ["Attention", "Meditation"]
        values = [attention, meditation]

    fig = go.Figure(
        go.Bar(
            x=values,
            y=labels,
            orientation="h",
            text=[f"{v:.1f}%" for v in values],
            textposition="auto",
            marker=dict(color="#22C55E"),
            hovertemplate="%{y}: %{x:.1f}%<extra></extra>",
        )
    )
    fig.update_layout(
        height=170,
        margin=dict(l=10, r=10, t=8, b=8),
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        font=dict(color="#334155", size=11),
        xaxis=dict(
            range=[0, 100],
            showgrid=True,
            gridcolor="#E2E8F0",
            tickfont=dict(color="#64748B", size=10),
            title=None,
        ),
        yaxis=dict(
            showgrid=False,
            tickfont=dict(color="#334155", size=10),
            title=None,
        ),
        showlegend=False,
    )
    st.plotly_chart(
        fig,
        width="stretch",
        config={"displayModeBar": False, "responsive": True},
    )


def render_neuroscientist_visual(waves):
    """Visual representation of the five EEG band features."""
    labels = ["Delta", "Theta", "Alpha", "Beta", "Gamma"]
    values = [
        float(waves["delta_norm"]) * 100,
        float(waves["theta_norm"]) * 100,
        float(waves["alpha_norm"]) * 100,
        float(waves["beta_norm"]) * 100,
        float(waves["gamma_norm"]) * 100,
    ]

    fig = go.Figure(
        go.Bar(
            x=labels,
            y=values,
            text=[f"{v:.1f}%" for v in values],
            textposition="outside",
            marker=dict(color="#2563EB"),
            hovertemplate="%{x}: %{y:.2f}%<extra></extra>",
        )
    )
    fig.update_layout(
        height=190,
        margin=dict(l=10, r=10, t=20, b=20),
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        font=dict(color="#334155", size=11),
        xaxis=dict(
            showgrid=False,
            tickfont=dict(color="#334155", size=10),
            title=None,
        ),
        yaxis=dict(
            range=[0, max(10, max(values) * 1.25)],
            showgrid=True,
            gridcolor="#E2E8F0",
            tickfont=dict(color="#64748B", size=10),
            title=None,
        ),
        showlegend=False,
    )
    st.plotly_chart(
        fig,
        width="stretch",
        config={"displayModeBar": False, "responsive": True},
    )


def render_ai(waves, state_result):
    st.markdown(
        '<div class="section">🤖 AI Intelligence</div>',
        unsafe_allow_html=True
    )

    coach, scientist = st.columns(2)

    state = (
        state_result["state"]
        if state_result and state_result.get("available")
        else "Unavailable"
    )

    # ---------------- Cognitive Coach ----------------
    with coach:
        st.markdown(
            '<div class="ai-panel">'
            '<div class="coach-title">🟢 AI Cognitive Coach</div>'
            '<div class="conf">Live recommendation • updates every second</div>',
            unsafe_allow_html=True
        )

        if state == "Focused":
            badge = "STABLE FOCUS"
            advice = "Excellent focus. Keep going."
            action = "Continue your current task while attention remains stable."
        elif state == "Moderate":
            badge = "ATTENTION DRIFTING"
            advice = "Attention is moderately engaged."
            action = "Use a short reset if your attention begins to drift."
        elif state == "Fatigued":
            badge = "FOCUS DROPOUT"
            advice = "Reduced cognitive engagement is detected."
            action = "Take a short break before continuing demanding work."
        else:
            badge = "COLLECTING EEG"
            advice = "The cognitive model is collecting EEG windows."
            action = "Continue the session so the AI can establish a stronger pattern."

        st.markdown(
            f'<div class="badge">🟢 {badge}</div>',
            unsafe_allow_html=True
        )
        st.markdown(
            f'<div class="advice">{advice}</div>',
            unsafe_allow_html=True
        )
        st.markdown(
            f'<div class="reason">{action}</div>',
            unsafe_allow_html=True
        )

        if st.button(
            "🔊 Speak AI Recommendation",
            key="voice_coach",
            width="stretch",
        ):
            speak_text(
                f"NeuroState AI cognitive coach. {advice} {action}"
            )

        st.markdown(
            '<div style="font-size:12px;font-weight:900;color:#64748B;margin-top:14px;">'
            'COGNITIVE MODEL VISUAL'
            '</div>',
            unsafe_allow_html=True,
        )
        render_coach_visual(state_result)

        st.markdown('</div>', unsafe_allow_html=True)

    # ---------------- AI Neuroscientist ----------------
    with scientist:
        st.markdown(
            '<div class="ai-panel">'
            '<div class="scientist-title">🔬 AI Neuroscientist</div>'
            '<div class="conf">Live EEG interpretation • updates every second</div>',
            unsafe_allow_html=True
        )

        values = {
            "Delta": waves["delta_norm"],
            "Theta": waves["theta_norm"],
            "Alpha": waves["alpha_norm"],
            "Beta": waves["beta_norm"],
            "Gamma": waves["gamma_norm"],
        }

        dominant = max(values, key=values.get)

        if values["Beta"] > values["Theta"]:
            interpretation = "ACTIVE COGNITIVE ENGAGEMENT"
            explanation = (
                "Beta activity is stronger than theta activity "
                "in the latest feature window."
            )
        elif values["Theta"] > values["Beta"]:
            interpretation = "LOWER ALERTNESS PATTERN"
            explanation = (
                "Theta activity is stronger than beta activity "
                "in the latest feature window."
            )
        else:
            interpretation = "BALANCED COGNITIVE ACTIVITY"
            explanation = (
                "Beta and theta activity are relatively balanced."
            )

        st.markdown(
            f'<div class="badge">🔵 {interpretation}</div>',
            unsafe_allow_html=True
        )
        st.markdown(
            f'<div class="advice">{dominant} activity is dominant</div>',
            unsafe_allow_html=True
        )
        st.markdown(
            f'<div class="reason">{explanation}<br><br>'
            'This is an AI-based EEG interpretation, not a clinical diagnosis.'
            '</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div style="font-size:12px;font-weight:900;color:#64748B;margin-top:14px;">'
            'LIVE EEG BAND PROFILE'
            '</div>',
            unsafe_allow_html=True,
        )
        render_neuroscientist_visual(waves)

        st.markdown('</div>', unsafe_allow_html=True)


# ============================================================
# LIVE SCREEN
# ============================================================

def render_live_screen():
    latest, results, features, signal_problem, duration_finished = process_live()

    if signal_problem:
        st.session_state.signal_lost = True
        finalize_session()
        st.rerun()

    if duration_finished:
        finalize_session()
        st.rerun()

    raw = latest.get("raw_eeg")
    attention = latest.get("attention")
    meditation = latest.get("meditation")
    samples = int(latest.get("sample_count") or 0)
    quality = latest.get("signal_quality")

    if quality is None:
        status = "🟢 MindLink Connected"
    else:
        status = f"🟢 MindLink Connected • Signal Quality: {quality}"

    st.markdown(
        f'<div class="status-card">{status}</div>',
        unsafe_allow_html=True
    )

    elapsed = int(
        time.time() - st.session_state.session_started
    )
    remaining = max(
        0,
        st.session_state.duration_seconds - elapsed
    )
    mm, ss = divmod(remaining, 60)

    st.markdown(
        f'<div class="status-card">'
        f'⏱️ <b>{st.session_state.duration_label}</b>'
        f' &nbsp; • &nbsp; '
        f'⏳ <b>{mm:02d}:{ss:02d}</b> remaining'
        f' &nbsp; • &nbsp; '
        f'📡 <b>{samples:,}</b> EEG samples'
        f'</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section">📡 Live EEG Signal</div>',
        unsafe_allow_html=True
    )

    # Main live telemetry cards
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-label">Raw EEG</div>'
            f'<div class="metric-value">'
            f'{"N/A" if raw is None else f"{raw:.0f}"}'
            f'</div></div>',
            unsafe_allow_html=True,
        )

    with c2:
        st.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-label">Attention</div>'
            f'<div class="metric-value">'
            f'{"N/A" if attention is None else f"{attention:.0f}"}'
            f'</div></div>',
            unsafe_allow_html=True,
        )

    with c3:
        st.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-label">Meditation</div>'
            f'<div class="metric-value">'
            f'{"N/A" if meditation is None else f"{meditation:.0f}"}'
            f'</div></div>',
            unsafe_allow_html=True,
        )

    with c4:
        st.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-label">EEG Samples</div>'
            f'<div class="metric-value">{samples:,}</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown(
        '<div style="height:10px"></div>',
        unsafe_allow_html=True
    )

    # Session telemetry
    t1, t2, t3 = st.columns(3)
    with t1:
        st.metric("Session Duration", st.session_state.duration_label)
    with t2:
        st.metric("Time Remaining", f"{mm:02d}:{ss:02d}")
    with t3:
        st.metric("Feature Windows", len(features))

    st.markdown(
        '<div class="section">📈 Raw EEG Signal</div>',
        unsafe_allow_html=True
    )
    render_graph(st.session_state.session_raw[-1000:])

    st.markdown(
        '<div class="section">🌊 Brainwave Activity</div>',
        unsafe_allow_html=True
    )

    waves = get_wave_values(features)
    wave_list = [
        ("DELTA", waves["delta_norm"]),
        ("THETA", waves["theta_norm"]),
        ("ALPHA", waves["alpha_norm"]),
        ("BETA", waves["beta_norm"]),
        ("GAMMA", waves["gamma_norm"]),
    ]

    columns = st.columns(5)
    wave_descriptions = {
        "DELTA": "Deep/restorative activity",
        "THETA": "Memory & internal processing",
        "ALPHA": "Relaxed alertness",
        "BETA": "Active attention",
        "GAMMA": "Complex processing",
    }

    for col, (name, value) in zip(columns, wave_list):
        with col:
            st.markdown(
                '<div class="wave-card">'
                f'<div class="wave-name">{name}</div>'
                f'<div class="wave-value">{value * 100:.2f}%</div>'
                f'<div style="font-size:11px;color:#64748B;min-height:30px;">'
                f'{wave_descriptions[name]}</div>',
                unsafe_allow_html=True
            )
            st.progress(min(max(float(value), 0), 1))
            st.markdown('</div>', unsafe_allow_html=True)

    # ---------------- Neural Age ----------------

    st.markdown(
        '<div class="section">👤 Live Neural Age</div>',
        unsafe_allow_html=True
    )

    if results and results.get("age"):
        age = results["age"]
        confidence = float(age.get("confidence", 0))
        probabilities = age.get("probabilities")

        a1, a2 = st.columns([1, 2])

        with a1:
            st.markdown(
                '<div class="age-card">'
                '<div class="metric-label">Predicted Neural Age Group</div>'
                f'<div class="age-number">{age.get("age_group", "N/A")}</div>'
                f'<div class="age-label">Model confidence: {confidence * 100:.2f}%</div>'
                '</div>',
                unsafe_allow_html=True
            )

        with a2:
            st.markdown(
                '<div class="summary-card">'
                '<div class="metric-label">AGE MODEL CONFIDENCE</div>',
                unsafe_allow_html=True
            )

            st.progress(min(max(confidence, 0), 1))

            if probabilities is not None:
                pc1, pc2 = st.columns(2)

                for col, label, probability in zip(
                    [pc1, pc2],
                    AGE_CLASSES,
                    probabilities,
                ):
                    with col:
                        st.markdown(
                            f'<div class="prob-label">{label}</div>'
                            f'<div style="font-size:25px;font-weight:900;color:#1D4ED8;">'
                            f'{float(probability) * 100:.2f}%</div>',
                            unsafe_allow_html=True,
                        )

            st.markdown('</div>', unsafe_allow_html=True)

    else:
        st.info("Collecting the first valid EEG feature window...")

    # ---------------- Cognitive State ----------------

    st.markdown(
        '<div class="section">🧠 Live Cognitive State</div>',
        unsafe_allow_html=True
    )

    state_result = (
        results.get("state")
        if results
        else None
    )

    if state_result and state_result.get("available"):
        state = state_result["state"]
        confidence = float(state_result["confidence"])
        probabilities = state_result.get("probabilities")

        s1, s2 = st.columns([1, 2])

        with s1:
            st.markdown(
                '<div class="ai-panel">'
                '<div class="metric-label">CURRENT COGNITIVE STATE</div>'
                f'<div class="state">{state}</div>'
                f'<div class="conf">Model confidence • {confidence * 100:.2f}%</div>'
                '</div>',
                unsafe_allow_html=True
            )

        with s2:
            st.markdown(
                '<div class="summary-card">'
                '<div class="metric-label">COGNITIVE STATE CONFIDENCE</div>',
                unsafe_allow_html=True
            )

            st.progress(min(max(confidence, 0), 1))

            if probabilities is not None:
                labels = ["Fatigued", "Focused", "Moderate"]
                cols = st.columns(3)

                for col, label, probability in zip(
                    cols,
                    labels,
                    probabilities,
                ):
                    with col:
                        st.markdown(
                            f'<div class="prob-label">{label}</div>'
                            f'<div style="font-size:22px;font-weight:900;color:#0F172A;">'
                            f'{float(probability) * 100:.1f}%</div>',
                            unsafe_allow_html=True,
                        )

            st.markdown('</div>', unsafe_allow_html=True)

    else:
        st.info(
            f"Cognitive-state model is collecting EEG windows: "
            f"{len(features)}/20"
        )

    render_ai(waves, state_result)


# ============================================================
# MANUAL UPLOAD
# ============================================================

def render_manual(results, features, raw_data):
    st.markdown(
        '<div class="status-card">📂 EEG recording loaded successfully</div>',
        unsafe_allow_html=True
    )

    data = raw_data.copy()
    data.columns = [
        str(c).strip().lower()
        for c in data.columns
    ]

    raw_column = next(
        (
            c for c in [
                "raw_eeg",
                "eegrawvalue",
                "eeg",
                "eeg_value",
                "eegvalue",
                "signal"
            ]
            if c in data.columns
        ),
        None
    )

    st.markdown(
        '<div class="section">📈 EEG Signal</div>',
        unsafe_allow_html=True
    )

    if raw_column:
        values = pd.to_numeric(
            data[raw_column],
            errors="coerce"
        ).dropna().tail(1000).to_numpy()
        render_graph(values)
    else:
        st.info("No raw EEG column found for visualization.")

    st.markdown(
        '<div class="section">🌊 Brainwave Activity</div>',
        unsafe_allow_html=True
    )

    waves = get_wave_values(features)
    columns = st.columns(5)

    for col, (name, value) in zip(
        columns,
        [
            ("DELTA", waves["delta_norm"]),
            ("THETA", waves["theta_norm"]),
            ("ALPHA", waves["alpha_norm"]),
            ("BETA", waves["beta_norm"]),
            ("GAMMA", waves["gamma_norm"]),
        ]
    ):
        with col:
            st.markdown(
                f'<div class="wave-name">{name}</div>',
                unsafe_allow_html=True
            )
            st.markdown(
                f'<div class="wave-value">{value * 100:.2f}%</div>',
                unsafe_allow_html=True
            )
            st.progress(min(max(float(value), 0), 1))

    age = results.get("age") if results else None

    st.markdown(
        '<div class="section">👤 Neural Age</div>',
        unsafe_allow_html=True
    )

    if age:
        confidence = float(age.get("confidence", 0))
        st.markdown(
            f'<div class="age-number">{age.get("age_group", "N/A")}</div>',
            unsafe_allow_html=True
        )
        st.markdown(
            f'<div class="age-label">'
            f'MindLink XGBoost Neural Age Model • '
            f'Confidence {confidence * 100:.2f}%'
            f'</div>',
            unsafe_allow_html=True
        )
        st.progress(min(max(confidence, 0), 1))

        probabilities = age.get("probabilities")
        if probabilities is not None:
            for label, probability in zip(
                AGE_CLASSES, probabilities
            ):
                st.write(
                    f"{label}: {float(probability) * 100:.2f}%"
                )

    state_result = results.get("state") if results else None

    st.markdown(
        '<div class="section">🧠 Cognitive State</div>',
        unsafe_allow_html=True
    )

    if state_result and state_result.get("available"):
        confidence = float(state_result["confidence"])
        st.markdown(
            f'<div class="state">{state_result["state"]}</div>',
            unsafe_allow_html=True
        )
        st.markdown(
            f'<div class="conf">'
            f'Model confidence • {confidence * 100:.2f}%'
            f'</div>',
            unsafe_allow_html=True
        )
        st.progress(min(max(confidence, 0), 1))
    else:
        st.info(
            f"Cognitive-state model is collecting EEG windows: "
            f"{len(features)}/20"
        )

    render_ai(waves, state_result)



# ============================================================
# EMAIL REPORT
# ============================================================

def _secret_value(section, key, default=None):
    """Read an email setting from Streamlit secrets without exposing it in the UI."""
    try:
        section_data = st.secrets.get(section, {})
        if hasattr(section_data, "get"):
            return section_data.get(key, default)
    except Exception:
        pass
    return default


def build_report_text(results, source_label, duration_label, sample_count,
                      feature_count, avg_attention, avg_meditation, waves):
    lines = [
        "NeuroState AI — EEG Analysis Report",
        "=" * 42,
        f"Data Source: {source_label}",
        f"Analysis Duration: {duration_label}",
        f"EEG Samples: {sample_count:,}",
        f"Feature Windows: {feature_count}",
        f"Average Attention: {'N/A' if avg_attention is None else f'{avg_attention:.1f}'}",
        f"Average Meditation: {'N/A' if avg_meditation is None else f'{avg_meditation:.1f}'}",
        "",
        "Brainwave Activity",
        "------------------",
        f"Delta: {waves.get('delta_norm', 0) * 100:.2f}%",
        f"Theta: {waves.get('theta_norm', 0) * 100:.2f}%",
        f"Alpha: {waves.get('alpha_norm', 0) * 100:.2f}%",
        f"Beta: {waves.get('beta_norm', 0) * 100:.2f}%",
        f"Gamma: {waves.get('gamma_norm', 0) * 100:.2f}%",
        "",
    ]

    if results and results.get("age"):
        age = results["age"]
        lines.extend([
            "Neural Age",
            "-----------",
            f"Age Group: {age.get('age_group', 'N/A')}",
            f"Confidence: {float(age.get('confidence', 0)) * 100:.2f}%",
            "",
        ])

    if results and results.get("state") and results["state"].get("available"):
        state = results["state"]
        lines.extend([
            "Cognitive State",
            "---------------",
            f"State: {state.get('state', 'N/A')}",
            f"Confidence: {float(state.get('confidence', 0)) * 100:.2f}%",
            "",
        ])

    lines.extend([
        "NeuroState AI uses EEG-derived features and trained models for analysis.",
        "This report is for AI-based analysis and is not a clinical diagnosis.",
    ])
    return "\n".join(lines)


def send_email_report(report_text, recipient):
    """Send a session report to the email entered by the current user.

    Only the NeuroState AI sender credentials are stored in Streamlit secrets.
    The recipient is supplied at runtime, so the app can be used by many people.
    """
    from email.utils import parseaddr

    smtp_server = _secret_value("email", "smtp_server", "smtp.gmail.com")
    smtp_port = int(_secret_value("email", "smtp_port", 465))
    sender = _secret_value("email", "sender_email")
    password = _secret_value("email", "sender_password")

    missing = []
    if not sender:
        missing.append("sender_email")
    if not password:
        missing.append("sender_password")

    if missing:
        return False, (
            "Email sender is not configured. Add the following values to "
            ".streamlit/secrets.toml: " + ", ".join(missing)
        )

    recipient = str(recipient or "").strip()
    parsed_name, parsed_email = parseaddr(recipient)
    if not parsed_email or "@" not in parsed_email or "." not in parsed_email.rsplit("@", 1)[-1]:
        return False, "Please enter a valid email address."

    try:
        message = EmailMessage()
        message["Subject"] = "NeuroState AI — EEG Analysis Report"
        message["From"] = sender
        message["To"] = parsed_email
        message.set_content(report_text)

        if smtp_port == 465:
            with smtplib.SMTP_SSL(smtp_server, smtp_port, timeout=20) as smtp:
                smtp.login(sender, password)
                smtp.send_message(message)
        else:
            with smtplib.SMTP(smtp_server, smtp_port, timeout=20) as smtp:
                smtp.ehlo()
                smtp.starttls()
                smtp.ehlo()
                smtp.login(sender, password)
                smtp.send_message(message)

        return True, f"Report sent successfully to {parsed_email}."
    except Exception as error:
        return False, f"Could not send email: {error}"


def render_email_report(results, source_label, duration_label, sample_count,
                        feature_count, avg_attention, avg_meditation, waves, key):
    """Email the current user's session report without changing the fixed dashboard UI."""
    report_text = build_report_text(
        results,
        source_label,
        duration_label,
        sample_count,
        feature_count,
        avg_attention,
        avg_meditation,
        waves,
    )

    show_form_key = f"{key}_show_form"
    email_key = f"{key}_recipient"
    send_key = f"{key}_send"

    if show_form_key not in st.session_state:
        st.session_state[show_form_key] = False

    if st.button("📧 Send Report", key=key, width="stretch"):
        st.session_state[show_form_key] = True

    if st.session_state[show_form_key]:
        st.markdown("**Email Session Report**")
        recipient = st.text_input(
            "Enter your email address",
            key=email_key,
            placeholder="you@example.com",
        )

        if st.button("Send Report", key=send_key, width="stretch"):
            with st.spinner("Sending NeuroState AI report..."):
                success, message = send_email_report(report_text, recipient)
            if success:
                st.success(message)
                st.session_state[show_form_key] = False
            else:
                st.error(message)


# ============================================================
# FINAL SUMMARY
# ============================================================

def render_final_summary():
    if not st.session_state.session_completed:
        return

    st.markdown(
        '<div class="section">📊 Final Session Summary</div>',
        unsafe_allow_html=True
    )

    results = st.session_state.final_results

    avg_attention = (
        np.mean(st.session_state.session_attention)
        if st.session_state.session_attention
        else None
    )

    avg_meditation = (
        np.mean(st.session_state.session_meditation)
        if st.session_state.session_meditation
        else None
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        "Session Duration",
        st.session_state.duration_label
    )
    c2.metric(
        "EEG Samples",
        f"{len(st.session_state.session_raw):,}"
    )
    c3.metric(
        "Feature Windows",
        str(len(st.session_state.session_features))
    )
    c4.metric(
        "Average Attention",
        "N/A" if avg_attention is None else f"{avg_attention:.1f}"
    )

    st.markdown(
        '<div class="summary-card">',
        unsafe_allow_html=True
    )

    if results and results.get("age"):
        age = results["age"]
        st.write(
            f"**Neural Age:** {age.get('age_group', 'N/A')} — "
            f"{float(age.get('confidence', 0)) * 100:.2f}% confidence"
        )

    if (
        results
        and results.get("state")
        and results["state"].get("available")
    ):
        state = results["state"]
        st.write(
            f"**Cognitive State:** {state.get('state', 'N/A')} — "
            f"{float(state.get('confidence', 0)) * 100:.2f}% confidence"
        )

    st.write(
        "**Average Meditation:** "
        + (
            "N/A"
            if avg_meditation is None
            else f"{avg_meditation:.1f}"
        )
    )

    st.markdown(
        '</div>',
        unsafe_allow_html=True
    )

    render_email_report(
        results,
        "Live MindLink",
        st.session_state.duration_label,
        len(st.session_state.session_raw),
        len(st.session_state.session_features),
        avg_attention,
        avg_meditation,
        get_wave_values(live_feature_df()),
        "email_live_report",
    )


# ============================================================
# HEADER
# ============================================================

left, right = st.columns([4, 1])

with left:
    st.markdown(
        '<div class="title">🧠 NeuroState AI</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<div class="sub">'
        'EEG Cognitive Intelligence • State Analysis • '
        'Neural Age • Forecasting'
        '</div>',
        unsafe_allow_html=True
    )

with right:
    if st.session_state.running:
        st.markdown(
            '<div style="text-align:right;color:#67E8F9;'
            'font-weight:900;padding-top:12px;">'
            '● LIVE SESSION</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            '<div style="text-align:right;color:#64748B;'
            'font-weight:900;padding-top:12px;">'
            '● READY</div>',
            unsafe_allow_html=True
        )

st.markdown("---")


# ============================================================
# CONTROLS
# ============================================================

c1, c2, c3 = st.columns([1.5, 1.5, 2])

with c1:
    duration_label = st.selectbox(
        "Analysis Duration",
        list(DURATIONS.keys()),
        index=0,
        disabled=st.session_state.running
    )

with c2:
    source = st.selectbox(
        "Data Source",
        ["Live MindLink", "Manual EEG Upload"],
        disabled=st.session_state.running
    )

with c3:
    uploaded = None
    if source == "Manual EEG Upload":
        uploaded = st.file_uploader(
            "Upload EEG CSV",
            type=["csv"]
        )


# ============================================================
# START / STOP
# ============================================================

if source == "Live MindLink":

    if not st.session_state.running:

        if st.button(
            "🧠 Start Analysis",
            width="stretch"
        ):
            reset_live_session()

            st.session_state.duration_label = duration_label
            st.session_state.duration_seconds = DURATIONS[
                duration_label
            ]

            if not MINDLINK_AVAILABLE:
             st.warning(
               "Live MindLink hardware is available only when running NeuroState AI locally. "
               "Please use CSV Upload for the deployed demo."
             )
            else:
             reader = MindLinkReader(save_csv=True)

            st.session_state.reader = reader
            st.session_state.session_started = time.time()
            st.session_state.running = True

            reader.start()

            st.rerun()

    else:

        if st.button(
            "⏹ Stop Analysis",
            width="stretch"
        ):
            finalize_session()
            st.rerun()


# ============================================================
# MANUAL CSV PROCESSING
# ============================================================

manual_results = None
manual_features = None
manual_data = None

if (
    source == "Manual EEG Upload"
    and uploaded is not None
):

    if st.session_state.uploaded_name != uploaded.name:

        temp_path = None

        try:
            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".csv"
            ) as temp_file:

                temp_file.write(
                    uploaded.getbuffer()
                )

                temp_path = temp_file.name

            _, manual_features = preprocess_csv(
                temp_path
            )

            manual_results = run_models(
                manual_features
            )

            manual_data = pd.read_csv(
                temp_path
            )

            st.session_state.manual_results = manual_results
            st.session_state.manual_features = manual_features
            st.session_state.uploaded_name = uploaded.name

        except Exception as error:
            st.error(
                f"EEG processing failed: {error}"
            )

        finally:
            if temp_path:
                try:
                    os.remove(temp_path)
                except Exception:
                    pass

    else:
        manual_results = st.session_state.manual_results
        manual_features = st.session_state.manual_features

    if manual_results is not None and manual_features is not None:
        if manual_data is None:
            manual_data = pd.read_csv(uploaded)

        render_manual(
            manual_results,
            manual_features,
            manual_data
        )

        manual_attention = None
        manual_meditation = None
        if manual_data is not None:
            manual_columns = {str(c).strip().lower(): c for c in manual_data.columns}
            attention_col = next((manual_columns[k] for k in ["attention"] if k in manual_columns), None)
            meditation_col = next((manual_columns[k] for k in ["meditation"] if k in manual_columns), None)
            if attention_col is not None:
                manual_attention_values = pd.to_numeric(manual_data[attention_col], errors="coerce").dropna()
                if not manual_attention_values.empty:
                    manual_attention = float(manual_attention_values.mean())
            if meditation_col is not None:
                manual_meditation_values = pd.to_numeric(manual_data[meditation_col], errors="coerce").dropna()
                if not manual_meditation_values.empty:
                    manual_meditation = float(manual_meditation_values.mean())

        render_email_report(
            manual_results,
            "Manual EEG Upload",
            duration_label,
            len(manual_data) if manual_data is not None else 0,
            len(manual_features),
            manual_attention,
            manual_meditation,
            get_wave_values(manual_features),
            "email_manual_report",
        )


# ============================================================
# LIVE FRAGMENT
# ============================================================

if (
    source == "Live MindLink"
    and st.session_state.running
):

    @st.fragment(run_every="1s")
    def live_fragment():
        render_live_screen()

    live_fragment()

elif (
    source == "Live MindLink"
    and st.session_state.session_completed
):

    st.markdown(
        '<div class="status-card">'
        '⏹ Live EEG session stopped.'
        '</div>',
        unsafe_allow_html=True
    )

    render_final_summary()

else:

    st.markdown(
        '<div class="status-card">'
        '🟡 Waiting for MindLink... '
        'Select a duration and press Start Analysis.'
        '</div>',
        unsafe_allow_html=True
    )


# ============================================================
# INFORMATION
# ============================================================

st.markdown(
    '<div class="section">📚 Brainwave Guide & NeuroState AI</div>',
    unsafe_allow_html=True
)

info1, info2 = st.columns(2)

with info1:
    st.markdown(
        '<div class="summary-card">'
        '<b>🧠 EEG Bands</b><br><br>'
        '<b>Delta</b> — very slow activity.<br>'
        '<b>Theta</b> — slower activity associated with memory '
        'and relaxed states.<br>'
        '<b>Alpha</b> — commonly seen during relaxed wakefulness.<br>'
        '<b>Beta</b> — faster activity associated with alertness '
        'and active cognitive processing.<br>'
        '<b>Gamma</b> — very fast activity associated with '
        'higher-level information processing.'
        '</div>',
        unsafe_allow_html=True
    )

with info2:
    st.markdown(
        '<div class="summary-card">'
        '<b>🚀 NeuroState AI Pipeline</b><br><br>'
        'MindLink raw EEG → in-memory EEG window → '
        'brainwave features → trained Neural Age model → '
        'trained Cognitive State model.<br><br>'
        'The live analysis does not read a CSV. '
        'The existing CSV recording in mindlink.py is kept '
        'unchanged but is not used as the live analysis source.'
        '</div>',
        unsafe_allow_html=True
    )


st.markdown("---")

st.markdown(
    '<div style="text-align:center;color:#475569;font-size:12px;">'
    'NeuroState AI • MindLink EEG • XGBoost Age Model • '
    'LSTM Cognitive State Model'
    '</div>',
    unsafe_allow_html=True
)