import streamlit as st
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="NeuroState AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ============================================================
# DASHBOARD PAGE
# ============================================================

dashboard_page = st.Page(
    "dashboard1.py",
    title="NeuroState AI Dashboard",
    icon="🧠"
)


# ============================================================
# LANDING PAGE
# ============================================================

def landing_page():

    # ========================================================
    # CUSTOM CSS
    # ========================================================

    st.markdown(
        """
<style>

.stApp {
    background:
        radial-gradient(
            circle at 10% 0%,
            rgba(34, 211, 238, 0.07),
            transparent 25%
        ),
        radial-gradient(
            circle at 90% 0%,
            rgba(124, 58, 237, 0.09),
            transparent 30%
        ),
        linear-gradient(
            180deg,
            #05070A 0%,
            #080D18 100%
        );

    color: #E5F6FF;
}


.block-container {
    max-width: 1500px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}


/* ==========================================================
   HERO
   ========================================================== */

.hero-title {
    font-size: 58px;
    font-weight: 900;
    color: #E5F6FF;
    line-height: 1.05;
    margin-bottom: 8px;
}


.hero-sub {
    color: #67E8F9;
    font-size: 20px;
    font-weight: 700;
    margin-bottom: 18px;
}


.hero-description {
    color: #94A3B8;
    font-size: 16px;
    line-height: 1.7;
}


/* ==========================================================
   SECTION TITLES
   ========================================================== */

.section-title {
    font-size: 30px;
    font-weight: 900;
    color: #E5F6FF;
    margin-top: 30px;
    margin-bottom: 18px;
}


/* ==========================================================
   CARDS
   ========================================================== */

.card {
    background: rgba(255, 255, 255, 0.035);
    border: 1px solid rgba(148, 163, 184, 0.18);
    border-radius: 16px;
    padding: 22px;
    margin-top: 10px;
    min-height: 145px;
}


.card h3 {
    color: #E5F6FF !important;
    font-size: 19px;
    font-weight: 800;
    margin-bottom: 10px;
}


.card p {
    color: #94A3B8 !important;
    font-size: 14px;
    line-height: 1.6;
}


/* ==========================================================
   FEATURE CARDS
   ========================================================== */

.feature-card {
    background:
        linear-gradient(
            145deg,
            rgba(34, 211, 238, 0.07),
            rgba(124, 58, 237, 0.06)
        );

    border: 1px solid rgba(103, 232, 249, 0.15);
    border-radius: 18px;
    padding: 24px;
    min-height: 175px;
}


.feature-icon {
    font-size: 30px;
    margin-bottom: 10px;
}


.feature-title {
    color: #E5F6FF;
    font-size: 19px;
    font-weight: 900;
}


.feature-text {
    color: #94A3B8;
    font-size: 14px;
    line-height: 1.6;
    margin-top: 8px;
}


/* ==========================================================
   CTA
   ========================================================== */

.cta-card {
    background:
        radial-gradient(
            circle at 20% 0%,
            rgba(34, 211, 238, 0.12),
            transparent 35%
        ),
        radial-gradient(
            circle at 90% 100%,
            rgba(124, 58, 237, 0.12),
            transparent 40%
        );

    border: 1px solid rgba(103, 232, 249, 0.20);
    border-radius: 20px;
    padding: 32px;
    text-align: center;
}


.cta-title {
    color: #E5F6FF;
    font-size: 30px;
    font-weight: 900;
}


.cta-text {
    color: #94A3B8;
    font-size: 15px;
    line-height: 1.6;
    margin-top: 10px;
}


/* ==========================================================
   BUTTON
   ========================================================== */

.stButton > button {
    background:
        linear-gradient(
            135deg,
            #22D3EE,
            #7C3AED
        );

    color: white !important;
    border: none;
    border-radius: 12px;
    font-weight: 900;
    font-size: 16px;
    min-height: 48px;
}


.stButton > button:hover {
    border: 1px solid #67E8F9;
    color: white !important;
}


/* ==========================================================
   PAGE LINK
   ========================================================== */

a[data-testid="stPageLink-NavLink"] {
    background:
        linear-gradient(
            135deg,
            #22D3EE,
            #7C3AED
        );

    color: white !important;
    border-radius: 12px;
    padding: 14px 22px;
    font-weight: 900;
    text-decoration: none;
    display: block;
    text-align: center;
}


a[data-testid="stPageLink-NavLink"]:hover {
    opacity: 0.9;
}


/* ==========================================================
   GENERAL
   ========================================================== */

hr {
    border: none;
    border-top: 1px solid rgba(148, 163, 184, 0.18);
    margin: 35px 0;
}


h1, h2, h3, p, label {
    color: #E5F6FF;
}


/* ==========================================================
   FOOTER
   ========================================================== */

.footer {
    text-align: center;
    color: #475569;
    font-size: 12px;
    margin-top: 35px;
}

</style>
        """,
        unsafe_allow_html=True
    )


    # ========================================================
    # HERO SECTION
    # ========================================================

    left, right = st.columns(
        [1.55, 1]
    )


    with left:

        st.markdown(
            '<div class="hero-title">🧠 NeuroState AI</div>',
            unsafe_allow_html=True
        )


        st.markdown(
            '<div class="hero-sub">'
            'Measure. Understand. Forecast.'
            '</div>',
            unsafe_allow_html=True
        )


        st.markdown(
            '<div class="hero-description">'
            'A real-time EEG intelligence platform that analyzes '
            'brainwave activity, detects cognitive states, '
            'estimates neural age patterns, forecasts attention '
            'changes, and provides personalized AI-driven insights.'
            '</div>',
            unsafe_allow_html=True
        )


        st.write("")


        # ----------------------------------------------------
        # LAUNCH BUTTON
        # ----------------------------------------------------

        st.page_link(
            dashboard_page,
            label="🧠  Launch NeuroState AI",
            use_container_width=True
        )


    with right:

        st.image(
    str(BASE_DIR / "assets" / "brain.png"),
    use_container_width=True
    )


    st.markdown("---")


    # ========================================================
    # REAL-TIME EEG INTELLIGENCE
    # ========================================================

    st.markdown(
        '<div class="section-title">'
        '📡 Real-Time EEG Intelligence'
        '</div>',
        unsafe_allow_html=True
    )


    f1, f2, f3 = st.columns(3)


    with f1:

        st.markdown(
            """
<div class="feature-card">

<div class="feature-icon">📡</div>

<div class="feature-title">
Live EEG Monitoring
</div>

<div class="feature-text">
Connect the MindLink EEG headset and observe raw EEG
signals, attention and meditation values in real time.
</div>

</div>
            """,
            unsafe_allow_html=True
        )


    with f2:

        st.markdown(
            """
<div class="feature-card">

<div class="feature-icon">🌊</div>

<div class="feature-title">
Brainwave Analysis
</div>

<div class="feature-text">
Analyze Delta, Theta, Alpha, Beta and Gamma activity
extracted from EEG recordings.
</div>

</div>
            """,
            unsafe_allow_html=True
        )


    with f3:

        st.markdown(
            """
<div class="feature-card">

<div class="feature-icon">🧠</div>

<div class="feature-title">
Cognitive Intelligence
</div>

<div class="feature-text">
Convert EEG features into neural age estimates,
cognitive-state predictions and attention insights.
</div>

</div>
            """,
            unsafe_allow_html=True
        )


    # ========================================================
    # BRAINWAVES
    # ========================================================

    st.markdown(
        '<div class="section-title">'
        '🌊 Understanding Brainwaves'
        '</div>',
        unsafe_allow_html=True
    )


    b1, b2 = st.columns(2)


    with b1:

        st.markdown(
            """
<div class="card">

<h3>Delta • 0.5–4 Hz</h3>

<p>
Very slow brain activity commonly associated with
deep sleep and restorative processes.
</p>

</div>
            """,
            unsafe_allow_html=True
        )


        st.markdown(
            """
<div class="card">

<h3>Theta • 4–8 Hz</h3>

<p>
Slower activity associated with memory,
learning and internal cognitive processing.
</p>

</div>
            """,
            unsafe_allow_html=True
        )


        st.markdown(
            """
<div class="card">

<h3>Alpha • 8–13 Hz</h3>

<p>
Commonly observed during relaxed wakefulness
and calm mental states.
</p>

</div>
            """,
            unsafe_allow_html=True
        )


    with b2:

        st.markdown(
            """
<div class="card">

<h3>Beta • 13–30 Hz</h3>

<p>
Faster activity associated with alertness,
active attention and cognitive processing.
</p>

</div>
            """,
            unsafe_allow_html=True
        )


        st.markdown(
            """
<div class="card">

<h3>Gamma • 30+ Hz</h3>

<p>
Very fast activity associated with complex
information processing and higher-level cognition.
</p>

</div>
            """,
            unsafe_allow_html=True
        )


        st.markdown(
            """
<div class="card">

<h3>📊 Real-Time Insight</h3>

<p>
NeuroState AI combines EEG bands with trained
AI models to understand the current cognitive pattern.
</p>

</div>
            """,
            unsafe_allow_html=True
        )


    st.markdown("---")


    # ========================================================
    # AI MODELS
    # ========================================================

    st.markdown(
        '<div class="section-title">'
        '🤖 NeuroState AI Intelligence'
        '</div>',
        unsafe_allow_html=True
    )


    m1, m2, m3 = st.columns(3)


    with m1:

        st.markdown(
            """
<div class="feature-card">

<div class="feature-icon">🧠</div>

<div class="feature-title">
Cognitive State Detection
</div>

<div class="feature-text">
Classifies EEG activity into Focused, Moderate
or Fatigued cognitive states.
</div>

</div>
            """,
            unsafe_allow_html=True
        )


    with m2:

        st.markdown(
            """
<div class="feature-card">

<div class="feature-icon">👤</div>

<div class="feature-title">
Neural Age Estimation
</div>

<div class="feature-text">
Uses normalized EEG spectral features and the trained
MindLink model to estimate neural age groups.
</div>

</div>
            """,
            unsafe_allow_html=True
        )


    with m3:

        st.markdown(
            """
<div class="feature-card">

<div class="feature-icon">🔮</div>

<div class="feature-title">
Cognitive Forecast
</div>

<div class="feature-text">
Forecasts short-term attention and meditation trends
from EEG activity.
</div>

</div>
            """,
            unsafe_allow_html=True
        )


    # ========================================================
    # ADVANCED INTELLIGENCE
    # ========================================================

    st.markdown(
        '<div class="section-title">'
        '🔬 Advanced EEG Intelligence'
        '</div>',
        unsafe_allow_html=True
    )


    a1, a2, a3 = st.columns(3)


    with a1:

        st.markdown(
            """
<div class="feature-card">

<div class="feature-icon">📡</div>

<div class="feature-title">
Cross-Device Correlation
</div>

<div class="feature-text">
Compare normalized EEG characteristics across
different EEG acquisition devices.
</div>

</div>
            """,
            unsafe_allow_html=True
        )


    with a2:

        st.markdown(
            """
<div class="feature-card">

<div class="feature-icon">⚡</div>

<div class="feature-title">
AI Cognitive Coach
</div>

<div class="feature-text">
Converts detected cognitive states into practical
focus and recovery recommendations.
</div>

</div>
            """,
            unsafe_allow_html=True
        )


    with a3:

        st.markdown(
            """
<div class="feature-card">

<div class="feature-icon">🔬</div>

<div class="feature-title">
AI Neuroscientist
</div>

<div class="feature-text">
Provides an EEG-oriented interpretation of the
observed brainwave pattern.
</div>

</div>
            """,
            unsafe_allow_html=True
        )


    st.markdown("---")


    # ========================================================
    # CTA
    # ========================================================

    st.markdown(
        """
<div class="cta-card">

<div class="cta-title">
🧠 Experience NeuroState AI
</div>

<div class="cta-text">
Connect your MindLink EEG headset or analyze an EEG
recording using the unified NeuroState AI dashboard.
</div>

</div>
        """,
        unsafe_allow_html=True
    )


    st.write("")


    st.page_link(
        dashboard_page,
        label="🚀  Start Analysis",
        use_container_width=True
    )


    # ========================================================
    # FOOTER
    # ========================================================

    st.markdown("---")

    st.markdown(
        """
<div class="footer">
NeuroState AI v1.0 • EEG Cognitive Intelligence Platform
</div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# NAVIGATION
# ============================================================

pg = st.navigation(
    [
        st.Page(
            landing_page,
            title="NeuroState AI",
            icon="🧠"
        ),
        dashboard_page
    ],
    position="hidden"
)


# ============================================================
# RUN
# ============================================================

pg.run()