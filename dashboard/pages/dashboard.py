import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import os
import tempfile

from utils.csv_pipeline import preprocess_csv
from utils.model_pipeline import run_models


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="NeuroState AI",
    page_icon="🧠",
    layout="wide"
)


# ============================================================
# SESSION STATE
# ============================================================

if "running" not in st.session_state:
    st.session_state.running = False

if "idx" not in st.session_state:
    st.session_state.idx = 0

if "mode" not in st.session_state:
    st.session_state.mode = "Studying"


# ============================================================
# CSS
# ============================================================

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
        padding-top: 1.5rem;
        padding-bottom: 3rem;
    }

    .title {
        font-size: 44px;
        font-weight: 900;
    }

    .sub {
        color: #94A3B8;
        font-size: 15px;
    }

    .section {
        font-size: 26px;
        font-weight: 900;
        margin-top: 28px;
        margin-bottom: 12px;
    }

    .label {
        color: #7DD3FC;
        font-size: 11px;
        font-weight: 900;
        letter-spacing: 1.4px;
    }

    .state {
        font-size: 42px;
        font-weight: 900;
    }

    .conf {
        color: #94A3B8;
        font-size: 13px;
    }

    .ai-title {
        color: #C4B5FD;
        font-size: 25px;
        font-weight: 900;
        margin-top: 8px;
    }

    .badge {
        display: inline-block;
        margin-top: 10px;
        padding: 7px 13px;
        border-radius: 20px;
        background: rgba(251, 191, 36, 0.08);
        border: 1px solid rgba(251, 191, 36, 0.30);
        color: #FCD34D;
        font-size: 11px;
        font-weight: 900;
    }

    .advice {
        font-size: 27px;
        font-weight: 900;
        margin-top: 15px;
    }

    .reason {
        color: #94A3B8;
        font-size: 14px;
        line-height: 1.5;
        margin-top: 8px;
    }

    .next {
        color: #67E8F9;
        font-size: 11px;
        font-weight: 900;
        letter-spacing: 1.2px;
        margin-top: 18px;
    }

    .action {
        font-size: 17px;
        font-weight: 700;
        margin-top: 5px;
    }

    .forecast-label {
        color: #94A3B8;
        font-size: 11px;
        font-weight: 900;
        letter-spacing: 1.2px;
    }

    .forecast-state {
        font-size: 29px;
        font-weight: 900;
        margin-top: 4px;
    }

    .wave-name {
        color: #94A3B8;
        font-size: 11px;
        font-weight: 900;
        letter-spacing: 1px;
    }

    .wave-value {
        font-size: 22px;
        font-weight: 900;
    }

    .age-number {
        font-size: 42px;
        font-weight: 900;
    }

    .age-label {
        color: #94A3B8;
        font-size: 13px;
        margin-top: 3px;
    }

    .device-note {
        color: #64748B;
        font-size: 12px;
        margin-top: 8px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# VOICE AI COGNITIVE COACH
# ============================================================

def speak_text(text):

    safe_text = (
        str(text)
        .replace("\\", "")
        .replace("'", "\\'")
        .replace("\n", " ")
    )

    st.components.v1.html(
        f"""
        <script>
        const message = new SpeechSynthesisUtterance(
            '{safe_text}'
        );

        message.rate = 0.95;
        message.pitch = 1.0;
        message.volume = 1.0;

        window.speechSynthesis.cancel();
        window.speechSynthesis.speak(message);
        </script>
        """,
        height=0
    )


# ============================================================
# HEADER
# ============================================================

header_left, header_right = st.columns([4, 1])


with header_left:

    st.markdown(
        '<div class="title">🧠 NeuroState AI</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="sub">'
        'EEG Cognitive Intelligence • '
        'State Analysis • Cognitive Forecasting'
        '</div>',
        unsafe_allow_html=True
    )


with header_right:

    if st.session_state.running:

        st.markdown(
            '<div style="text-align:right;'
            'color:#67E8F9;'
            'font-weight:900;'
            'padding-top:12px;">'
            '● LIVE SESSION'
            '</div>',
            unsafe_allow_html=True
        )

    else:

        st.markdown(
            '<div style="text-align:right;'
            'color:#64748B;'
            'font-weight:900;'
            'padding-top:12px;">'
            '● READY'
            '</div>',
            unsafe_allow_html=True
        )


st.markdown("---")


# ============================================================
# CONTROLS
# ============================================================

control1, control2, control3 = st.columns(
    [2, 2, 2]
)


with control1:

    st.session_state.mode = st.selectbox(
        "Current Activity",
        [
            "Studying",
            "Coding",
            "Reading",
            "Meeting",
            "Exam Preparation",
            "Creative Work",
            "Stress Monitoring",
            "Deep Sleep",
            "Simple Attention Test",
            "Deep Focus Test"
        ]
    )


with control2:

    source = st.selectbox(
        "Data Source",
        [
            "Live Simulation",
            "CSV Upload"
        ]
    )


with control3:

    uploaded = st.file_uploader(
        "Upload EEG CSV",
        type=["csv"]
    )


# ============================================================
# START / STOP
# ============================================================

button_left, button_right = st.columns(
    [1, 5]
)


with button_left:

    if st.session_state.running:

        if st.button(
            "⏹ Stop Session",
            width="stretch"
        ):

            st.session_state.running = False
            st.rerun()

    else:

        if st.button(
            "▶ Start Session",
            width="stretch"
        ):

            st.session_state.running = True
            st.session_state.idx = 0
            st.rerun()


# ============================================================
# DATA SOURCE
# ============================================================

# Real model output is populated only for CSV Upload.
model_results = None
processed_features = None

if (
    source == "CSV Upload"
    and uploaded is not None
):

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


        # RAW EEG -> FEATURES
        original_data, processed_features = preprocess_csv(
            temp_path
        )


        # FEATURES -> TRAINED MODELS
        model_results = run_models(
            processed_features
        )


        # Raw data for dashboard plots
        data = pd.read_csv(
            temp_path
        )

        data.columns = [
            str(column).strip().lower()
            for column in data.columns
        ]


        st.success(
            f"CSV loaded successfully • "
            f"{len(data):,} rows • "
            f"Real MindLink AI models active"
        )


    except Exception as error:

        st.error(
            f"EEG processing failed: {error}"
        )

        st.stop()


    finally:

        if temp_path is not None:

            try:
                os.remove(temp_path)
            except Exception:
                pass


else:

    # ========================================================
    # SIMULATED EEG
    # ========================================================

    rng = np.random.default_rng(42)

    sample_count = 3000

    time_axis = np.linspace(
        0,
        120,
        sample_count
    )


    eeg_signal = (
        0.30 * np.sin(
            2 * np.pi * 2 * time_axis
        )
        +
        0.18 * np.sin(
            2 * np.pi * 6 * time_axis
        )
        +
        0.14 * np.sin(
            2 * np.pi * 10 * time_axis
        )
        +
        0.08 * np.sin(
            2 * np.pi * 20 * time_axis
        )
        +
        rng.normal(
            0,
            0.035,
            sample_count
        )
    )


    attention_signal = np.clip(
        65
        + 18 * np.sin(
            time_axis / 7
        )
        + 8 * np.sin(
            time_axis / 2.5
        ),
        20,
        95
    )


    meditation_signal = np.clip(
        60
        + 12 * np.sin(
            time_axis / 9
        )
        + 5 * np.sin(
            time_axis / 3
        ),
        20,
        95
    )


    delta_signal = (
        0.20
        + 0.035 * np.sin(
            time_axis / 8
        )
    )

    theta_signal = (
        0.18
        + 0.040 * np.sin(
            time_axis / 6
        )
    )

    alpha_signal = (
        0.21
        + 0.035 * np.sin(
            time_axis / 9
        )
    )

    beta_signal = (
        0.24
        + 0.045 * np.sin(
            time_axis / 5
        )
    )

    gamma_signal = (
        0.17
        + 0.025 * np.sin(
            time_axis / 4
        )
    )


    data = pd.DataFrame(
        {
            "time": time_axis,
            "eeg": eeg_signal,
            "attention": attention_signal,
            "meditation": meditation_signal,
            "delta_norm": delta_signal,
            "theta_norm": theta_signal,
            "alpha_norm": alpha_signal,
            "beta_norm": beta_signal,
            "gamma_norm": gamma_signal
        }
    )


# ============================================================
# COLUMN FINDER
# ============================================================

def find_column(possible_names):

    for name in possible_names:

        if name in data.columns:

            return name

    return None


attention_column = find_column(
    [
        "attention",
        "attention_score"
    ]
)

meditation_column = find_column(
    [
        "meditation",
        "meditation_score"
    ]
)

eeg_column = find_column(
    
    [
        "eegrawvalue",
        "eegrawvaluevolts",
        "eeg_raw_value",
        "raw_eeg",
        "eeg_signal",
        "eegvalue",
        "eeg_value",
        "rawvalue",
        "raw_value",
        "eeg"
    ]
)

delta_column = find_column(
    [
        "delta_norm",
        "delta"
    ]
)

theta_column = find_column(
    [
        "theta_norm",
        "theta"
    ]
)

alpha_column = find_column(
    [
        "alpha_norm",
        "alpha"
    ]
)

beta_column = find_column(
    [
        "beta_norm",
        "beta"
    ]
)

gamma_column = find_column(
    [
        "gamma_norm",
        "gamma"
    ]
)


# ============================================================
# SAFE VALUE
# ============================================================

def get_value(
    column,
    default_value,
    row_index
):

    if column is None:

        return default_value

    try:

        value = float(
            data.iloc[row_index][column]
        )

        if np.isnan(value):

            return default_value

        return value

    except Exception:

        return default_value


# ============================================================
# BASE PLOT
# ============================================================

def create_base_figure(
    height
):

    figure = go.Figure()

    figure.update_layout(
        height=height,
        margin=dict(
            l=10,
            r=10,
            t=10,
            b=10
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(
            color="#94A3B8"
        )
    )

    return figure


# ============================================================
# AGE ESTIMATION
#
# Real CSV analysis uses the trained MindLink XGBoost model
# through run_models(). No simulated age is used.
# ============================================================


# ============================================================
# MAIN DASHBOARD
# ============================================================

def show_dashboard():

    # ========================================================
    # MOVE LIVE DATA
    # ========================================================

    if st.session_state.running:

        st.session_state.idx += 8

        if (
            st.session_state.idx
            >= len(data)
        ):

            st.session_state.idx = 0


    current_index = st.session_state.idx


    # ========================================================
    # CURRENT VALUES
    # ========================================================

    attention = get_value(
        attention_column,
        65,
        current_index
    )

    meditation = get_value(
        meditation_column,
        60,
        current_index
    )

    delta = get_value(
        delta_column,
        0.20,
        current_index
    )

    theta = get_value(
        theta_column,
        0.18,
        current_index
    )

    alpha = get_value(
        alpha_column,
        0.21,
        current_index
    )

    beta = get_value(
        beta_column,
        0.24,
        current_index
    )

    gamma = get_value(
        gamma_column,
        0.17,
        current_index
    )


    # ========================================================
    # COGNITIVE STATE
    # ========================================================

    if (
        model_results is not None
        and
        model_results.get("state") is not None
        and
        model_results["state"].get("available")
    ):

        current_state = (
            model_results["state"]["state"]
        )

        state_confidence = (
            model_results["state"]["confidence"]
            * 100
        )

    else:

        # Original simulation behavior
        if attention >= 75:

            current_state = "Focused"
            state_confidence = 91

        elif attention >= 50:

            current_state = "Moderate"
            state_confidence = 78

        else:

            current_state = "Fatigued"
            state_confidence = 83


    # ========================================================
    # LIVE EEG
    # ========================================================

    st.markdown(
        '<div class="section">'
        '📡 Live EEG Signal'
        '</div>',
        unsafe_allow_html=True
    )


    graph_start = max(
        0,
        current_index - 350
    )


    # ============================================================
# EEG VALUES FOR LIVE GRAPH
# ============================================================

if eeg_column:

    eeg_values = pd.to_numeric(
        data.iloc[
            graph_start:current_index + 1
        ][eeg_column],
        errors="coerce"
    ).fillna(0).to_numpy()

else:

    # --------------------------------------------------------
    # Some MindLink recordings do not have a column literally
    # called "eeg". Do NOT assume "eeg" exists.
    #
    # Fall back to the first suitable numeric EEG-like column.
    # --------------------------------------------------------

    excluded_columns = {
        "time",
        "timestamp",
        "attention",
        "meditation",
        "delta_norm",
        "theta_norm",
        "alpha_norm",
        "beta_norm",
        "gamma_norm",
        "alpha_beta_ratio",
        "theta_beta_ratio",
        "delta_theta_ratio",
        "cognitive_intensity"
    }

    numeric_candidates = []

    for column in data.columns:

        if column in excluded_columns:
            continue

        numeric_series = pd.to_numeric(
            data[column],
            errors="coerce"
        )

        valid_count = numeric_series.notna().sum()

        if valid_count > 0:

            numeric_candidates.append(
                (
                    column,
                    valid_count
                )
            )


    if numeric_candidates:

        # Prefer the column with the most usable numeric values.
        fallback_column = max(
            numeric_candidates,
            key=lambda item: item[1]
        )[0]

        eeg_values = pd.to_numeric(
            data.iloc[
                graph_start:current_index + 1
            ][fallback_column],
            errors="coerce"
        ).fillna(0).to_numpy()

    else:

        # Last-resort safe fallback.
        # This prevents the dashboard from crashing.
        eeg_values = np.zeros(
            current_index - graph_start + 1,
            dtype=float
        )
          


    if len(eeg_values) == 0:

        eeg_values = np.array(
            [0.0]
        )


    graph_x = np.arange(
        len(eeg_values)
    )


    eeg_figure = create_base_figure(
        350
    )


    eeg_figure.add_trace(
        go.Scatter(
            x=graph_x,
            y=eeg_values,
            mode="lines",
            line=dict(
                width=2
            )
        )
    )


    eeg_figure.update_layout(
        showlegend=False,

        xaxis=dict(
            title="Recent EEG samples",
            gridcolor="rgba(255,255,255,0.05)"
        ),

        yaxis=dict(
            title="EEG amplitude",
            gridcolor="rgba(255,255,255,0.05)"
        )
    )


    st.plotly_chart(
        eeg_figure,
        width="stretch",
        config={
            "displayModeBar": False
        }
    )


    # ========================================================
    # CURRENT COGNITIVE STATE
    # ========================================================

    st.markdown(
        '<div class="section">'
        '🧠 Current Cognitive State'
        '</div>',
        unsafe_allow_html=True
    )


    state_left, state_right = st.columns(
        [1, 2]
    )


    with state_left:

        st.markdown(
            '<div class="label">'
            'CURRENT COGNITIVE STATE'
            '</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            f'<div class="state">'
            f'{current_state}'
            f'</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            f'<div class="conf">'
            f'Model confidence • '
            f'{state_confidence}%'
            f'</div>',
            unsafe_allow_html=True
        )

        st.progress(
            state_confidence / 100
        )


    with state_right:

        metric1, metric2 = st.columns(
            2
        )

        with metric1:

            st.metric(
                "Attention",
                f"{attention:.0f}/100"
            )

        with metric2:

            st.metric(
                "Meditation",
                f"{meditation:.0f}/100"
            )


    # ========================================================
    # BRAINWAVE PROFILE
    # ========================================================

    st.markdown(
        '<div class="section">'
        '🌊 Brainwave Profile'
        '</div>',
        unsafe_allow_html=True
    )


    wave_columns = st.columns(
        5
    )


    waves = [
        ("DELTA", delta),
        ("THETA", theta),
        ("ALPHA", alpha),
        ("BETA", beta),
        ("GAMMA", gamma)
    ]


    for column, wave in zip(
        wave_columns,
        waves
    ):

        wave_name = wave[0]
        wave_value = wave[1]

        with column:

            st.markdown(
                f'<div class="wave-name">'
                f'{wave_name}'
                f'</div>',
                unsafe_allow_html=True
            )

            st.markdown(
                f'<div class="wave-value">'
                f'{wave_value:.3f}'
                f'</div>',
                unsafe_allow_html=True
            )

            st.progress(
                min(
                    max(
                        wave_value / 0.35,
                        0
                    ),
                    1
                )
            )


    # ========================================================
    # COGNITIVE FORECAST
    # ========================================================

    st.markdown(
        '<div class="section">'
        '🔮 Cognitive Forecast'
        '</div>',
        unsafe_allow_html=True
    )


    if current_state == "Focused":

        forecast_state = "Moderate"

        decline = np.array(
            [
                0,
                1,
                2,
                3,
                4,
                5,
                7
            ]
        )

    elif current_state == "Moderate":

        forecast_state = "Moderate"

        decline = np.array(
            [
                0,
                0,
                1,
                1,
                2,
                2,
                3
            ]
        )

    else:

        forecast_state = "Fatigued"

        decline = np.array(
            [
                0,
                2,
                3,
                5,
                6,
                8,
                10
            ]
        )


    future_seconds = np.array(
        [
            0,
            5,
            10,
            15,
            20,
            25,
            30
        ]
    )


    future_attention = np.clip(
        attention - decline,
        0,
        100
    )


    forecast_left, forecast_right = st.columns(
        [1, 2]
    )


    with forecast_left:

        st.markdown(
            '<div class="forecast-label">'
            'NOW'
            '</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            f'<div class="forecast-state">'
            f'{current_state}'
            f'</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div style="'
            'font-size:28px;'
            'color:#A78BFA;'
            '">'
            '↓'
            '</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="forecast-label">'
            'EXPECTED IN ~30 SECONDS'
            '</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            f'<div class="forecast-state">'
            f'{forecast_state}'
            f'</div>',
            unsafe_allow_html=True
        )


    with forecast_right:

        forecast_figure = create_base_figure(
            290
        )


        forecast_figure.add_trace(
            go.Scatter(
                x=future_seconds,
                y=future_attention,
                mode="lines+markers",
                line=dict(
                    width=4
                ),
                marker=dict(
                    size=7
                )
            )
        )


        forecast_figure.update_layout(
            showlegend=False,

            xaxis=dict(
                title="Future",

                tickvals=future_seconds,

                ticktext=[
                    "NOW",
                    "+5s",
                    "+10s",
                    "+15s",
                    "+20s",
                    "+25s",
                    "+30s"
                ],

                gridcolor="rgba(255,255,255,0.05)"
            ),

            yaxis=dict(
                title="Attention",

                range=[
                    0,
                    100
                ],

                gridcolor="rgba(255,255,255,0.05)"
            )
        )


        st.plotly_chart(
            forecast_figure,
            width="stretch",
            config={
                "displayModeBar": False
            }
        )


    # ========================================================
    # AGE + CROSS DEVICE
    # ========================================================

    st.markdown(
        '<div class="section">'
        '👤 Neural Profile & Device Analysis'
        '</div>',
        unsafe_allow_html=True
    )


    age_column, device_column = st.columns(
        [1, 2]
    )


    # ========================================================
    # AGE ESTIMATION
    # ========================================================

    with age_column:

        st.markdown(
            '<div class="ai-title">'
            '👤 Age Estimation'
            '</div>',
            unsafe_allow_html=True
        )


        st.markdown(
            '<div class="label">'
            'AI ESTIMATED NEURAL AGE GROUP'
            '</div>',
            unsafe_allow_html=True
        )


        # ----------------------------------------------------
        # REAL TRAINED MINDLINK AGE MODEL
        # ----------------------------------------------------

        if (
            model_results is not None
            and
            model_results.get("age") is not None
        ):

            age_result = model_results["age"]

            age_group = age_result["age_group"]

            confidence = (
                age_result["confidence"]
                * 100
            )


            st.markdown(
                f'<div class="age-number">'
                f'{age_group}'
                f'</div>',
                unsafe_allow_html=True
            )


            st.markdown(
                '<div class="age-label">'
                'MindLink XGBoost Neural Age Model'
                '</div>',
                unsafe_allow_html=True
            )


            st.progress(
                min(
                    max(
                        confidence / 100,
                        0
                    ),
                    1
                )
            )


            st.markdown(
                f'<div class="conf">'
                f'Model confidence • '
                f'{confidence:.2f}%'
                f'</div>',
                unsafe_allow_html=True
            )


            probabilities = (
                age_result.get("probabilities")
            )


            if probabilities is not None:

                classes = [
                    "18-30",
                    "31-50"
                ]


                for class_name, probability in zip(
                    classes,
                    probabilities
                ):

                    st.markdown(
                        f'<div style="'
                        f'display:flex;'
                        f'justify-content:space-between;'
                        f'margin-top:8px;'
                        f'font-size:13px;'
                        f'color:#94A3B8;">'
                        f'<span>{class_name}</span>'
                        f'<b>{probability * 100:.2f}%</b>'
                        f'</div>',
                        unsafe_allow_html=True
                    )


        else:

            st.markdown(
                '<div class="age-number">'
                'Waiting'
                '</div>',
                unsafe_allow_html=True
            )


            st.markdown(
                '<div class="age-label">'
                'Upload a MindLink EEG CSV '
                'to run the trained age model.'
                '</div>',
                unsafe_allow_html=True
            )


    # ========================================================
    # CROSS DEVICE CORRELATION
    # ========================================================

    with device_column:

        st.markdown(
            '<div class="ai-title">'
            '📡 Cross-Device Correlation'
            '</div>',
            unsafe_allow_html=True
        )


        device_data = pd.DataFrame(
            {
                "EEG Device": [
                    "Muse 2",
                    "Muse S",
                    "Emotiv EPOC X",
                    "OpenBCI Cyton",
                    "NeuroSky TGAM",
                    "MindLink"
                ],

                "Estimated Correlation (%)": [
                    99.28,
                    99.15,
                    98.83,
                    98.64,
                    97.55,
                    96.35
                ]
            }
        )


        st.dataframe(
            device_data,
            width="stretch",
            hide_index=True,

            column_config={
                "Estimated Correlation (%)":
                    st.column_config.ProgressColumn(
                        "Correlation %",
                        min_value=90,
                        max_value=100,
                        format="%.2f%%"
                    )
            }
        )


        st.markdown(
            '<div class="device-note">'
            'Reference compatibility estimates only. '
            'Actual cross-device correlation will be calculated '
            'from EEG data after Bluetooth integration.'
            '</div>',
            unsafe_allow_html=True
        )


    # ========================================================
    # AI INTELLIGENCE
    # ========================================================

    st.markdown(
        '<div class="section">'
        '🤖 AI Intelligence'
        '</div>',
        unsafe_allow_html=True
    )


    coach_column, scientist_column = st.columns(
        2
    )


    # ========================================================
    # AI COGNITIVE COACH
    # ========================================================

    with coach_column:

        st.markdown(
            '<div class="ai-title">'
            '⚡ AI Cognitive Coach'
            '</div>',
            unsafe_allow_html=True
        )


        if current_state == "Focused":

            badge = "STABLE FOCUS"

            advice = (
                "Excellent focus. Keep going."
            )

            reason = (
                "Attention is currently stable "
                "inside the focused range."
            )

            action = (
                "Continue your current task."
            )


        elif current_state == "Moderate":

            badge = "ATTENTION DRIFTING"

            advice = (
                "Take a 20-second reset."
            )

            reason = (
                "Attention is moving toward "
                "the moderate range."
            )

            action = (
                "Look away for 20 seconds, "
                "then resume."
            )


        else:

            badge = "FOCUS DROPOUT"

            advice = (
                "Pause briefly and reset."
            )

            reason = (
                "The current pattern shows "
                "reduced cognitive engagement."
            )

            action = (
                "Take a short break before continuing."
            )


        st.markdown(
            f'<div class="badge">'
            f'{badge}'
            f'</div>',
            unsafe_allow_html=True
        )


        st.markdown(
            f'<div class="advice">'
            f'{advice}'
            f'</div>',
            unsafe_allow_html=True
        )


        st.markdown(
            f'<div class="reason">'
            f'{reason}'
            f'</div>',
            unsafe_allow_html=True
        )


        st.markdown(
            '<div class="next">'
            '🎯 NEXT ACTION'
            '</div>',
            unsafe_allow_html=True
        )


        st.markdown(
            f'<div class="action">'
            f'{action}'
            f'</div>',
            unsafe_allow_html=True
        )


        # ----------------------------------------------------
        # VOICE AI COGNITIVE COACH
        # ----------------------------------------------------

        voice_message = (
            f"NeuroState AI cognitive coach. "
            f"Your current cognitive state is "
            f"{current_state}. "
            f"{advice} "
            f"Your recommended next action is: "
            f"{action}"
        )


        if st.button(
            "🔊 Speak AI Recommendation",
            key="voice_coach_button",
            width="stretch"
        ):

            speak_text(
                voice_message
            )


        # ----------------------------------------------------
        # FOCUS GAUGE
        # ----------------------------------------------------

        gauge = go.Figure(
            go.Indicator(
                mode="gauge+number",

                value=attention,

                number={
                    "suffix": "%",
                    "font": {
                        "size": 30
                    }
                },

                title={
                    "text": "FOCUS SCORE"
                },

                gauge={
                    "axis": {
                        "range": [
                            0,
                            100
                        ]
                    },

                    "bar": {
                        "thickness": 0.7
                    },

                    "steps": [
                        {
                            "range": [
                                0,
                                50
                            ],
                            "color":
                                "rgba(239,68,68,.10)"
                        },

                        {
                            "range": [
                                50,
                                75
                            ],
                            "color":
                                "rgba(251,191,36,.10)"
                        },

                        {
                            "range": [
                                75,
                                100
                            ],
                            "color":
                                "rgba(34,197,94,.10)"
                        }
                    ]
                }
            )
        )


        gauge.update_layout(
            height=220,

            margin=dict(
                l=20,
                r=20,
                t=30,
                b=0
            ),

            paper_bgcolor="rgba(0,0,0,0)",

            font=dict(
                color="#E5F6FF"
            )
        )


        st.plotly_chart(
            gauge,
            width="stretch",
            config={
                "displayModeBar": False
            }
        )


    # ========================================================
    # AI NEUROSCIENTIST
    # ========================================================

    with scientist_column:

        st.markdown(
            '<div class="ai-title">'
            '🔬 AI Neuroscientist Agent'
            '</div>',
            unsafe_allow_html=True
        )


        # ----------------------------------------------------
        # Dominant wave
        # ----------------------------------------------------

        wave_values = {
            "Delta": delta,
            "Theta": theta,
            "Alpha": alpha,
            "Beta": beta,
            "Gamma": gamma
        }


        dominant_wave = max(
            wave_values,
            key=wave_values.get
        )


        # ----------------------------------------------------
        # Interpretation
        # ----------------------------------------------------

        if beta > theta:

            interpretation = (
                "ACTIVE COGNITIVE ENGAGEMENT"
            )

            explanation = (
                "Beta activity is stronger than "
                "theta, suggesting active "
                "cognitive engagement."
            )


        elif theta > beta:

            interpretation = (
                "LOWER ALERTNESS PATTERN"
            )

            explanation = (
                "Theta activity is stronger than "
                "beta, suggesting a lower-alertness "
                "pattern."
            )


        else:

            interpretation = (
                "BALANCED COGNITIVE ACTIVITY"
            )

            explanation = (
                "Beta and theta activity are "
                "relatively balanced."
            )


        st.markdown(
            f'<div class="badge">'
            f'{interpretation}'
            f'</div>',
            unsafe_allow_html=True
        )


        # ----------------------------------------------------
        # Radar
        # ----------------------------------------------------

        radar_labels = [
            "Delta",
            "Theta",
            "Alpha",
            "Beta",
            "Gamma"
        ]


        radar_values = [
            delta,
            theta,
            alpha,
            beta,
            gamma
        ]


        radar = go.Figure()


        radar.add_trace(
            go.Scatterpolar(
                r=radar_values + [
                    radar_values[0]
                ],

                theta=radar_labels + [
                    radar_labels[0]
                ],

                fill="toself",

                line=dict(
                    width=3
                )
            )
        )


        radar.update_layout(
            height=320,

            margin=dict(
                l=30,
                r=30,
                t=20,
                b=20
            ),

            paper_bgcolor="rgba(0,0,0,0)",

            font=dict(
                color="#CBD5E1"
            ),

            showlegend=False,

            polar=dict(

                bgcolor="rgba(0,0,0,0)",

                radialaxis=dict(
                    visible=True,

                    range=[
                        0,
                        0.35
                    ],

                    gridcolor=
                        "rgba(255,255,255,.08)"
                ),

                angularaxis=dict(
                    gridcolor=
                        "rgba(255,255,255,.08)"
                )
            )
        )


        st.plotly_chart(
            radar,
            width="stretch",
            config={
                "displayModeBar": False
            }
        )


        st.markdown(
            f'<div class="advice">'
            f'{dominant_wave} activity is currently dominant'
            f'</div>',
            unsafe_allow_html=True
        )


        st.markdown(
            f'<div class="reason">'
            f'{explanation}'
            f'</div>',
            unsafe_allow_html=True
        )


        # ----------------------------------------------------
        # Wave metrics
        # ----------------------------------------------------

        wave_metric1, wave_metric2, wave_metric3 = st.columns(
            3
        )


        with wave_metric1:

            st.metric(
                "Delta",
                f"{delta:.3f}"
            )


        with wave_metric2:

            st.metric(
                "Theta",
                f"{theta:.3f}"
            )


        with wave_metric3:

            st.metric(
                "Alpha",
                f"{alpha:.3f}"
            )


        wave_metric4, wave_metric5 = st.columns(
            2
        )


        with wave_metric4:

            st.metric(
                "Beta",
                f"{beta:.3f}"
            )


        with wave_metric5:

            st.metric(
                "Gamma",
                f"{gamma:.3f}"
            )


# ============================================================
# LIVE REFRESH
# ============================================================

if st.session_state.running:

    @st.fragment(run_every="1s")
    def live_dashboard():

        show_dashboard()


    live_dashboard()

else:

    show_dashboard()


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.markdown(
    """
    <div style="
        text-align:center;
        color:#475569;
        font-size:12px;
    ">
        NeuroState AI • EEG Cognitive Intelligence Platform
    </div>
    """,
    unsafe_allow_html=True
)