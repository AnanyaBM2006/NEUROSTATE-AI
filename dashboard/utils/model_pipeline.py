# ============================================================
# NEUROSTATE AI
# MODEL PIPELINE
#
# RAW EEG FEATURES
#       ↓
# MINDLINK AGE MODEL
#       ↓
# COGNITIVE STATE MODEL
# ============================================================

import os
import joblib
import numpy as np

from tensorflow.keras.models import load_model


# ============================================================
# PROJECT PATH
# ============================================================

BASE_DIR = r"C:\Users\bmana\Documents\NeuroStateAI"


# ============================================================
# MODEL DIRECTORIES
# ============================================================

AGE_DIR = os.path.join(
    BASE_DIR,
    "models",
    "age_estimation"
)

STATE_DIR = os.path.join(
    BASE_DIR,
    "models",
    "cognitive_state"
)


# ============================================================
# MODEL FEATURES
# ============================================================

FEATURE_COLUMNS = [
    "delta_norm",
    "theta_norm",
    "alpha_norm",
    "beta_norm",
    "gamma_norm",
    "alpha_beta_ratio",
    "theta_beta_ratio",
    "delta_theta_ratio",
    "cognitive_intensity"
]


# ============================================================
# LOAD AGE MODEL
# ============================================================

age_model = joblib.load(
    os.path.join(
        AGE_DIR,
        "age_model.pkl"
    )
)

age_encoder = joblib.load(
    os.path.join(
        AGE_DIR,
        "age_label_encoder.pkl"
    )
)

age_features = joblib.load(
    os.path.join(
        AGE_DIR,
        "age_features.pkl"
    )
)


# ============================================================
# LOAD COGNITIVE STATE MODEL
# ============================================================

state_model = load_model(
    os.path.join(
        STATE_DIR,
        "state_model.keras"
    )
)

state_info = joblib.load(
    os.path.join(
        STATE_DIR,
        "state_info.pkl"
    )
)


# ============================================================
# STATE LABELS
#
# 0 → Fatigued
# 1 → Focused
# 2 → Moderate
# ============================================================

STATE_LABELS = {
    0: "Fatigued",
    1: "Focused",
    2: "Moderate"
}


# ============================================================
# AGE PREDICTION
# ============================================================

def predict_age(feature_df):

    # --------------------------------------------------------
    # CHECK FEATURES
    # --------------------------------------------------------

    missing = [
        feature
        for feature in age_features
        if feature not in feature_df.columns
    ]

    if missing:

        raise ValueError(
            "Missing age-model features: "
            + ", ".join(missing)
        )

    # --------------------------------------------------------
    # REMOVE INVALID VALUES
    # --------------------------------------------------------

    age_data = (
        feature_df[
            age_features
        ]
        .replace(
            [np.inf, -np.inf],
            np.nan
        )
        .dropna()
    )

    if len(age_data) == 0:

        raise ValueError(
            "No valid EEG feature rows available "
            "for age prediction."
        )

    # --------------------------------------------------------
    # PREDICT MULTIPLE EEG ROWS
    # --------------------------------------------------------

    predictions = age_model.predict(
        age_data
    )

    probabilities = age_model.predict_proba(
        age_data
    )

    # --------------------------------------------------------
    # AVERAGE PROBABILITY ACROSS THE SESSION
    # --------------------------------------------------------

    mean_probabilities = (
        probabilities.mean(
            axis=0
        )
    )

    predicted_index = int(
        np.argmax(
            mean_probabilities
        )
    )

    age_group = (
        age_encoder
        .inverse_transform(
            [predicted_index]
        )[0]
    )

    confidence = float(
        mean_probabilities[
            predicted_index
        ]
    )

    # --------------------------------------------------------
    # ROW-LEVEL PREDICTION DISTRIBUTION
    # --------------------------------------------------------

    prediction_counts = {}

    for index, label in enumerate(
        age_encoder.classes_
    ):

        prediction_counts[label] = int(
            np.sum(
                predictions == index
            )
        )

    return {
        "age_group": age_group,
        "confidence": confidence,
        "probabilities": mean_probabilities,
        "prediction_counts": prediction_counts,
        "samples_used": len(age_data)
    }


# ============================================================
# COGNITIVE STATE PREDICTION
# ============================================================

def predict_cognitive_state(feature_df):

    missing = [
        feature
        for feature in FEATURE_COLUMNS
        if feature not in feature_df.columns
    ]

    if missing:

        raise ValueError(
            "Missing cognitive-state features: "
            + ", ".join(missing)
        )

    # --------------------------------------------------------
    # MODEL REQUIRES 20 SAMPLES
    # --------------------------------------------------------

    sequence_length = int(
        state_info.get(
            "sequence_length",
            20
        )
    )

    if len(feature_df) < sequence_length:

        return {
            "available": False,
            "reason": (
                f"Cognitive-state analysis requires "
                f"{sequence_length} valid EEG samples. "
                f"Only {len(feature_df)} were found."
            ),
            "state": None,
            "class": None,
            "confidence": None,
            "probabilities": None
        }

    # --------------------------------------------------------
    # TAKE MOST RECENT 20 SAMPLES
    # --------------------------------------------------------

    sequence = (
        feature_df[
            FEATURE_COLUMNS
        ]
        .tail(
            sequence_length
        )
        .to_numpy(
            dtype=np.float32
        )
    )

    sequence = np.expand_dims(
        sequence,
        axis=0
    )

    # --------------------------------------------------------
    # MODEL PREDICTION
    # --------------------------------------------------------

    probabilities = state_model.predict(
        sequence,
        verbose=0
    )[0]

    predicted_class = int(
        np.argmax(
            probabilities
        )
    )

    state_name = STATE_LABELS.get(
        predicted_class,
        "Unknown"
    )

    confidence = float(
        probabilities[
            predicted_class
        ]
    )

    return {
        "available": True,
        "reason": None,
        "state": state_name,
        "class": predicted_class,
        "confidence": confidence,
        "probabilities": probabilities,
        "sequence": sequence
    }


# ============================================================
# RUN ALL AVAILABLE MODELS
# ============================================================

def run_models(feature_df):

    age_result = predict_age(
        feature_df
    )

    state_result = predict_cognitive_state(
        feature_df
    )

    return {
        "age": age_result,
        "state": state_result
    }