# ============================================================
# NEUROSTATE AI
# MINDLINK RAW CSV PIPELINE
#
# Raw MindLink CSV
#       ↓
# Validate columns
#       ↓
# Remove poor-signal rows
#       ↓
# Combine EEG bands
#       ↓
# Normalize bands
#       ↓
# Generate 9 model features
# ============================================================

import pandas as pd
import numpy as np


# ============================================================
# REQUIRED MINDLINK COLUMNS
# ============================================================

REQUIRED_COLUMNS = [
    "timestampMs",
    "poorSignal",
    "eegRawValue",
    "eegRawValueVolts",
    "attention",
    "meditation",
    "delta",
    "theta",
    "alphaLow",
    "alphaHigh",
    "betaLow",
    "betaHigh",
    "gammaLow",
    "gammaMid"
]


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
# LOAD CSV
# ============================================================

def load_mindlink_csv(file):

    df = pd.read_csv(file)

    # --------------------------------------------------------
    # Check required columns
    # --------------------------------------------------------

    missing = [
        col
        for col in REQUIRED_COLUMNS
        if col not in df.columns
    ]

    if missing:

        raise ValueError(
            "This CSV does not appear to be a supported "
            "MindLink EEG export.\n\n"
            "Missing columns:\n"
            + "\n".join(missing)
        )

    return df


# ============================================================
# PREPROCESS EEG BANDS
# ============================================================

def create_eeg_features(df):

    data = df.copy()

    # --------------------------------------------------------
    # Convert numeric columns
    # --------------------------------------------------------

    numeric_columns = [
        "timestampMs",
        "poorSignal",
        "eegRawValue",
        "eegRawValueVolts",
        "attention",
        "meditation",
        "delta",
        "theta",
        "alphaLow",
        "alphaHigh",
        "betaLow",
        "betaHigh",
        "gammaLow",
        "gammaMid"
    ]

    for column in numeric_columns:

        data[column] = pd.to_numeric(
            data[column],
            errors="coerce"
        )

    # --------------------------------------------------------
    # Remove rows where essential EEG values are missing
    # --------------------------------------------------------

    data = data.dropna(
        subset=[
            "delta",
            "theta",
            "alphaLow",
            "alphaHigh",
            "betaLow",
            "betaHigh",
            "gammaLow",
            "gammaMid"
        ]
    ).copy()

    # --------------------------------------------------------
    # Remove poor signal rows
    #
    # MindLink:
    # poorSignal = 0 → good signal
    # --------------------------------------------------------

    if "poorSignal" in data.columns:

        good_signal = (
            data["poorSignal"] == 0
        )

        # Only filter if there are actually good rows.
        if good_signal.any():

            data = data[
                good_signal
            ].copy()

    # --------------------------------------------------------
    # Combine MindLink sub-bands
    # --------------------------------------------------------

    data["alpha"] = (
        data["alphaLow"] +
        data["alphaHigh"]
    )

    data["beta"] = (
        data["betaLow"] +
        data["betaHigh"]
    )

    data["gamma"] = (
        data["gammaLow"] +
        data["gammaMid"]
    )

    # --------------------------------------------------------
    # Prevent negative/invalid values
    # --------------------------------------------------------

    band_columns = [
        "delta",
        "theta",
        "alpha",
        "beta",
        "gamma"
    ]

    for column in band_columns:

        data[column] = (
            data[column]
            .clip(lower=0)
        )

    # --------------------------------------------------------
    # Total EEG power
    # --------------------------------------------------------

    data["total_power"] = (
        data["delta"] +
        data["theta"] +
        data["alpha"] +
        data["beta"] +
        data["gamma"]
    )

    # Remove rows with zero total power

    data = data[
        data["total_power"] > 0
    ].copy()

    # --------------------------------------------------------
    # NORMALIZED BAND POWERS
    # --------------------------------------------------------

    data["delta_norm"] = (
        data["delta"] /
        data["total_power"]
    )

    data["theta_norm"] = (
        data["theta"] /
        data["total_power"]
    )

    data["alpha_norm"] = (
        data["alpha"] /
        data["total_power"]
    )

    data["beta_norm"] = (
        data["beta"] /
        data["total_power"]
    )

    data["gamma_norm"] = (
        data["gamma"] /
        data["total_power"]
    )

    # --------------------------------------------------------
    # EEG RATIOS
    # --------------------------------------------------------

    EPSILON = 1e-6

    data["alpha_beta_ratio"] = (
        data["alpha_norm"] /
        (data["beta_norm"] + EPSILON)
    )

    data["theta_beta_ratio"] = (
        data["theta_norm"] /
        (data["beta_norm"] + EPSILON)
    )

    data["delta_theta_ratio"] = (
        data["delta_norm"] /
        (data["theta_norm"] + EPSILON)
    )

    # --------------------------------------------------------
    # COGNITIVE INTENSITY
    # --------------------------------------------------------

    data["cognitive_intensity"] = (
        data["beta_norm"] +
        data["gamma_norm"]
    )

    # --------------------------------------------------------
    # Clean infinite values
    # --------------------------------------------------------

    data.replace(
        [np.inf, -np.inf],
        np.nan,
        inplace=True
    )

    data.dropna(
        subset=FEATURE_COLUMNS,
        inplace=True
    )

    # Reset index

    data.reset_index(
        drop=True,
        inplace=True
    )

    return data


# ============================================================
# MAIN PREPROCESSING FUNCTION
# ============================================================

def preprocess_csv(file):

    original_df = load_mindlink_csv(
        file
    )

    feature_df = create_eeg_features(
        original_df
    )

    if len(feature_df) == 0:

        raise ValueError(
            "No valid EEG samples remain after preprocessing."
        )

    return original_df, feature_df