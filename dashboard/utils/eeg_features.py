# ============================================================
# NEUROSTATE AI
# LIVE EEG FEATURE EXTRACTION
#
# RAW EEG
#     ↓
# Filtering
#     ↓
# Welch PSD
#     ↓
# Delta / Theta / Alpha / Beta / Gamma
#     ↓
# Normalized powers
#     ↓
# Ratios + Cognitive Intensity
# ============================================================

import numpy as np

from scipy.signal import (
    butter,
    sosfiltfilt,
    welch
)


# ============================================================
# CONFIGURATION
# ============================================================

# Temporary sampling rate.
# We will verify the real MindLink sampling rate later.
DEFAULT_FS = 512.0


# Minimum number of EEG samples required
MIN_SAMPLES = 256


# ============================================================
# EEG FREQUENCY BANDS
# ============================================================

BANDS = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 45.0),
}


# ============================================================
# BANDPASS FILTER
# ============================================================

def bandpass_filter(
    eeg,
    fs=DEFAULT_FS,
    lowcut=0.5,
    highcut=45.0,
    order=4
):

    eeg = np.asarray(
        eeg,
        dtype=np.float64
    )

    if len(eeg) < 20:
        return eeg

    nyquist = fs / 2.0

    highcut = min(
        highcut,
        nyquist - 1.0
    )

    if highcut <= lowcut:

        raise ValueError(
            f"Invalid filter range for "
            f"sampling rate {fs} Hz."
        )

    sos = butter(
        order,
        [lowcut, highcut],
        btype="bandpass",
        fs=fs,
        output="sos"
    )

    filtered = sosfiltfilt(
        sos,
        eeg
    )

    return filtered


# ============================================================
# POWER SPECTRAL DENSITY
# ============================================================

def calculate_psd(
    eeg,
    fs=DEFAULT_FS
):

    eeg = np.asarray(
        eeg,
        dtype=np.float64
    )

    if len(eeg) < MIN_SAMPLES:

        return None, None

    nperseg = min(
        len(eeg),
        int(fs * 2)
    )

    if nperseg < 64:

        return None, None

    frequencies, power = welch(
        eeg,
        fs=fs,
        nperseg=nperseg,
        noverlap=nperseg // 2,
        detrend="constant"
    )

    return frequencies, power


# ============================================================
# BAND POWER
# ============================================================

def calculate_band_power(
    frequencies,
    power,
    low,
    high
):

    mask = (
        (frequencies >= low)
        &
        (frequencies < high)
    )

    if not np.any(mask):

        return 0.0

    band_power = np.trapezoid(
        power[mask],
        frequencies[mask]
    )

    return float(
        max(band_power, 0.0)
    )


# ============================================================
# EXTRACT BRAINWAVE BANDS
# ============================================================

def extract_band_powers(
    eeg,
    fs=DEFAULT_FS
):

    eeg = np.asarray(
        eeg,
        dtype=np.float64
    )

    if len(eeg) < MIN_SAMPLES:

        return None

    # Remove DC component
    eeg = eeg - np.mean(eeg)

    # Filter
    filtered_eeg = bandpass_filter(
        eeg,
        fs=fs
    )

    # PSD
    frequencies, power = calculate_psd(
        filtered_eeg,
        fs=fs
    )

    if frequencies is None:

        return None

    # Calculate bands

    delta = calculate_band_power(
        frequencies,
        power,
        *BANDS["delta"]
    )

    theta = calculate_band_power(
        frequencies,
        power,
        *BANDS["theta"]
    )

    alpha = calculate_band_power(
        frequencies,
        power,
        *BANDS["alpha"]
    )

    beta = calculate_band_power(
        frequencies,
        power,
        *BANDS["beta"]
    )

    gamma = calculate_band_power(
        frequencies,
        power,
        *BANDS["gamma"]
    )

    return {
        "delta": delta,
        "theta": theta,
        "alpha": alpha,
        "beta": beta,
        "gamma": gamma
    }


# ============================================================
# NORMALIZE BAND POWERS
# ============================================================

def normalize_band_powers(
    band_powers
):

    total_power = sum(
        band_powers.values()
    )

    if total_power <= 0:

        return {
            "delta_norm": 0.0,
            "theta_norm": 0.0,
            "alpha_norm": 0.0,
            "beta_norm": 0.0,
            "gamma_norm": 0.0
        }

    return {
        "delta_norm":
            band_powers["delta"] / total_power,

        "theta_norm":
            band_powers["theta"] / total_power,

        "alpha_norm":
            band_powers["alpha"] / total_power,

        "beta_norm":
            band_powers["beta"] / total_power,

        "gamma_norm":
            band_powers["gamma"] / total_power
    }


# ============================================================
# SAFE RATIO
# ============================================================

def safe_ratio(
    numerator,
    denominator,
    epsilon=1e-6,
    max_ratio=100.0
):

    if abs(denominator) < epsilon:

        return 0.0

    ratio = numerator / denominator

    return float(
        np.clip(
            ratio,
            0.0,
            max_ratio
        )
    )


# ============================================================
# CALCULATE MODEL FEATURES
# ============================================================

def calculate_model_features(
    band_powers
):

    normalized = normalize_band_powers(
        band_powers
    )

    delta_norm = normalized[
        "delta_norm"
    ]

    theta_norm = normalized[
        "theta_norm"
    ]

    alpha_norm = normalized[
        "alpha_norm"
    ]

    beta_norm = normalized[
        "beta_norm"
    ]

    gamma_norm = normalized[
        "gamma_norm"
    ]

    # Ratios

    alpha_beta_ratio = safe_ratio(
        alpha_norm,
        beta_norm
    )

    theta_beta_ratio = safe_ratio(
        theta_norm,
        beta_norm
    )

    delta_theta_ratio = safe_ratio(
        delta_norm,
        theta_norm
    )

    # Cognitive intensity

    cognitive_intensity = (
        beta_norm
        +
        gamma_norm
    )

    return {

        "delta_norm":
            float(delta_norm),

        "theta_norm":
            float(theta_norm),

        "alpha_norm":
            float(alpha_norm),

        "beta_norm":
            float(beta_norm),

        "gamma_norm":
            float(gamma_norm),

        "alpha_beta_ratio":
            float(alpha_beta_ratio),

        "theta_beta_ratio":
            float(theta_beta_ratio),

        "delta_theta_ratio":
            float(delta_theta_ratio),

        "cognitive_intensity":
            float(cognitive_intensity)
    }


# ============================================================
# COMPLETE EEG → FEATURES PIPELINE
# ============================================================

def extract_features(
    eeg,
    fs=DEFAULT_FS
):

    eeg = np.asarray(
        eeg,
        dtype=np.float64
    )

    if len(eeg) < MIN_SAMPLES:

        return None

    # Get raw band powers

    band_powers = extract_band_powers(
        eeg,
        fs=fs
    )

    if band_powers is None:

        return None

    # Get model features

    features = calculate_model_features(
        band_powers
    )

    # Get normalized powers

    normalized = normalize_band_powers(
        band_powers
    )

    return {

        "bands":
            band_powers,

        "normalized":
            normalized,

        "features":
            features
    }


# ============================================================
# IMPORTANT
# ============================================================
#
# There is NO test code here.
#
# This file is imported by:
#
#     test_live_features.py
#
# Therefore we should NOT run a synthetic test automatically
# when this module is imported.
#
# ============================================================