# ============================================================
# NEUROSTATE AI
# BUILD MINDLINK AGE DATASET
#
# MindLink CSV
#      ↓
# Signal-quality filtering
#      ↓
# EEG band features
#      ↓
# 9 model features
#      ↓
# Age label
#      ↓
# mindlink_age_dataset.csv
# ============================================================

import os
import pandas as pd
import numpy as np


# ============================================================
# PATHS
# ============================================================

BASE_DIR = r"C:\Users\bmana\Documents\NeuroStateAI"

MINDLINK_DIR = os.path.join(
    BASE_DIR,
    "data",
    "mindlink"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "data",
    "mindlink_age"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# AGE 20 FILES
# ============================================================

AGE_20_FILES = [
    "A class.csv",
    "A meditation.csv",
    "k class.csv",
    "K meditation.csv",
    "S reels.csv",
    "Spoorti class.csv"
]


# ============================================================
# AGE 46 FILES
# ============================================================

AGE_46_FILES = [
    "eegIDRecord_2.csv",
    "eegIDRecord_1.csv",
    "geetha mam.csv"
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
# PROCESS ONE MINDLINK FILE
# ============================================================

def process_file(
    file_path,
    actual_age,
    participant_id
):

    print()
    print("----------------------------------------")
    print("Processing:", os.path.basename(file_path))
    print("Age:", actual_age)

    # --------------------------------------------------------
    # LOAD CSV
    # --------------------------------------------------------

    df = pd.read_csv(file_path)

    print(
        "Original rows:",
        len(df)
    )

    # --------------------------------------------------------
    # REQUIRED COLUMNS
    # --------------------------------------------------------

    required = [
        "poorSignal",
        "delta",
        "theta",
        "alphaLow",
        "alphaHigh",
        "betaLow",
        "betaHigh",
        "gammaLow",
        "gammaMid"
    ]

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:

        raise ValueError(
            f"{os.path.basename(file_path)} "
            f"is missing required columns: {missing}"
        )

    # --------------------------------------------------------
    # CONVERT EEG COLUMNS TO NUMERIC
    # --------------------------------------------------------

    numeric_columns = [
        "poorSignal",
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

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    # --------------------------------------------------------
    # REMOVE MISSING VALUES
    # --------------------------------------------------------

    df = df.dropna(
        subset=numeric_columns
    ).copy()

    print(
        "After missing-value removal:",
        len(df)
    )

    # --------------------------------------------------------
    # KEEP GOOD SIGNAL ONLY
    #
    # MindLink:
    # poorSignal = 0 means good signal
    # --------------------------------------------------------

    df = df[
        df["poorSignal"] == 0
    ].copy()

    print(
        "Good-signal rows:",
        len(df)
    )

    if len(df) == 0:

        print(
            "WARNING: No good-signal rows."
        )

        return None

    # --------------------------------------------------------
    # COMBINE SUB-BANDS
    # --------------------------------------------------------

    df["alpha"] = (
        df["alphaLow"] +
        df["alphaHigh"]
    )

    df["beta"] = (
        df["betaLow"] +
        df["betaHigh"]
    )

    df["gamma"] = (
        df["gammaLow"] +
        df["gammaMid"]
    )

    # --------------------------------------------------------
    # PREVENT NEGATIVE POWER VALUES
    # --------------------------------------------------------

    band_columns = [
        "delta",
        "theta",
        "alpha",
        "beta",
        "gamma"
    ]

    for column in band_columns:

        df[column] = (
            df[column]
            .clip(lower=0)
        )

    # --------------------------------------------------------
    # TOTAL POWER
    # --------------------------------------------------------

    df["total_power"] = (
        df["delta"] +
        df["theta"] +
        df["alpha"] +
        df["beta"] +
        df["gamma"]
    )

    # Remove rows with zero total power

    df = df[
        df["total_power"] > 0
    ].copy()

    # --------------------------------------------------------
    # NORMALIZED EEG BAND POWER
    # --------------------------------------------------------

    df["delta_norm"] = (
        df["delta"] /
        df["total_power"]
    )

    df["theta_norm"] = (
        df["theta"] /
        df["total_power"]
    )

    df["alpha_norm"] = (
        df["alpha"] /
        df["total_power"]
    )

    df["beta_norm"] = (
        df["beta"] /
        df["total_power"]
    )

    df["gamma_norm"] = (
        df["gamma"] /
        df["total_power"]
    )

    # --------------------------------------------------------
    # RATIOS
    # --------------------------------------------------------

    EPSILON = 1e-6

    df["alpha_beta_ratio"] = (
        df["alpha_norm"] /
        (df["beta_norm"] + EPSILON)
    )

    df["theta_beta_ratio"] = (
        df["theta_norm"] /
        (df["beta_norm"] + EPSILON)
    )

    df["delta_theta_ratio"] = (
        df["delta_norm"] /
        (df["theta_norm"] + EPSILON)
    )

    # --------------------------------------------------------
    # COGNITIVE INTENSITY
    # --------------------------------------------------------

    df["cognitive_intensity"] = (
        df["beta_norm"] +
        df["gamma_norm"]
    )

    # --------------------------------------------------------
    # REMOVE INVALID VALUES
    # --------------------------------------------------------

    df.replace(
        [np.inf, -np.inf],
        np.nan,
        inplace=True
    )

    df.dropna(
        subset=FEATURE_COLUMNS,
        inplace=True
    )

    # ========================================================
    # IMPORTANT: AGE LABEL
    # ========================================================

    # Store the actual age for every feature row

    df["age"] = actual_age

    # --------------------------------------------------------
    # AGE GROUP
    #
    # 20-year-old → 18-30
    # 46-year-old → 31-50
    # --------------------------------------------------------

    if actual_age <= 30:

        df["age_group"] = "18-30"

    elif actual_age <= 50:

        df["age_group"] = "31-50"

    else:

        df["age_group"] = "51+"

    # --------------------------------------------------------
    # PARTICIPANT ID
    # --------------------------------------------------------

    df["participant_id"] = participant_id

    # --------------------------------------------------------
    # SOURCE FILE
    # --------------------------------------------------------

    df["source_file"] = (
        os.path.basename(file_path)
    )

    # --------------------------------------------------------
    # FINAL COLUMNS
    # --------------------------------------------------------

    output_columns = (
        FEATURE_COLUMNS
        + [
            "age",
            "age_group",
            "participant_id",
            "source_file"
        ]
    )

    return df[
        output_columns
    ].copy()


# ============================================================
# MAIN
# ============================================================

all_data = []


# ============================================================
# PROCESS AGE 20 DATA
# ============================================================

print()
print("========================================")
print("PROCESSING AGE 20 DATA")
print("========================================")

for index, filename in enumerate(
    AGE_20_FILES,
    start=1
):

    path = os.path.join(
        MINDLINK_DIR,
        filename
    )

    if not os.path.exists(path):

        print(
            "WARNING: File not found:",
            filename
        )

        continue

    result = process_file(
        path,
        actual_age=20,
        participant_id=f"AGE20_P{index}"
    )

    if result is not None:

        all_data.append(
            result
        )


# ============================================================
# PROCESS AGE 46 DATA
# ============================================================

print()
print("========================================")
print("PROCESSING AGE 46 DATA")
print("========================================")

for index, filename in enumerate(
    AGE_46_FILES,
    start=1
):

    path = os.path.join(
        MINDLINK_DIR,
        filename
    )

    if not os.path.exists(path):

        print(
            "WARNING: File not found:",
            filename
        )

        continue

    result = process_file(
        path,
        actual_age=46,
        participant_id=f"AGE46_P{index}"
    )

    if result is not None:

        all_data.append(
            result
        )


# ============================================================
# CHECK DATA
# ============================================================

if len(all_data) == 0:

    raise RuntimeError(
        "No valid MindLink data was found."
    )


# ============================================================
# COMBINE ALL RECORDINGS
# ============================================================

mindlink_age_dataset = pd.concat(
    all_data,
    ignore_index=True
)


# ============================================================
# SAVE DATASET
# ============================================================

output_path = os.path.join(
    OUTPUT_DIR,
    "mindlink_age_dataset.csv"
)

mindlink_age_dataset.to_csv(
    output_path,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print()
print("========================================")
print("MINDLINK AGE DATASET CREATED")
print("========================================")

print()
print(
    "Total feature rows:",
    len(mindlink_age_dataset)
)

print()
print("Age distribution:")

print(
    mindlink_age_dataset[
        "age"
    ]
    .value_counts()
    .sort_index()
)

print()
print("Age-group distribution:")

print(
    mindlink_age_dataset[
        "age_group"
    ]
    .value_counts()
)

print()
print("Participant distribution:")

print(
    mindlink_age_dataset[
        "participant_id"
    ]
    .value_counts()
)

print()
print("Feature columns:")

for column in FEATURE_COLUMNS:

    print(
        " -",
        column
    )

print()
print(
    "Saved to:",
    output_path
)

print()
print("========================================")
print("DATASET BUILD COMPLETE")
print("========================================")