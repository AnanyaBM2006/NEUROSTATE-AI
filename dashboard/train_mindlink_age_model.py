# ============================================================
# NEUROSTATE AI
# TRAIN MINDLINK AGE MODEL
#
# Input:
#   mindlink_age_dataset.csv
#
# Output:
#   models/age_estimation/
#       age_model.pkl
#       age_label_encoder.pkl
#       age_features.pkl
# ============================================================

import os
import pickle
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight

from xgboost import XGBClassifier


# ============================================================
# PATHS
# ============================================================

BASE_DIR = r"C:\Users\bmana\Documents\NeuroStateAI"

DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "mindlink_age",
    "mindlink_age_dataset.csv"
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models",
    "age_estimation"
)

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)


# ============================================================
# FEATURES
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
# LOAD DATA
# ============================================================

print()
print("========================================")
print("MINDLINK AGE MODEL TRAINING")
print("========================================")

print()
print("Loading dataset:")

print(DATA_PATH)

df = pd.read_csv(
    DATA_PATH
)

print()
print("Original dataset shape:")
print(df.shape)


# ============================================================
# CHECK REQUIRED COLUMNS
# ============================================================

required_columns = (
    FEATURE_COLUMNS
    + [
        "age",
        "age_group",
        "participant_id",
        "source_file"
    ]
)

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:

    raise ValueError(
        "Missing required columns: "
        + str(missing_columns)
    )


# ============================================================
# REMOVE INVALID VALUES
# ============================================================

df = df.replace(
    [np.inf, -np.inf],
    np.nan
)

df = df.dropna(
    subset=FEATURE_COLUMNS + ["age_group"]
).copy()


# ============================================================
# SHOW ORIGINAL DISTRIBUTION
# ============================================================

print()
print("Original age-group distribution:")

print(
    df["age_group"]
    .value_counts()
)


print()
print("Original participant distribution:")

print(
    df["participant_id"]
    .value_counts()
)


# ============================================================
# BALANCE DATA
#
# We take the same number of samples from each age group.
# ============================================================

class_counts = (
    df["age_group"]
    .value_counts()
)

if len(class_counts) != 2:

    raise ValueError(
        "Expected exactly 2 age groups, "
        f"but found: {list(class_counts.index)}"
    )


minority_count = class_counts.min()

print()
print(
    "Samples available in smallest class:",
    minority_count
)


# ------------------------------------------------------------
# To avoid one long recording dominating the balanced set,
# distribute the available samples approximately equally
# across source files within each age group.
# ------------------------------------------------------------

balanced_parts = []

for age_group in sorted(
    df["age_group"].unique()
):

    group_df = df[
        df["age_group"] == age_group
    ].copy()

    source_files = (
        group_df["source_file"]
        .unique()
    )

    samples_per_file = max(
        1,
        minority_count // len(source_files)
    )

    file_parts = []

    for source_file in source_files:

        file_df = group_df[
            group_df["source_file"] == source_file
        ].copy()

        n = min(
            samples_per_file,
            len(file_df)
        )

        sampled = file_df.sample(
            n=n,
            random_state=42
        )

        file_parts.append(
            sampled
        )

    class_balanced = pd.concat(
        file_parts,
        ignore_index=True
    )

    balanced_parts.append(
        class_balanced
    )


balanced_df = pd.concat(
    balanced_parts,
    ignore_index=True
)


# ============================================================
# FINAL BALANCE CHECK
# ============================================================

print()
print("Balanced dataset shape:")

print(
    balanced_df.shape
)

print()
print("Balanced age-group distribution:")

print(
    balanced_df[
        "age_group"
    ].value_counts()
)


print()
print("Balanced source-file distribution:")

print(
    balanced_df[
        "source_file"
    ].value_counts()
)


# ============================================================
# FEATURES / TARGET
# ============================================================

X = balanced_df[
    FEATURE_COLUMNS
].copy()

y_text = balanced_df[
    "age_group"
].copy()


# ============================================================
# LABEL ENCODING
# ============================================================

label_encoder = LabelEncoder()

y = label_encoder.fit_transform(
    y_text
)

print()
print("Encoded classes:")

for index, label in enumerate(
    label_encoder.classes_
):

    print(
        index,
        "->",
        label
    )


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


print()
print("Training shape:")
print(X_train.shape)

print()
print("Testing shape:")
print(X_test.shape)


# ============================================================
# SAMPLE WEIGHTS
# ============================================================

sample_weights = compute_sample_weight(
    class_weight="balanced",
    y=y_train
)


# ============================================================
# XGBOOST MODEL
# ============================================================

model = XGBClassifier(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42
)


print()
print("========================================")
print("TRAINING XGBOOST")
print("========================================")


model.fit(
    X_train,
    y_train,
    sample_weight=sample_weights
)


# ============================================================
# TEST
# ============================================================

y_pred = model.predict(
    X_test
)

y_probability = model.predict_proba(
    X_test
)


# ============================================================
# METRICS
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

print()
print("========================================")
print("MINDLINK AGE MODEL RESULTS")
print("========================================")

print()
print(
    "Accuracy:",
    round(accuracy * 100, 2),
    "%"
)


print()
print("Classification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=label_encoder.classes_,
        zero_division=0
    )
)


print()
print("Confusion Matrix:")

print(
    confusion_matrix(
        y_test,
        y_pred
    )
)


# ============================================================
# SAMPLE PREDICTIONS
# ============================================================

print()
print("========================================")
print("SAMPLE PREDICTIONS")
print("========================================")

sample_count = min(
    10,
    len(X_test)
)

sample_X = X_test.iloc[
    :sample_count
]

sample_y = y_test[
    :sample_count
]

sample_pred = model.predict(
    sample_X
)

sample_prob = model.predict_proba(
    sample_X
)


for i in range(
    sample_count
):

    predicted_label = (
        label_encoder
        .inverse_transform(
            [sample_pred[i]]
        )[0]
    )

    actual_label = (
        label_encoder
        .inverse_transform(
            [sample_y[i]]
        )[0]
    )

    confidence = (
        np.max(
            sample_prob[i]
        )
        * 100
    )

    print()
    print(
        "Sample",
        i + 1
    )

    print(
        "Actual:",
        actual_label
    )

    print(
        "Predicted:",
        predicted_label
    )

    print(
        "Confidence:",
        round(
            confidence,
            2
        ),
        "%"
    )


# ============================================================
# SAVE MODEL
# ============================================================

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "age_model.pkl"
)

ENCODER_PATH = os.path.join(
    MODEL_DIR,
    "age_label_encoder.pkl"
)

FEATURE_PATH = os.path.join(
    MODEL_DIR,
    "age_features.pkl"
)


# ------------------------------------------------------------
# Save model
# ------------------------------------------------------------

with open(
    MODEL_PATH,
    "wb"
) as file:

    pickle.dump(
        model,
        file
    )


# ------------------------------------------------------------
# Save label encoder
# ------------------------------------------------------------

with open(
    ENCODER_PATH,
    "wb"
) as file:

    pickle.dump(
        label_encoder,
        file
    )


# ------------------------------------------------------------
# Save feature list
# ------------------------------------------------------------

with open(
    FEATURE_PATH,
    "wb"
) as file:

    pickle.dump(
        FEATURE_COLUMNS,
        file
    )


# ============================================================
# FINAL MESSAGE
# ============================================================

print()
print("========================================")
print("AGE MODEL SAVED SUCCESSFULLY")
print("========================================")

print()
print(
    "age_model.pkl"
)

print(
    "age_label_encoder.pkl"
)

print(
    "age_features.pkl"
)

print()
print(
    "Location:",
    MODEL_DIR
)

print()
print("========================================")
print("MINDLINK AGE TRAINING COMPLETE")
print("========================================")