"""Student Social-Media Impact Classifier - Streamlit app (ML Assignment 2).

Loads the five saved scikit-learn pipelines from model/ and scores an uploaded
test CSV with them. Nothing is trained or fitted in this file.
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score, recall_score,
                             f1_score, matthews_corrcoef, classification_report,
                             ConfusionMatrixDisplay)

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
BASE_DIR = Path(__file__).parent
MODEL_DIR = BASE_DIR / "model"
BUNDLED_TEST_FILE = BASE_DIR / "test_data.csv"

TARGET = "Overall_Impact"
NUMERIC_FEATURES = ["Age", "Daily_Usage_Hours", "Weekend_Extra_Hours", "Sleep_Duration_Hours",
                    "Sleep_Quality_Score", "Perceived_Stress_Score", "Mental_Health_Index",
                    "Academic_Performance_GPA"]
CATEGORICAL_FEATURES = ["Gender", "Academic_Level", "Primary_Platform", "Device_Type",
                        "Social_Comparison_Frequency"]
FLAG_FEATURE = "Late_Night_Usage"
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES + [FLAG_FEATURE]

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "svm.joblib",
}
MODEL_NOTES = {
    "Logistic Regression": "Linear model on standardised features. Fast and easy to interpret.",
    "Decision Tree": "Learns a tree of if/else splits on the readings.",
    "KNN": "Labels a student like the most similar students in the training data.",
    "Naive Bayes": "Probabilistic model that assumes the features are independent.",
    "Support Vector Machine": "Finds the widest-margin boundary between the three classes.",
}
METRIC_NAMES = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]


# --------------------------------------------------------------------------- #
# Helper functions (no Streamlit calls, so they are easy to test)
# --------------------------------------------------------------------------- #
@st.cache_resource
def load_model(model_name):
    """Load one saved pipeline (preprocessing + classifier) from disk."""
    return joblib.load(MODEL_DIR / MODEL_FILES[model_name])


def prepare_input(raw):
    """Check an uploaded dataframe and return (clean_df, errors, warnings).

    errors   -> the file cannot be scored
    warnings -> the file can be scored, but the user should know what was adjusted
    """
    errors, warnings = [], []

    missing_cols = [c for c in FEATURES + [TARGET] if c not in raw.columns]
    if missing_cols:
        errors.append("Missing required column(s): " + ", ".join(missing_cols))
        return None, errors, warnings

    df = raw[FEATURES + [TARGET]].copy()        # extra columns / different order are fine

    # The True/False column may arrive as bool, text ("True") or 0/1
    flag_map = {True: True, False: False, "true": True, "false": False,
                "1": True, "0": False, 1: True, 0: False}
    df[FLAG_FEATURE] = df[FLAG_FEATURE].map(
        lambda v: flag_map.get(v.strip().lower() if isinstance(v, str) else v, np.nan))
    if df[FLAG_FEATURE].isna().any():
        warnings.append(f"{int(df[FLAG_FEATURE].isna().sum())} unreadable value(s) in "
                        f"{FLAG_FEATURE}; the saved model treats them as unknown.")
    df[FLAG_FEATURE] = df[FLAG_FEATURE].astype(object)

    # Numeric columns must really be numeric
    for col in NUMERIC_FEATURES:
        before = df[col].isna().sum()
        df[col] = pd.to_numeric(df[col], errors="coerce")
        created = int(df[col].isna().sum() - before)
        if created > 0:
            warnings.append(f"{created} non-numeric value(s) in {col} were treated as missing.")

    # Rows without a target cannot be evaluated
    no_target = df[TARGET].isna()
    if no_target.any():
        warnings.append(f"{int(no_target.sum())} row(s) without a {TARGET} value were dropped.")
        df = df[~no_target]
    if df.empty:
        errors.append("The file has no rows with a target value.")
        return None, errors, warnings

    df[TARGET] = df[TARGET].astype(str).str.strip()
    return df.reset_index(drop=True), errors, warnings


def evaluate(model, X, y):
    """Predict with a saved model and compute the six metrics (same definitions as the notebook)."""
    y_pred = model.predict(X)
    proba = model.predict_proba(X)
    classes = list(model.classes_)

    # AUC (one-vs-rest, macro). If the file contains only some classes, use only those.
    present = [c for c in classes if c in set(y)]
    try:
        if len(present) < 2:
            auc = np.nan
        elif len(present) == len(classes):
            auc = roc_auc_score(y, proba, multi_class="ovr", average="macro", labels=classes)
        else:
            idx = [classes.index(c) for c in present]
            sub = proba[:, idx]
            sub = sub / sub.sum(axis=1, keepdims=True)
            if len(present) == 2:                       # plain binary AUC for the second class
                auc = roc_auc_score(np.asarray(y) == present[1], sub[:, 1])
            else:
                auc = roc_auc_score(y, sub, multi_class="ovr", average="macro", labels=present)
    except ValueError:
        auc = np.nan

    metrics = {
        "Accuracy": accuracy_score(y, y_pred),
        "AUC": auc,
        "Precision": precision_score(y, y_pred, average="macro", zero_division=0),
        "Recall": recall_score(y, y_pred, average="macro", zero_division=0),
        "F1": f1_score(y, y_pred, average="macro", zero_division=0),
        "MCC": matthews_corrcoef(y, y_pred),
    }
    return metrics, y_pred


def format_metric(value):
    return "n/a" if pd.isna(value) else f"{value:.4f}"


def confusion_figure(model, y, y_pred):
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    ConfusionMatrixDisplay.from_predictions(y, y_pred, labels=list(model.classes_),
                                            cmap="Blues", ax=ax, colorbar=False)
    ax.set_title("Confusion matrix (rows = actual, columns = predicted)", fontsize=10)
    ax.grid(False)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# Page
# --------------------------------------------------------------------------- #
st.set_page_config(page_title="Social Media Impact Classifier", page_icon="📱", layout="wide")

st.title("Social Media Impact Classifier")
st.write("Five pre-trained classifiers predict whether a student's social-media use has a "
         "**Beneficial**, **Neutral** or **Negative** overall impact. Upload a test CSV, pick a "
         "model and see how it performs. The saved models are only loaded here, never retrained.")

# ---- Sidebar: data + model selection --------------------------------------
with st.sidebar:
    st.header("1. Test data")
    uploaded = st.file_uploader("Upload test CSV", type=["csv"],
                                help="Needs the 14 feature columns and the Overall_Impact column.")
    use_bundled = st.checkbox("Use the bundled test_data.csv", value=True,
                              help="Ignored when you upload your own file.")
    if BUNDLED_TEST_FILE.exists():
        st.download_button("Download test_data.csv", BUNDLED_TEST_FILE.read_bytes(),
                           file_name="test_data.csv", mime="text/csv")

    st.header("2. Model")
    model_name = st.selectbox("Choose a classifier", list(MODEL_FILES))
    st.caption(MODEL_NOTES[model_name])

# ---- Read the data ----------------------------------------------------------
if uploaded is not None:
    source_label = uploaded.name
    try:
        raw = pd.read_csv(uploaded)
    except Exception as exc:                     # unreadable / not a CSV
        st.error(f"Could not read the uploaded file as a CSV: {exc}")
        st.stop()
elif use_bundled and BUNDLED_TEST_FILE.exists():
    source_label = "test_data.csv (bundled)"
    raw = pd.read_csv(BUNDLED_TEST_FILE)
else:
    st.info("Upload a test CSV in the sidebar, or tick the bundled test data, to begin.")
    st.stop()

data, errors, warnings = prepare_input(raw)
for message in errors:
    st.error(message)
if errors:
    st.stop()
for message in warnings:
    st.warning(message)

model = load_model(model_name)
unknown_labels = sorted(set(data[TARGET]) - set(model.classes_))
if unknown_labels:
    st.error(f"{TARGET} contains labels the models do not know: {unknown_labels}. "
             f"Expected only: {list(model.classes_)}.")
    st.stop()

X, y = data[FEATURES], data[TARGET]
st.caption(f"Source: {source_label}  |  Rows: {len(data)}  |  Features: {len(FEATURES)}  |  "
           f"Classes in file: {y.nunique()}")


# ---- Section 1: scorecard, confusion matrix and report for the selected model ----
scorecard = st.container()
with scorecard:
    metrics, y_pred = evaluate(model, X, y)
    n_correct = int((y_pred == y).sum())
    st.subheader(f"{model_name}: {n_correct} of {len(y)} students classified correctly")

    row1 = st.columns(3)
    row2 = st.columns(3)
    for col, name in zip(row1 + row2, METRIC_NAMES):
        col.metric(name, format_metric(metrics[name]))
    st.caption("Precision, Recall and F1 are macro-averaged; AUC is one-vs-rest macro-averaged; "
               "MCC is the multi-class version.")

    left, right = st.columns(2)
    with left:
        st.markdown("**Confusion matrix**")
        fig = confusion_figure(model, y, y_pred)
        st.pyplot(fig)
        plt.close(fig)
    with right:
        st.markdown("**Classification report**")
        report = pd.DataFrame(classification_report(y, y_pred, output_dict=True,
                                                    zero_division=0)).T
        report["support"] = report["support"].astype(int)
        st.dataframe(report.round(3))

    predictions = data.copy()
    predictions["Predicted_" + TARGET] = y_pred
    st.download_button("Download predictions as CSV", predictions.to_csv(index=False),
                       file_name="predictions.csv", mime="text/csv")

# ---- Section 2: all five models on the same file -----------------------------
st.divider()
with st.expander("Compare all five models on this file", expanded=True):
    rows = {name: evaluate(load_model(name), X, y)[0] for name in MODEL_FILES}
    comparison = pd.DataFrame(rows).T[METRIC_NAMES]
    comparison.index.name = "ML Model Name"
    st.subheader("All five models on this file")
    st.dataframe(comparison.round(4))

    chosen_metric = st.selectbox("Metric to plot", METRIC_NAMES, index=METRIC_NAMES.index("MCC"))
    st.bar_chart(comparison[chosen_metric])
    if comparison[chosen_metric].notna().any():
        st.caption(f"Best model for {chosen_metric} on this file: "
                   f"**{comparison[chosen_metric].idxmax()}**")

# ---- Section 3: data preview ----------------------------------------------------
with st.expander("Peek at the data", expanded=False):
    st.subheader("Class balance in this file")
    st.bar_chart(y.value_counts())
    st.subheader("First rows")
    st.dataframe(data.head(20))
