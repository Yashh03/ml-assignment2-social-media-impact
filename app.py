from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score, recall_score,
                             f1_score, matthews_corrcoef, classification_report,
                             confusion_matrix, ConfusionMatrixDisplay, roc_curve)

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
MODEL_SETTINGS = {
    "Logistic Regression": ["C", "class_weight", "max_iter"],
    "Decision Tree": ["criterion", "max_depth", "min_samples_leaf"],
    "KNN": ["n_neighbors", "weights"],
    "Naive Bayes": ["var_smoothing"],
    "Support Vector Machine": ["kernel", "C", "class_weight"],
}
METRIC_NAMES = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]
CLASS_COLORS = {"Beneficial": "#3f9b6e", "Neutral": "#e0a030", "Negative": "#d05050"}

CATEGORY_OPTIONS = {
    "Gender": ["Female", "Male", "Non-Binary", "Prefer not to say"],
    "Academic_Level": ["High School", "Undergraduate", "Postgraduate"],
    "Primary_Platform": ["Instagram", "TikTok", "YouTube", "Snapchat", "Reddit",
                         "X (Twitter)", "LinkedIn"],
    "Device_Type": ["Smartphone", "Laptop/PC", "Tablet"],
    "Social_Comparison_Frequency": ["Never", "Rarely", "Sometimes", "Frequently", "Always"],
}
NUMERIC_RANGES = {
    "Age": (15, 26, 19, 1),
    "Daily_Usage_Hours": (0.9, 14.0, 4.7, 0.1),
    "Weekend_Extra_Hours": (0.0, 4.5, 1.8, 0.1),
    "Sleep_Duration_Hours": (3.0, 10.5, 6.8, 0.1),
    "Sleep_Quality_Score": (1, 5, 3, 1),
    "Perceived_Stress_Score": (0.0, 40.0, 13.0, 1.0),
    "Mental_Health_Index": (32, 98, 82, 1),
    "Academic_Performance_GPA": (1.9, 4.0, 3.46, 0.01),
}


@st.cache_resource
def load_model(model_name):
    return joblib.load(MODEL_DIR / MODEL_FILES[model_name])


def prepare_input(raw):
    errors, warnings = [], []

    missing_cols = [c for c in FEATURES + [TARGET] if c not in raw.columns]
    if missing_cols:
        errors.append("Missing required column(s): " + ", ".join(missing_cols))
        return None, errors, warnings

    df = raw[FEATURES + [TARGET]].copy()

    flag_map = {True: True, False: False, "true": True, "false": False,
                "1": True, "0": False, 1: True, 0: False}
    df[FLAG_FEATURE] = df[FLAG_FEATURE].map(
        lambda v: flag_map.get(v.strip().lower() if isinstance(v, str) else v, np.nan))
    if df[FLAG_FEATURE].isna().any():
        warnings.append(f"{int(df[FLAG_FEATURE].isna().sum())} unreadable value(s) in "
                        f"{FLAG_FEATURE}; the saved model treats them as unknown.")
    df[FLAG_FEATURE] = df[FLAG_FEATURE].astype(object)

    for col in NUMERIC_FEATURES:
        before = df[col].isna().sum()
        df[col] = pd.to_numeric(df[col], errors="coerce")
        created = int(df[col].isna().sum() - before)
        if created > 0:
            warnings.append(f"{created} non-numeric value(s) in {col} were treated as missing.")

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
    y_pred = model.predict(X)
    proba = model.predict_proba(X)
    classes = list(model.classes_)

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
            if len(present) == 2:
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
    return metrics, y_pred, proba


def format_metric(value):
    return "n/a" if pd.isna(value) else f"{value:.4f}"


def compare_all_models(X, y):
    rows = {name: evaluate(load_model(name), X, y)[0] for name in MODEL_FILES}
    table = pd.DataFrame(rows).T[METRIC_NAMES]
    table.index.name = "ML Model Name"
    return table


def model_settings(model_name, model):
    params = model.named_steps["clf"].get_params()
    return pd.DataFrame({"Setting": MODEL_SETTINGS[model_name],
                         "Value": [str(params[k]) for k in MODEL_SETTINGS[model_name]]})


def predict_student(row):
    one = pd.DataFrame([row])[FEATURES]
    out = []
    for name in MODEL_FILES:
        m = load_model(name)
        proba = m.predict_proba(one)[0]
        best = int(np.argmax(proba))
        out.append({"Model": name, "Predicted impact": str(m.classes_[best]),
                    "Confidence": float(proba[best]), "_proba": proba, "_classes": list(m.classes_)})
    return out


def fig_confusion(model, y, y_pred, as_percent):
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    ConfusionMatrixDisplay.from_predictions(
        y, y_pred, labels=list(model.classes_), cmap="Blues", ax=ax, colorbar=False,
        normalize="true" if as_percent else None, values_format=".2f" if as_percent else "d")
    ax.set_title("Rows = actual class, columns = predicted class", fontsize=10)
    ax.grid(False)
    fig.tight_layout()
    return fig


def fig_per_class(report):
    classes = [c for c in report.index if c in CLASS_COLORS]
    x = np.arange(len(classes))
    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    for i, col in enumerate(["precision", "recall", "f1-score"]):
        ax.bar(x + (i - 1) * 0.26, report.loc[classes, col], width=0.26, label=col)
    ax.set_xticks(x)
    ax.set_xticklabels(classes)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Score")
    ax.set_title("Per-class precision, recall and F1", fontsize=10)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    return fig


def fig_roc(model, y, proba):
    fig, ax = plt.subplots(figsize=(5.4, 4.2))
    y = np.asarray(y)
    for i, cls in enumerate(model.classes_):
        positive = y == cls
        if positive.sum() == 0 or positive.sum() == len(y):
            continue
        fpr, tpr, _ = roc_curve(positive, proba[:, i])
        area = roc_auc_score(positive, proba[:, i])
        ax.plot(fpr, tpr, color=CLASS_COLORS.get(cls), label=f"{cls} (AUC {area:.3f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("One-vs-rest ROC curves", fontsize=10)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    return fig


def fig_compare(table):
    fig, ax = plt.subplots(figsize=(10, 4.2))
    models = list(table.index)
    x = np.arange(len(METRIC_NAMES))
    width = 0.8 / len(models)
    for i, name in enumerate(models):
        ax.bar(x + (i - (len(models) - 1) / 2) * width, table.loc[name].fillna(0).values,
               width=width, label=name)
    ax.set_xticks(x)
    ax.set_xticklabels(METRIC_NAMES)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score")
    ax.set_title("All five models, all six metrics", fontsize=11)
    ax.legend(ncol=5, loc="upper center", fontsize=8)
    fig.tight_layout()
    return fig


def fig_probabilities(classes, proba):
    fig, ax = plt.subplots(figsize=(5, 2.6))
    ax.barh(classes, proba, color=[CLASS_COLORS.get(c, "#888") for c in classes])
    ax.set_xlim(0, 1)
    ax.set_xlabel("Predicted probability")
    for i, p in enumerate(proba):
        ax.text(min(p + 0.02, 0.85), i, f"{p:.1%}", va="center", fontsize=9)
    fig.tight_layout()
    return fig


def fig_feature_hist(data, feature):
    fig, ax = plt.subplots(figsize=(6, 3.6))
    for cls in ["Beneficial", "Neutral", "Negative"]:
        values = data.loc[data[TARGET] == cls, feature].dropna()
        if len(values):
            ax.hist(values, bins=25, alpha=0.55, color=CLASS_COLORS[cls], label=cls, density=True)
    ax.set_xlabel(feature)
    ax.set_ylabel("Share of students (density)")
    ax.set_title(f"{feature} by overall impact", fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def show_figure(fig):
    st.pyplot(fig)
    plt.close(fig)


st.set_page_config(page_title="Social Media Impact Lab", page_icon="📱", layout="wide")

st.markdown(
    """
    <style>
      .hero {background: linear-gradient(90deg, #1f3b73 0%, #3a7bd5 100%);
             padding: 1.1rem 1.5rem; border-radius: 12px; margin-bottom: 1rem;}
      .hero h1 {margin: 0; font-size: 1.9rem; color: #ffffff;}
      .hero p {margin: 0.35rem 0 0 0; color: #e8f0ff;}
    </style>
    <div class="hero">
      <h1>📱 Social Media Impact Lab</h1>
      <p>Five pre-trained classifiers predict whether a student's social-media use has a
      Beneficial, Neutral or Negative overall impact. The saved models are only loaded here,
      never retrained.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

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

model = load_model(model_name)

view = st.radio("View", ["Model report", "Compare models", "Try a student", "Data explorer"],
                horizontal=True, label_visibility="collapsed")

data, source_label, errors, warnings = None, None, [], []
if uploaded is not None:
    source_label = uploaded.name
    try:
        raw = pd.read_csv(uploaded)
        data, errors, warnings = prepare_input(raw)
    except Exception as exc:
        errors = [f"Could not read the uploaded file as a CSV: {exc}"]
elif use_bundled and BUNDLED_TEST_FILE.exists():
    source_label = "test_data.csv (bundled)"
    data, errors, warnings = prepare_input(pd.read_csv(BUNDLED_TEST_FILE))

if view != "Try a student":
    for message in errors:
        st.error(message)
    if errors:
        st.stop()
    if data is None:
        st.info("Upload a test CSV in the sidebar, or tick the bundled test data, to begin.")
        st.stop()
    for message in warnings:
        st.warning(message)
    unknown_labels = sorted(set(data[TARGET]) - set(model.classes_))
    if unknown_labels:
        st.error(f"{TARGET} contains labels the models do not know: {unknown_labels}. "
                 f"Expected only: {list(model.classes_)}.")
        st.stop()
    X, y = data[FEATURES], data[TARGET]
    st.caption(f"Source: {source_label}  |  Rows: {len(data)}  |  Features: {len(FEATURES)}  |  "
               f"Classes in file: {y.nunique()}")

if view == "Model report":
    metrics, y_pred, proba = evaluate(model, X, y)
    comparison = compare_all_models(X, y)
    n_correct = int((y_pred == y).sum())
    st.subheader(f"{model_name}: {n_correct} of {len(y)} students classified correctly")

    row1, row2 = st.columns(3), st.columns(3)
    for col, name in zip(row1 + row2, METRIC_NAMES):
        average = comparison[name].mean()
        delta = None if pd.isna(metrics[name]) or pd.isna(average) \
            else f"{metrics[name] - average:+.4f} vs average of 5 models"
        col.metric(name, format_metric(metrics[name]), delta)
    if comparison["MCC"].notna().all():
        rank = int(comparison["MCC"].rank(ascending=False)[model_name])
        st.caption(f"Ranked {rank} of 5 models by MCC on this file. Precision, Recall and F1 are "
                   "macro-averaged; AUC is one-vs-rest macro-averaged; MCC is the multi-class version.")

    left, right = st.columns(2)
    with left:
        st.markdown("**Confusion matrix**")
        mode = st.radio("Show matrix as", ["Counts", "Row %"], horizontal=True)
        show_figure(fig_confusion(model, y, y_pred, as_percent=(mode == "Row %")))
    with right:
        st.markdown("**Classification report**")
        report = pd.DataFrame(classification_report(y, y_pred, output_dict=True,
                                                    zero_division=0)).T
        report["support"] = report["support"].astype(int)
        st.dataframe(report.round(3))
        show_figure(fig_per_class(report))

    with st.expander("ROC curves (one-vs-rest)"):
        show_figure(fig_roc(model, y, proba))

    with st.expander("Students this model got wrong"):
        mistakes = data.copy()
        mistakes["Predicted_" + TARGET] = y_pred
        mistakes = mistakes[y_pred != y.values]
        st.write(f"{len(mistakes)} misclassified row(s).")
        st.dataframe(mistakes)
        predictions = data.copy()
        predictions["Predicted_" + TARGET] = y_pred
        st.download_button("Download all predictions as CSV", predictions.to_csv(index=False),
                           file_name="predictions.csv", mime="text/csv")

    with st.expander("Tuned settings of this model"):
        st.dataframe(model_settings(model_name, model))

elif view == "Compare models":
    comparison = compare_all_models(X, y)
    st.subheader("All five models on this file")
    st.dataframe(comparison.round(4))

    best = pd.DataFrame({
        "Best model": {m: (comparison[m].idxmax() if comparison[m].notna().any() else "n/a")
                       for m in METRIC_NAMES},
        "Score": comparison.max().round(4)})
    best.index.name = "Metric"
    st.markdown("**Best model for each metric**")
    st.dataframe(best)
    show_figure(fig_compare(comparison))
    if comparison["MCC"].notna().any():
        st.caption(f"Winner by MCC on this file: **{comparison['MCC'].idxmax()}**. "
                   "MCC is the most reliable single score for an imbalanced dataset.")

elif view == "Try a student":
    st.subheader("Predict the impact for one student")
    st.write("Set the values below. All five saved models predict live; no data file is needed.")

    row = {}
    left, right = st.columns(2)
    numeric_columns = [left, right]
    for i, name in enumerate(NUMERIC_FEATURES):
        low, high, default, step = NUMERIC_RANGES[name]
        with numeric_columns[i % 2]:
            row[name] = st.slider(name.replace("_", " "), low, high, default, step)
    cat_cols = st.columns(len(CATEGORICAL_FEATURES))
    for col, name in zip(cat_cols, CATEGORICAL_FEATURES):
        with col:
            row[name] = st.selectbox(name.replace("_", " "), CATEGORY_OPTIONS[name])
    row[FLAG_FEATURE] = bool(st.checkbox("Uses social media late at night", value=True))

    results = predict_student(row)
    chosen = next(r for r in results if r["Model"] == model_name)

    st.divider()
    a, b = st.columns(2)
    with a:
        st.markdown(f"**{model_name}** predicts: **{chosen['Predicted impact']}** "
                    f"({chosen['Confidence']:.1%} confidence)")
        show_figure(fig_probabilities(chosen["_classes"], chosen["_proba"]))
    with b:
        st.markdown("**What every model says**")
        summary = pd.DataFrame([{"Model": r["Model"], "Predicted impact": r["Predicted impact"],
                                 "Confidence": f"{r['Confidence']:.1%}"} for r in results])
        st.dataframe(summary)
        labels = {r["Predicted impact"] for r in results}
        if len(labels) == 1:
            st.success("All five models agree on this student.")
        else:
            st.warning("The models disagree on this student, so treat the prediction with care.")

else:
    st.subheader("Class balance")
    st.bar_chart(y.value_counts())
    st.subheader("How a feature differs between classes")
    feature = st.selectbox("Feature", NUMERIC_FEATURES, index=NUMERIC_FEATURES.index("Mental_Health_Index"))
    show_figure(fig_feature_hist(data, feature))
    st.subheader("Numeric summary")
    st.dataframe(data[NUMERIC_FEATURES].describe().T.round(2))
    st.subheader("First rows")
    st.dataframe(data.head(20))
