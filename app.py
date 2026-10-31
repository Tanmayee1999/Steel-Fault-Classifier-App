import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, confusion_matrix, ConfusionMatrixDisplay, classification_report)

st.set_page_config(page_title="Steel Plate Fault Checker", layout="wide")
st.title("Steel Plate Fault Checker")
st.write("Upload test data, choose a model, and see how well it detects steel plate faults.")

MODEL_FILES = {
    "Logistic Regression": "model/logistic_regression.joblib",
    "Decision Tree": "model/decision_tree.joblib",
    "KNN": "model/knn.joblib",
    "Naive Bayes": "model/naive_bayes.joblib",
    "SVM": "model/svm.joblib",
}

le = joblib.load("model/label_encoder.joblib")
feature_names = joblib.load("model/feature_names.joblib")

def score_model(model, X, y):
    pred = model.predict(X)
    proba = model.predict_proba(X)
    try:
        auc = roc_auc_score(y, proba, multi_class="ovr", average="weighted")
    except Exception:
        auc = float("nan")
    return {
        "Accuracy": accuracy_score(y, pred),
        "AUC": auc,
        "Precision": precision_score(y, pred, average="weighted", zero_division=0),
        "Recall": recall_score(y, pred, average="weighted", zero_division=0),
        "F1": f1_score(y, pred, average="weighted", zero_division=0),
        "MCC": matthews_corrcoef(y, pred),
    }

choice = st.selectbox("Choose a model", list(MODEL_FILES))
file = st.file_uploader("Upload test CSV", type="csv")

if file:
    data = pd.read_csv(file)
    st.write("Rows loaded:", len(data))
    st.dataframe(data.head())

    missing = [c for c in feature_names + ["Fault"] if c not in data.columns]
    if missing:
        st.error(f"Missing columns: {missing}")
        st.stop()

    X_up = data[feature_names]
    y_up = le.transform(data["Fault"])

    model = joblib.load(MODEL_FILES[choice])
    scores = score_model(model, X_up, y_up)

    st.subheader("Evaluation metrics: " + choice)
    st.dataframe(pd.Series(scores).round(4).to_frame("Score"))

    st.subheader("Confusion matrix")
    fig, ax = plt.subplots()
    ConfusionMatrixDisplay(confusion_matrix(y_up, model.predict(X_up)),
                           display_labels=le.classes_).plot(ax=ax)
    plt.xticks(rotation=45)
    st.pyplot(fig)

    st.subheader("Classification report")
    st.text(classification_report(y_up, model.predict(X_up),
                                  target_names=le.classes_, zero_division=0))
    
    st.subheader("All models compared on your uploaded data")
    all_scores = {}
    for name, path in MODEL_FILES.items():
        all_scores[name] = score_model(joblib.load(path), X_up, y_up)
    comparison = pd.DataFrame(all_scores).T.round(4)
    st.dataframe(comparison)
    st.bar_chart(comparison["MCC"])