import streamlit as st
import pandas as pd
import joblib
import numpy as np

# ---------------- CONFIG ----------------
MODEL_PATH = "svm_sentiment_model"

st.set_page_config(page_title="SVM Sentiment Analysis", layout="wide")
st.title("📊 SVM Sentiment Analysis Dashboard")

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    svm_model = joblib.load(f"{MODEL_PATH}/svm_model.pkl")
    tfidf = joblib.load(f"{MODEL_PATH}/tfidf.pkl")
    le = joblib.load(f"{MODEL_PATH}/label_encoder.pkl")
    return svm_model, tfidf, le

svm_model, tfidf, le = load_model()

st.success("✅ SVM Model Loaded Successfully")

# ---------------- PREDICTION FUNCTION ----------------
def predict_svm(text):
    vec = tfidf.transform([text])
    probs = svm_model.predict_proba(vec)[0]

    pred_idx = np.argmax(probs)
    pred_label = le.inverse_transform([pred_idx])[0]
    confidence = probs[pred_idx]

    prob_dict = {
        le.classes_[0]: float(probs[0]),
        le.classes_[1]: float(probs[1]),
        le.classes_[2]: float(probs[2]),
    }

    return pred_label, confidence, prob_dict


# ---------------- MENU ----------------
menu = st.sidebar.selectbox(
    "Choose Option",
    ["Predict Single Review", "Upload CSV File"]
)

# ---------------- SINGLE REVIEW ----------------
if menu == "Predict Single Review":
    st.subheader("✍️ Enter a product review")

    user_text = st.text_area("Review:")

    if st.button("Predict Sentiment"):
        if user_text.strip() == "":
            st.warning("Please enter a review.")
        else:
            label, confidence, probs = predict_svm(user_text)

            st.success(f"Predicted Sentiment: **{label.upper()}**")
            st.info(f"Confidence Score: {confidence:.4f}")

            st.subheader("Class Probabilities")
            st.json(probs)


# ---------------- CSV UPLOAD ----------------
elif menu == "Upload CSV File":
    st.subheader("📂 Upload CSV file with reviews")

    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        st.write("Preview of uploaded data:")
        st.dataframe(df.head())

        if "review" not in df.columns:
            st.error("CSV must contain a column named 'review'")
        else:
            predictions = []
            confidences = []

            for text in df["review"].astype(str):
                label, confidence, _ = predict_svm(text)
                predictions.append(label)
                confidences.append(confidence)

            df["predicted_sentiment"] = predictions
            df["confidence"] = confidences

            st.subheader("✅ Sentiment Distribution")
            st.bar_chart(df["predicted_sentiment"].value_counts())

            st.dataframe(df)

            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download Results as CSV",
                data=csv,
                file_name="svm_sentiment_predictions.csv",
                mime="text/csv",
            )
