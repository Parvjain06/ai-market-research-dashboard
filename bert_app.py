import streamlit as st
import pandas as pd
import torch
import joblib
import torch.nn.functional as F
from transformers import BertTokenizer, BertForSequenceClassification

# ---------------- CONFIG ----------------
st.set_page_config(page_title="BERT Sentiment App", layout="wide")

MODEL_PATH = "bert_sentiment_model"

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
    model = BertForSequenceClassification.from_pretrained(MODEL_PATH)
    label_encoder = joblib.load(f"{MODEL_PATH}/label_encoder.pkl")
    model.eval()
    return tokenizer, model, label_encoder

tokenizer, model, le = load_model()

# ---------------- PREDICTION FUNCTION ----------------
def bert_predict(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)

    with torch.no_grad():
        outputs = model(**inputs)

    probs = F.softmax(outputs.logits, dim=1)[0].numpy()

    pred_idx = probs.argmax()
    pred_label = le.inverse_transform([pred_idx])[0]
    confidence = float(probs[pred_idx])

    prob_dict = dict(zip(le.classes_, probs))

    return pred_label, confidence, prob_dict


# ---------------- UI ----------------
st.title("📊 Market Research using BERT Sentiment Analysis")
st.write("Predict sentiment from customer reviews using trained BERT model")

menu = st.sidebar.selectbox(
    "Choose Option",
    ["Predict Single Review", "Upload CSV File"]
)

# ---------------- SINGLE REVIEW ----------------
if menu == "Predict Single Review":
    st.subheader("✍️ Enter a product review")

    text = st.text_area("Review:", height=150)

    if st.button("Predict Sentiment"):
        if text.strip() == "":
            st.warning("Please enter a review.")
        else:
            sentiment, confidence, probs = bert_predict(text)

            st.success(f"**Predicted Sentiment: {sentiment.upper()}**")
            st.info(f"Confidence Score: {round(confidence, 4)}")

            st.subheader("Class Probabilities")
            st.json(probs)


# ---------------- CSV UPLOAD ----------------
elif menu == "Upload CSV File":
    st.subheader("📂 Upload CSV File with reviews")

    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        st.write("Preview of uploaded data:")
        st.dataframe(df.head())

        if "review" not in df.columns:
            st.error("CSV must contain a column named 'review'")
        else:
            sentiments = []
            confidences = []

            with st.spinner("Predicting sentiments..."):
                for review in df["review"].astype(str):
                    sent, conf, _ = bert_predict(review)
                    sentiments.append(sent)
                    confidences.append(conf)

            df["predicted_sentiment"] = sentiments
            df["confidence"] = confidences   # ✅ confidence column added

            st.subheader("✅ Results")
            st.dataframe(df)

            st.subheader("📊 Sentiment Distribution")
            st.bar_chart(df["predicted_sentiment"].value_counts())

            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download Results CSV",
                csv,
                "bert_predictions.csv",
                "text/csv"
            )
