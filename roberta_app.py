"""
import streamlit as st
import pandas as pd
import torch
import joblib
import numpy as np
import matplotlib.pyplot as plt
from transformers import RobertaTokenizer, RobertaForSequenceClassification, pipeline
from wordcloud import WordCloud
import torch.nn.functional as F

st.set_page_config(page_title="Market Research Dashboard (ABSA + RoBERTa)", layout="wide")

MODEL_PATH = "roberta_sentiment_model"

# ---------------- LOAD MODELS ---------------- #
@st.cache_resource
def load_models():
    tokenizer = RobertaTokenizer.from_pretrained(MODEL_PATH)
    model = RobertaForSequenceClassification.from_pretrained(MODEL_PATH)
    label_encoder = joblib.load(f"{MODEL_PATH}/label_encoder.pkl")
    model.eval()

    absa_pipeline = pipeline(
    task="text-classification",
    model="yangheng/deberta-v3-base-absa-v1.1",
    tokenizer="yangheng/deberta-v3-base-absa-v1.1",
    device=0 if torch.cuda.is_available() else -1,
    truncation=True
)


    return tokenizer, model, label_encoder, absa_pipeline


tokenizer, model, le, absa_pipeline = load_models()

# ---------------- ASPECTS ---------------- #
ASPECTS = ["price", "quality", "delivery", "service", "packaging", "quantity"]

# ---------------- FUNCTIONS ---------------- #
def roberta_predict(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)

    probs = F.softmax(outputs.logits, dim=1)[0].numpy()
    pred_idx = np.argmax(probs)
    pred_label = le.inverse_transform([pred_idx])[0]
    confidence = float(probs[pred_idx])

    return pred_label, confidence


def extract_absa(text):
    results = []
    for aspect in ASPECTS:
        pair = f"{aspect}: {text}"
        output = absa_pipeline(pair, truncation=True, max_length=256)[0]
        sentiment = output["label"].lower()
        results.append((aspect, sentiment))
    return results


# ---------------- UI ---------------- #
st.title("📊 Market Research Dashboard with ABSA & RoBERTa")

menu = st.sidebar.selectbox("Select Option", ["Single Review Prediction", "Upload & Analyze Reviews"])

# ---------------- SINGLE REVIEW ---------------- #
if menu == "Single Review Prediction":
    text = st.text_area("Enter Review:", height=150)

    if st.button("Predict"):
        sentiment, confidence = roberta_predict(text)
        absa_results = extract_absa(text)

        st.success(f"Overall Sentiment: {sentiment.upper()}")
        st.info(f"Confidence: {round(confidence, 3)}")

        st.subheader("Aspect Based Sentiment")
        st.table(pd.DataFrame(absa_results, columns=["Aspect", "Sentiment"]))


# ---------------- BULK ANALYSIS ---------------- #
elif menu == "Upload & Analyze Reviews":

    uploaded_file = st.file_uploader("Upload CSV (must contain 'review' column)", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)

        if "review" not in df.columns:
            st.error("CSV must contain column named 'review'")
            st.stop()

        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")

        if "product" not in df.columns:
            df["product"] = "General"

        sentiments = []
        confidences = []
        aspect_results = []

        with st.spinner("Analyzing reviews..."):
            for text in df["review"].astype(str):
                sent, conf = roberta_predict(text)
                sentiments.append(sent)
                confidences.append(conf)

                absa = extract_absa(text)
                for asp, asp_sent in absa:
                    aspect_results.append((asp, asp_sent))

        df["predicted_sentiment"] = sentiments
        df["confidence"] = confidences
        df["review_length"] = df["review"].str.len()

        aspect_df = pd.DataFrame(aspect_results, columns=["Aspect", "Sentiment"])

        # ---------------- FILTERS ---------------- #
        st.sidebar.subheader("Filters")

        product_filter = st.sidebar.multiselect("Product", df["product"].unique(), df["product"].unique())
        df_filtered = df[df["product"].isin(product_filter)]

        sentiment_filter = st.sidebar.multiselect(
            "Sentiment",
            df_filtered["predicted_sentiment"].unique(),
            df_filtered["predicted_sentiment"].unique()
        )
        df_filtered = df_filtered[df_filtered["predicted_sentiment"].isin(sentiment_filter)]

        if "date" in df_filtered.columns:
            min_date = df_filtered["date"].min()
            max_date = df_filtered["date"].max()

            date_range = st.sidebar.date_input("Date Range", [min_date, max_date])

            if len(date_range) == 2:
                start_date = pd.to_datetime(date_range[0])
                end_date = pd.to_datetime(date_range[1])
                df_filtered = df_filtered[
                    (df_filtered["date"] >= start_date) &
                    (df_filtered["date"] <= end_date)
                ]

        conf_filter = st.sidebar.slider("Minimum Confidence", 0.0, 1.0, 0.5)
        df_filtered = df_filtered[df_filtered["confidence"] >= conf_filter]

        keyword = st.sidebar.text_input("Keyword Search")
        if keyword:
            df_filtered = df_filtered[df_filtered["review"].str.contains(keyword, case=False)]

        length_filter = st.sidebar.slider("Minimum Review Length", 0, 500, 20)
        df_filtered = df_filtered[df_filtered["review_length"] >= length_filter]

        top_n = st.sidebar.selectbox("Top N Reviews", [50, 100, 200, 500])
        df_filtered = df_filtered.head(top_n)

        # ---------------- VISUALS ---------------- #
        col1, col2 = st.columns(2)

        with col1:
            counts = df_filtered["predicted_sentiment"].value_counts()
            fig1, ax1 = plt.subplots()
            ax1.pie(counts, labels=counts.index, autopct="%1.1f%%")
            ax1.set_title("Sentiment Distribution")
            st.pyplot(fig1)

        with col2:
            if not aspect_df.empty:
                pivot = aspect_df.value_counts().reset_index(name="Count") \
                    .pivot(index="Aspect", columns="Sentiment", values="Count").fillna(0)
                st.bar_chart(pivot)

        if "date" in df_filtered.columns:
            trend = df_filtered.groupby(
                [df_filtered["date"].dt.to_period("M"), "predicted_sentiment"]
            ).size().unstack().fillna(0)
            st.subheader("Sentiment Trend Over Time")
            st.line_chart(trend)

        st.subheader("Word Cloud (Negative Reviews)")
        negative_text = " ".join(df_filtered[df_filtered["predicted_sentiment"] == "negative"]["review"].astype(str))
        if negative_text.strip():
            wc = WordCloud(width=800, height=400, background_color="white").generate(negative_text)
            fig_wc, ax_wc = plt.subplots(figsize=(10, 5))
            ax_wc.imshow(wc)
            ax_wc.axis("off")
            st.pyplot(fig_wc)
        else:
            st.info("No negative reviews")

        if not aspect_df.empty:
            st.subheader("Problem Areas (Negative Aspects)")
            st.write(aspect_df[aspect_df["Sentiment"] == "negative"]["Aspect"].value_counts())

        st.subheader("Final Filtered Data")
        st.dataframe(df_filtered)

        csv = df_filtered.to_csv(index=False).encode("utf-8")
        st.download_button("Download Report", csv, "market_research_report.csv", "text/csv")
        """

import streamlit as st
import pandas as pd
import torch
import joblib
import numpy as np
import matplotlib.pyplot as plt
from transformers import RobertaTokenizer, RobertaForSequenceClassification, pipeline
from wordcloud import WordCloud
import torch.nn.functional as F
from huggingface_hub import hf_hub_download

st.set_page_config(page_title="Market Research Dashboard (ABSA + RoBERTa)", layout="wide")

MODEL_PATH = "parvj-06/roberta-sentiment-classweighted"

@st.cache_resource
def load_models():
    tokenizer = RobertaTokenizer.from_pretrained(MODEL_PATH)
    model = RobertaForSequenceClassification.from_pretrained(MODEL_PATH)
    le_path = hf_hub_download(
    repo_id=MODEL_PATH,
    filename="label_encoder.pkl"
    )
    label_encoder = joblib.load(le_path)
    model.eval()

    absa_pipeline = pipeline(
        task="text-classification",
        model="yangheng/deberta-v3-base-absa-v1.1",
        tokenizer="yangheng/deberta-v3-base-absa-v1.1",
        device=0 if torch.cuda.is_available() else -1,
        truncation=True
    )

    return tokenizer, model, label_encoder, absa_pipeline


tokenizer, model, le, absa_pipeline = load_models()

ASPECTS = ["price", "quality", "delivery", "service", "packaging", "quantity"]

def roberta_predict(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)

    probs = F.softmax(outputs.logits, dim=1)[0].numpy()
    pred_idx = np.argmax(probs)
    pred_label = le.inverse_transform([pred_idx])[0]
    confidence = float(probs[pred_idx])

    return pred_label, confidence


def extract_absa(text):
    results = []
    for aspect in ASPECTS:
        pair = f"{aspect}: {text}"
        output = absa_pipeline(pair, truncation=True, max_length=256)[0]
        sentiment = output["label"].lower()
        results.append((aspect, sentiment))
    return results


st.title("📊 Market Research Dashboard with ABSA & RoBERTa")

menu = st.sidebar.selectbox(
    "Select Option",
    ["Single Review Prediction", "Upload & Analyze Reviews", "Model Comparison"]
)

if menu == "Single Review Prediction":
    text = st.text_area("Enter Review:", height=150)

    if st.button("Predict"):
        sentiment, confidence = roberta_predict(text)
        absa_results = extract_absa(text)

        st.success(f"Overall Sentiment: {sentiment.upper()}")
        st.info(f"Confidence: {round(confidence, 3)}")

        st.subheader("Aspect Based Sentiment")
        st.table(pd.DataFrame(absa_results, columns=["Aspect", "Sentiment"]))


elif menu == "Upload & Analyze Reviews":

    uploaded_file = st.file_uploader("Upload CSV (must contain 'review' column)", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)

        if "review" not in df.columns:
            st.error("CSV must contain column named 'review'")
            st.stop()

        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")

        if "product" not in df.columns:
            df["product"] = "General"

        if "last_file" not in st.session_state:
            st.session_state.last_file = None

        if st.session_state.last_file != uploaded_file.name:
            st.session_state.last_file = uploaded_file.name
            st.session_state.analysis_done = False

        if "analysis_done" not in st.session_state:
            st.session_state.analysis_done = False

        if not st.session_state.analysis_done:

            sentiments = []
            confidences = []
            aspect_results = []

            with st.spinner("Analyzing reviews (first time only)..."):
                for text in df["review"].astype(str):
                    sent, conf = roberta_predict(text)
                    sentiments.append(sent)
                    confidences.append(conf)

                    absa = extract_absa(text)
                    for asp, asp_sent in absa:
                        aspect_results.append((asp, asp_sent))

            df["predicted_sentiment"] = sentiments
            df["confidence"] = confidences
            df["review_length"] = df["review"].str.len()

            st.session_state.df = df
            st.session_state.aspect_df = pd.DataFrame(aspect_results, columns=["Aspect", "Sentiment"])
            st.session_state.analysis_done = True


        df = st.session_state.df
        aspect_df = st.session_state.aspect_df
 

        st.sidebar.subheader("Filters")

        product_filter = st.sidebar.multiselect("Product", df["product"].unique(), df["product"].unique())
        df_filtered = df[df["product"].isin(product_filter)]

        sentiment_filter = st.sidebar.multiselect(
            "Sentiment",
            df_filtered["predicted_sentiment"].unique(),
            df_filtered["predicted_sentiment"].unique()
        )
        df_filtered = df_filtered[df_filtered["predicted_sentiment"].isin(sentiment_filter)]

        if "date" in df_filtered.columns:
            min_date = df_filtered["date"].min()
            max_date = df_filtered["date"].max()
            date_range = st.sidebar.date_input("Date Range", [min_date, max_date])

            if len(date_range) == 2:
                start_date = pd.to_datetime(date_range[0])
                end_date = pd.to_datetime(date_range[1])
                df_filtered = df_filtered[
                    (df_filtered["date"] >= start_date) &
                    (df_filtered["date"] <= end_date)
                ]

        conf_filter = st.sidebar.slider("Minimum Confidence", 0.0, 1.0, 0.5)
        df_filtered = df_filtered[df_filtered["confidence"] >= conf_filter]

        keyword = st.sidebar.text_input("Keyword Search")
        if keyword:
            df_filtered = df_filtered[df_filtered["review"].str.contains(keyword, case=False)]

        length_filter = st.sidebar.slider("Minimum Review Length", 0, 500, 20)
        df_filtered = df_filtered[df_filtered["review_length"] >= length_filter]

        top_n = st.sidebar.selectbox("Top N Reviews", [50, 100, 200, 500])
        df_filtered = df_filtered.head(top_n)


        main_col, ai_col = st.columns([3,1])

        with main_col:
            counts = df_filtered["predicted_sentiment"].value_counts()
            fig1, ax1 = plt.subplots()
            ax1.pie(counts, labels=counts.index, autopct="%1.1f%%")
            ax1.set_title("Sentiment Distribution")
            st.pyplot(fig1)

            st.subheader("Aspect Sentiment Distribution")

            if not aspect_df.empty:
                pivot = aspect_df.value_counts().reset_index(name="Count") \
                    .pivot(index="Aspect", columns="Sentiment", values="Count") \
                    .fillna(0)

                st.bar_chart(pivot)
            else:
                st.info("No aspect data available")
                
            if "date" in df_filtered.columns:
                trend = df_filtered.groupby(
                    [df_filtered["date"].dt.to_period("M"), "predicted_sentiment"]
                ).size().unstack().fillna(0)
                st.subheader("Sentiment Trend Over Time")
                st.line_chart(trend)

            st.subheader("Word Cloud (Negative Reviews)")
            negative_text = " ".join(df_filtered[df_filtered["predicted_sentiment"] == "negative"]["review"].astype(str))
            if negative_text.strip():
                wc = WordCloud(width=800, height=400, background_color="white").generate(negative_text)
                fig_wc, ax_wc = plt.subplots(figsize=(10, 5))
                ax_wc.imshow(wc)
                ax_wc.axis("off")
                st.pyplot(fig_wc)
            else:
                st.info("No negative reviews")

            if not aspect_df.empty:
                st.subheader("Problem Areas (Negative Aspects)")
                st.write(aspect_df[aspect_df["Sentiment"] == "negative"]["Aspect"].value_counts())

            st.subheader("Final Filtered Data")
            st.dataframe(df_filtered)

            st.subheader("Complete Dataset with Predictions")
            st.caption("All uploaded reviews with model predictions and confidence scores")
            st.dataframe(df)
            
            st.download_button(
                "Download Filtered Report",
                df_filtered.to_csv(index=False).encode("utf-8"),
                "filtered_report.csv",
                "text/csv"
            )

            st.download_button(
                "Download Full Dataset",
                df.to_csv(index=False).encode("utf-8"),
                "full_analysis.csv",
                "text/csv"
            )

        with ai_col:

                st.markdown("### 🤖 AI Assistant")

                import requests

                def call_ollama(prompt):
                    try:
                        r = requests.post(
                            "http://localhost:11434/api/generate",
                            json={"model": "llama3", "prompt": prompt, "stream": False},
                            timeout=60
                        )
                        return r.json().get("response", "No response.")
                    except Exception as e:
                        return f"AI error: {e}"

                def build_ai_context(df_f, aspect_df):
                    return f"""
            Sentiment Distribution:
            {df_f["predicted_sentiment"].value_counts().to_string()}

            Negative Aspects:
            {aspect_df[aspect_df["Sentiment"]=="negative"]["Aspect"].value_counts().to_string()}

            Sample Negative Reviews:
            {chr(10).join(df_f[df_f["predicted_sentiment"]=="negative"]["review"].head(5))}
            """

                if "ai_chat" not in st.session_state:
                    st.session_state.ai_chat = []

                chat_box = st.container(height=450)

                with chat_box:
                    for r, m in st.session_state.ai_chat:
                        if r == "user":
                            st.markdown(f"**🧑 You:** {m}")
                        else:
                            st.markdown(f"**🤖 AI:** {m}")

                user_q = st.text_input("Ask AI about issues, trends, improvements")

                colA, colB = st.columns(2)

                with colA:
                    if st.button("Ask"):
                        if user_q.strip():
                            context = build_ai_context(df_filtered, aspect_df)

                            prompt = f"""
            You are a market research analyst.

            DATA:
            {context}

            QUESTION:
            {user_q}

            Answer with insights, causes and business recommendations.
            """

                            with st.spinner("AI thinking..."):
                                reply = call_ollama(prompt)

                            st.session_state.ai_chat.append(("user", user_q))
                            st.session_state.ai_chat.append(("ai", reply))
                            st.rerun()

                with colB:
                    if st.button("Clear"):
                        st.session_state.ai_chat = []
                        st.rerun()

                st.divider()

                if st.session_state.ai_chat:
                    chat_text = "\n\n".join(
                        [f"{r.upper()}: {m}" for r, m in st.session_state.ai_chat]
                    )

                    st.download_button(
                        "⬇ Download Chat",
                        chat_text,
                        "ai_chat.txt",
                        "text/plain"
                    )

                if st.button("📄 Generate AI Report"):
                    context = build_ai_context(df_filtered, aspect_df)

                    report_prompt = f"""
            Create a structured business report:

            1. Executive Summary
            2. Key Problems
            3. Aspect-wise Issues
            4. Actionable Recommendations

            DATA:
            {context}
            """

                    with st.spinner("Generating report..."):
                        report = call_ollama(report_prompt)

                    st.download_button(
                        "Download Report",
                        report,
                        "ai_report.txt",
                        "text/plain"
                    )

elif menu == "Model Comparison":

    st.title("Model Performance Comparison Dashboard")

    st.markdown("### Final Model Evaluation (Class-Weighted Training)")
    st.caption("All models trained and evaluated on the same dataset split using class weights to handle imbalance.")

    st.divider()

    import pandas as pd

    data = {
        "Model": [
            "SVM (Class Weighted)",
            "BERT (Class Weighted)",
            "RoBERTa (Class Weighted - Final)"
        ],
        "Accuracy": [0.8251, 0.8321, 0.8763],
        "Precision": [0.7912, 0.8578, 0.8851],
        "Recall": [0.8251, 0.8321, 0.8763],
        "F1 Score": [0.7948, 0.8432, 0.8803],
        "F1 Macro": [0.5523, 0.6653, 0.7166]
    }

    df_models = pd.DataFrame(data)

    col1, col2, col3 = st.columns(3)

    col1.metric("🏆 Best Accuracy", "RoBERTa", "87.63%")
    col2.metric("🏆 Best F1 Score", "RoBERTa", "88.03%")
    col3.metric("📊 Strong Baseline", "SVM", "82.51%")

    st.divider()

    st.subheader("Detailed Performance Metrics")
    st.dataframe(df_models, use_container_width=True)

    st.subheader("Visual Comparison")
    st.bar_chart(df_models.set_index("Model"))

    st.divider()

    st.info("""
Key Insights from Final Experiments:

• RoBERTa achieved the highest performance across all major metrics.
• Class-weighted training significantly improved handling of imbalanced data.
• BERT showed strong contextual understanding and outperformed SVM.
• SVM remains a solid baseline but struggles with nuanced language patterns.
• Neutral sentiment remains the hardest class across all models.
• Transformer models better capture semantic context than traditional ML.
""")