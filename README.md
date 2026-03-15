# SentIQ — Market Intelligence Dashboard

> AI-powered sentiment intelligence platform using RoBERTa, DeBERTa ABSA, and LLaMA 3.3 for large-scale customer review analysis.

---

## 🚀 Live Demo

> Deployed on Streamlit Community Cloud
> https://ai-market-research-dashboard-hju8skkacvqfxv3n8k8ddy.streamlit.app

---

## 📌 What is SentIQ?

SentIQ is a production-ready NLP system that automatically understands customer feedback at scale and extracts actionable business insights.

It combines classical ML, transformer models, aspect-based sentiment analysis, and an AI analyst powered by LLaMA 3.3 (via Groq), all wrapped in a clean, professional Streamlit dashboard.

---

## 🧠 Models

All models trained on the same dataset split with class-weighted training to handle imbalance.

| Model | Accuracy | Weighted F1 | Macro F1 |
|-------|----------|-------------|----------|
| SVM (Baseline) | 82.51% | 79.48% | 55.23% |
| BERT | 83.21% | 84.32% | 66.53% |
| **RoBERTa (Production)** | **87.63%** | **88.03%** | **71.66%** |

RoBERTa is used as the production model, hosted on HuggingFace and auto-downloaded at runtime.

**Model:** https://huggingface.co/parvj-06/roberta-sentiment-classweighted

---

## ⚖️ Class Imbalance Strategy

- Positive → dominant class
- Negative → medium
- Neutral → minority (hardest class)

Techniques explored: random undersampling, SMOTE, class weights.  
**Final choice:** class-weighted training → preserved data volume and gave best performance.

---

## 🔍 Aspect-Based Sentiment Analysis (ABSA)

Uses **DeBERTa** to extract sentiment across 6 business-critical aspects:

`Price` · `Quality` · `Delivery` · `Service` · `Packaging` · `Quantity`

Enables precise insights like:
- *"Customers praise quality but consistently complain about delivery"*
- *"Price dissatisfaction is the #1 negative aspect (38% of complaints)"*

---

## 📊 Dashboard Features

### Upload & Analyze
- Upload any CSV with a `review` column
- Batched RoBERTa inference (fast as it processes all reviews at once)
- Batched ABSA extraction across all 6 aspects
- Confidence scoring per prediction

### Visual Analytics (Plotly)
- Donut chart — sentiment distribution
- Bar chart — avg confidence by sentiment
- Histogram — confidence score distribution
- Aspect grouped bar chart
- Aspect health radar chart
- Sentiment trend over time (line chart)
- Rolling sentiment score (smoothed)
- Word cloud — switchable between positive/negative/neutral

### Smart Filtering (inline card)
- Product filter
- Sentiment filter
- Keyword search
- Confidence threshold slider
- Show top N selector
- Date range (if date column present)

### Data Export
- Download filtered CSV
- Download full dataset with predictions

### 🤖 SentIQ Analyst (AI Chat)
- Powered by **LLaMA 3.3 70B via Groq API** (free, fast, deployable)
- Answers questions grounded strictly in your uploaded data
- Quick insight buttons: Top Issues, Worst Aspect, Recommendations, Executive Summary
- Generate and download full structured business report
- Download full chat history

---

## ⚡ Performance Optimizations

- `@st.cache_resource` — models load once, stay in memory
- **Batched RoBERTa** — all reviews processed in batches of 32
- **Batched ABSA** — all aspect-review pairs sent in a single pipeline call
- **MPS/CUDA/CPU auto-detection** — uses Apple M1/M2 GPU automatically
- Analysis cached per file — re-upload same file = instant, no re-inference

---

## 🏗️ System Architecture

```
CSV Upload
    ↓
RoBERTa (batched sentiment classification)
    ↓
DeBERTa ABSA (batched aspect extraction)
    ↓
Filter Engine (product / sentiment / keyword / confidence / date)
    ↓
Plotly Dashboard (interactive charts)
    ↓
Groq LLaMA 3.3 (AI analyst — data-grounded answers)
```

---

## 🛠️ Tech Stack

| Layer | Tools |
|-------|-------|
| Sentiment Model | RoBERTa (HuggingFace, fine-tuned) |
| Aspect Model | DeBERTa ABSA |
| Classical Baseline | SVM + TF-IDF (scikit-learn) |
| Deep Learning | PyTorch, HuggingFace Transformers |
| AI Analyst | LLaMA 3.3 70B via Groq API |
| Dashboard | Streamlit |
| Charts | Plotly, Matplotlib, WordCloud |
| Fonts | Lora (serif) + DM Sans |

---

## ▶️ Run Locally

```bash
pip install -r requirements.txt
streamlit run roberta_app.py
```

Add your Groq API key to `.streamlit/secrets.toml`:
```toml
GROQ_API_KEY = "your_key_here"
```

Get a free key at **console.groq.com** — no credit card required.

---

## 📁 CSV Format

Required column:
- `review`

Optional columns:
- `product` — enables product filter
- `date` — enables trend charts and date range filter
- `rating` — can be used for correlation analysis

---

## 🗂️ Project Structure

```
ai-market-research-dashboard/
├── roberta_app.py        # Main dashboard (production)
├── bert_app.py           # BERT dashboard
├── svm_app.py            # SVM dashboard
├── requirements.txt
├── sample_reviews.csv    # Example dataset
├── notebooks/            # Training notebooks
├── README.md
├── LICENSE
└── .gitignore
```

---

## 🎯 Use Cases

- Product feedback analysis
- Market research at scale
- Customer satisfaction tracking
- Competitive intelligence
- Business decision support

---

## 👨‍💻 Author

**Parv Jain**

Built as a full-stack applied NLP system covering data preprocessing, model training, imbalance handling, transformer fine-tuning, batched inference optimization, interactive analytics, and AI-driven insights.
