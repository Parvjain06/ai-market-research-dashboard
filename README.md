📊 Market Research Dashboard with ABSA + SVM + BERT + RoBERTa

An end-to-end NLP system for large-scale sentiment analysis and market intelligence built using classical ML, transformers, aspect-based sentiment analysis, and LLM-powered insights.
This project analyzes customer reviews, identifies sentiment, extracts aspect-level opinions, and generates business recommendations through an interactive Streamlit dashboard.


🚀 Project Overview

This system was designed to solve a real industry problem:
How can businesses automatically understand customer feedback at scale and extract actionable insights?

The solution combines:
	•	Classical Machine Learning
	•	Transformer-based Deep Learning
	•	Class imbalance handling
	•	Aspect-Based Sentiment Analysis (ABSA)
	•	AI-powered reporting
	•	Interactive analytics dashboard


🧠 Models Implemented

All models were trained on the same dataset split for fair comparison.

1️⃣ SVM (Class Weighted – Baseline)
	•	TF-IDF vectorization
	•	GridSearchCV tuning
	•	Class weights used to handle imbalance
  
Performance:
	•	Accuracy: 82.51%
	•	Weighted F1: 79.48%
	•	Macro F1: 55.23%


2️⃣ BERT (Class Weighted)
	•	Transformer fine-tuning
	•	Weighted cross-entropy loss
	•	Early stopping

Performance:
	•	Accuracy: 83.21%
	•	Weighted F1: 84.32%
	•	Macro F1: 66.53%


3️⃣ RoBERTa (Final Production Model)
	•	Class-weighted training
	•	5–6 epochs + early stopping
	•	Best handling of contextual sentiment

Performance:
	•	Accuracy: 87.63%
	•	Weighted F1: 88.03%
	•	Macro F1: 71.66%

RoBERTa significantly outperformed both SVM and BERT and is used as the primary prediction model in the dashboard.


⚖️ Class Imbalance Strategy

Dataset distribution:
	•	Positive → dominant class
	•	Negative → medium
	•	Neutral → minority and hardest to predict

Techniques explored:
	•	Random undersampling
	•	SMOTE
	•	Class weights (final choice)

Final decision: Class-weighted training performed best and preserved data volume.


🔍 Aspect-Based Sentiment Analysis (ABSA)

We integrated a DeBERTa ABSA model to extract sentiment for key business aspects:
	•	Price
	•	Quality
	•	Delivery
	•	Service
	•	Packaging
	•	Quantity

This enables insights like:
	•	“Customers like product quality but complain about delivery”
	•	“Pricing dissatisfaction increasing over time”


📈 Dashboard Features (Streamlit)

📂 Upload & Analyze Reviews
	•	Upload CSV with reviews
	•	Automatic sentiment prediction using RoBERTa
	•	Confidence scoring
	•	Aspect-level sentiment extraction

📊 Visual Analytics
	•	Sentiment distribution pie chart
	•	Aspect sentiment bar chart
	•	Time-based sentiment trend
	•	Negative review word cloud

🔎 Smart Filtering
	•	Product filter
	•	Sentiment filter
	•	Date range filter
	•	Confidence threshold
	•	Keyword search
	•	Review length filter

📄 Data Tables
	•	Filtered insights table
	•	Exportable CSV reports

🤖 AI Assistant (LLM Integration)
	•	Ask questions about customer feedback
	•	Generate executive reports
	•	Identify key problems
	•	Suggest business improvements


🏗️ System Architecture

User Reviews CSV
        ↓
RoBERTa Sentiment Model
        ↓
ABSA Aspect Model
        ↓
Filtered Analytics Engine
        ↓
Streamlit Dashboard
        ↓
LLM Insight Generator


📦 Models Hosted on HuggingFace

To avoid large GitHub file limits, trained models are hosted on HuggingFace.

RoBERTa:
parvj-06/roberta-sentiment-classweighted

Model link:
https://huggingface.co/parvj-06/roberta-sentiment-classweighted

The dashboard auto-downloads and caches the model at runtime.


🗂️ Project Structure
ai-market-research-dashboard/

├── roberta_app.py
├── bert_app.py
├── svm_app.py
├── requirements.txt
├── sample_reviews.csv
├── README.md
├── LICENSE
└── .gitignore


🛠️ Tech Stack

Machine Learning
	•	Scikit-learn
	•	SVM
	•	TF-IDF

Deep Learning
	•	PyTorch
	•	HuggingFace Transformers
	•	BERT
	•	RoBERTa

NLP Enhancements
	•	Aspect-Based Sentiment Analysis (DeBERTa)
	•	Class-weighted loss optimization

Visualization
	•	Streamlit
	•	Matplotlib
	•	Seaborn
	•	WordCloud

AI Integration
	•	LLaMA via Ollama
	•	Business insight generation


📊 Key Insights from Experiments
	•	RoBERTa captures context better than BERT and SVM.
	•	Class weighting significantly improved minority class performance.
	•	Neutral sentiment remains the hardest to classify.
	•	Transformer models outperform classical ML on nuanced language.
	•	SVM remains a strong baseline for structured sentiment signals.


▶️ How to Run Locally

1) Install dependencies
pip install -r requirements.txt

2) Run the dashboard
streamlit run roberta_app.py


📁 CSV Format Required

Must contain:
review

Optional:
product
date
rating


🎯 Use Cases
	•	Market research
	•	Product feedback analysis
	•	Customer satisfaction tracking
	•	Competitive intelligence
	•	Business decision support


📌 Future Improvements
	•	Deploy dashboard online
	•	Add live review scraping
	•	Improve neutral sentiment detection
	•	Add misclassification explorer
	•	Convert models into REST APIs
	•	Real-time sentiment monitoring


👨‍💻 Parv Jain

Built as an end-to-end applied NLP project covering:
	•	Data preprocessing
	•	Model training
	•	Imbalance handling
	•	Deep learning fine-tuning
	•	Interactive analytics
	•	AI-driven insights

This project demonstrates full-stack data science capability from modeling to deployment.


⭐ Final Result

A production-ready sentiment intelligence platform that combines:
	•	Classical ML
	•	Transformers
	•	ABSA
	•	LLM reasoning
	•	Interactive dashboards

Designed to simulate real-world industry sentiment analysis systems.
