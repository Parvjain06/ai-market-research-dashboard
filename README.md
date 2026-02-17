📊 Market Research Dashboard with ABSA + SVM + BERT + RoBERTa

🚀 Production-ready NLP system for sentiment intelligence using SVM, BERT, RoBERTa + ABSA + LLM insights.

An end-to-end NLP system for large-scale sentiment analysis and market intelligence built using classical ML, transformer models, aspect-based sentiment analysis, and LLM-powered insights.

This project analyzes customer reviews, detects sentiment, extracts aspect-level opinions, and generates business recommendations through an interactive Streamlit dashboard.

---

🚀 Project Overview

This system was designed to solve a real industry problem:

How can businesses automatically understand customer feedback at scale and extract actionable insights?

The solution combines:

	-	Classical Machine Learning
	-	Transformer-based Deep Learning
	-	Class imbalance handling
	-	Aspect-Based Sentiment Analysis (ABSA)
	-	AI-powered insight generation
	-	Interactive analytics dashboard

---

🧠 Models Implemented

All models were trained on the same dataset split for fair comparison.

1️⃣ SVM (Class-Weighted Baseline)

	-	TF-IDF vectorization
	-	GridSearchCV hyperparameter tuning
	-	Class weights to handle imbalance

Performance

	-	Accuracy: 82.51%
	-	Weighted F1: 79.48%
	-	Macro F1: 55.23%

---

2️⃣ BERT (Class-Weighted)

	-	Transformer fine-tuning
	-	Weighted cross-entropy loss
	-	Early stopping

Performance

	-	Accuracy: 83.21%
	-	Weighted F1: 84.32%
	-	Macro F1: 66.53%

---

3️⃣ RoBERTa (Final Production Model)

	-	Class-weighted training
	-	5–6 epochs + early stopping
	-	Strong contextual sentiment understanding

Performance

	-	Accuracy: 87.63%
	-	Weighted F1: 88.03%
	-	Macro F1: 71.66%

RoBERTa improved accuracy by +5% over SVM and +4% over BERT.

---

| Model    | Accuracy | Weighted F1 | Macro F1 |
|----------|----------|-------------|----------|
| SVM      | 82.51%   | 79.48%      | 55.23%   |
| BERT     | 83.21%   | 84.32%      | 66.53%   |
| RoBERTa  | 87.63%   | 88.03%      | 71.66%   |


---

⚖️ Class Imbalance Strategy

Dataset distribution:

	-	Positive → dominant class
	-	Negative → medium
	-	Neutral → minority (hardest class)

Techniques explored:

	-	Random undersampling
	-	SMOTE
	-	Class weights (final choice)

Final decision:
Class-weighted training preserved data volume and delivered the best performance.

---

🔍 Aspect-Based Sentiment Analysis (ABSA)

Integrated DeBERTa ABSA model to extract sentiment across business-critical aspects:

	-	Price
	-	Quality
	-	Delivery
	-	Service
	-	Packaging
	-	Quantity

Enables insights like:

	-	“Customers like product quality but complain about delivery”
	-	“Pricing dissatisfaction increasing over time”

---

📈 Dashboard Features (Streamlit)

📂 Upload & Analyze Reviews

	-	Upload CSV with reviews
	-	Automatic sentiment prediction (RoBERTa)
	-	Confidence scoring
	-	Aspect-level sentiment extraction

📊 Visual Analytics

	-	Sentiment distribution pie chart
	-	Aspect sentiment bar chart
	-	Time-based sentiment trend
	-	Negative review word cloud

🔎 Smart Filtering

	-	Product filter
	-	Sentiment filter
	-	Date range filter
	-	Confidence threshold
	-	Keyword search
	-	Review length filter

📄 Data Tables

	-	Filtered insights table
	-	Exportable CSV reports

🤖 AI Assistant (LLM Integration)

	-	Ask questions about customer feedback
	-	Generate executive reports
	-	Identify key problems
	-	Suggest business improvements

---

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


---

📦 Models Hosted on HuggingFace

To avoid GitHub file size limits, trained models are hosted externally.

RoBERTa Model:
https://huggingface.co/parvj-06/roberta-sentiment-classweighted

The dashboard auto-downloads and caches the model at runtime.

---

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


---

🛠️ Tech Stack

Machine Learning

	-	Scikit-learn
	-	SVM
	-	TF-IDF

Deep Learning

	-	PyTorch
	-	HuggingFace Transformers
	-	BERT
	-	RoBERTa

NLP Enhancements

	-	Aspect-Based Sentiment Analysis (DeBERTa)
	-	Class-weighted loss optimization

Visualization

	-	Streamlit
	-	Matplotlib
	-	Seaborn
	-	WordCloud

AI Integration

	-	LLaMA via Ollama
	-	Automated business insight generation

---

📊 Key Insights from Experiments

	-	RoBERTa captures context better than BERT and SVM
	-	Class weighting significantly improved minority class performance
	-	Neutral sentiment remains the hardest class
	-	Transformer models outperform classical ML on nuanced language
	-	SVM remains a strong baseline

---

▶️ How to Run Locally

Install dependencies:

pip install -r requirements.txt

Run dashboard:

streamlit run roberta_app.py


---

📁 CSV Format Required

Must contain:

	-	review

Optional:

	-	product
	-	date
	-	rating

---

🎯 Use Cases

	-	Market research
	-	Product feedback analysis
	-	Customer satisfaction tracking
	-	Competitive intelligence
	-	Business decision support

---

📌 Future Improvements

	-	Deploy dashboard online
	-	Improve neutral sentiment detection
	-	Add misclassification explorer
	-	Convert models into REST APIs
	-	Real-time sentiment monitoring

---

👨‍💻 Author

Parv Jain

Built as a full-stack applied NLP system covering:

	-	Data preprocessing
	-	Model training
	-	Imbalance handling
	-	Deep learning fine-tuning
	-	Interactive analytics
	-	AI-driven insights

---

⭐ Final Result

A production-ready sentiment intelligence platform combining:

	-	Classical ML
	-	Transformers
	-	ABSA
	-	LLM reasoning
	-	Interactive dashboards

Designed to simulate real-world industry sentiment analysis systems.
