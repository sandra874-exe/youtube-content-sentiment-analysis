# YouTube Content Sentiment Analysis

## Overview

A data science project analyzing sentiment and engagement in YouTube comments across three content domains:

- **Technology** — Mrwhosetheboss
- **Travel** — Ryan Trahan
- **Cats** — Abram Engle

The project uses **750 comments from 15 videos** and combines data preprocessing, exploratory analysis, VADER sentiment analysis, transformer-based sentiment analysis, model validation, and an interactive Streamlit dashboard.

## Dataset

| Domain | Videos | Comments |
|---|---:|---:|
| Technology | 5 | 250 |
| Travel | 5 | 250 |
| Cats | 5 | 250 |
| **Total** | **15** | **750** |

## Workflow

```text
YouTube Data API
       ↓
Data Collection
       ↓
Data Cleaning
       ↓
EDA
       ↓
Sentiment Analysis
       ↓
Human Validation
       ↓
Model Comparison
       ↓
Error Analysis
       ↓
Streamlit Dashboard
```

## Sentiment Analysis

Two approaches were evaluated:

- **VADER**
- **Twitter-RoBERTa** (`cardiffnlp/twitter-roberta-base-sentiment-latest`)

A manually labeled sample of **120 comments** was used for validation.

| Model | Accuracy | Macro F1 |
|---|---:|---:|
| VADER | 60.83% | 55.38% |
| Transformer | 71.67% | 69.07% |

## Project Structure

```text
youtube-content-sentiment-analysis/
├── app.py
├── config/
├── data/
├── notebooks/
├── src/
├── streamlit/
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

## Technologies

Python · Pandas · NumPy · Scikit-learn · VADER · Hugging Face Transformers · Matplotlib · Plotly · Streamlit · YouTube Data API

## Run the Dashboard

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Author

**Sandra Wilson**  
BSc Data Science
