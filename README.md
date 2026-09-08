# 🔑 AI-Powered Keyword Extraction Web App

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://keywordextraction-j5c5aityzyfdfbdlvssxwx.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.x-blue?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-TF--IDF-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)

An end-to-end Machine Learning web application that extracts key terms and concepts from documents in **TXT, CSV, PDF, and DOCX** formats, powered by a pre-trained **TF-IDF (Term Frequency–Inverse Document Frequency)** model.

**🔗 [Try the Live App](https://keywordextraction-j5c5aityzyfdfbdlvssxwx.streamlit.app/)**

---

## 📑 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [Project Architecture & Workflow](#️-project-architecture--workflow)
- [Tech Stack](#-tech-stack)
- [Repository Structure](#-repository-structure)
- [Getting Started](#-getting-started)
- [Usage](#-usage)
- [Author](#-author)

---

## 📖 Overview

This project implements a complete Machine Learning lifecycle — from model training and data cleaning to deployment as an interactive web application. It's built to help users quickly surface the most relevant terms in unstructured text, whether pasted manually or extracted straight from a document.

## ✨ Key Features

- **📝 Multi-Format Support** — extract keywords from raw pasted text, PDF, DOCX, or CSV files
- **📊 Batch Analysis** — for CSV datasets, pick specific rows/columns to analyze in bulk
- **🧹 Advanced Text Preprocessing** — custom NLTK pipeline (tokenization, stopword removal, lemmatization)
- **🏆 Top-10 Ranking** — surfaces the top 10 most relevant keywords along with their TF-IDF scores
- **🎨 Responsive UI** — clean, tab-based layout with visual score metric cards

## 🛠️ Project Architecture & Workflow

### 1. Model Training & Data Cleaning
- Developed and trained on **Kaggle** using `scikit-learn`, `pandas`, and `nltk`
- Cleaned text data: removed stop words, applied lemmatization, filtered tokens
- Trained a `TfidfVectorizer` to capture important unigrams and n-grams
- Serialized the trained model with `joblib` into `tfidf_model.pkl`

### 2. Web Application & Deployment
- Built an interactive UI with **Streamlit**
- Integrated robust text-extraction tools for unstructured data (`pdfplumber`, `python-docx`)
- Deployed and hosted live via **Streamlit Community Cloud**, connected directly to this GitHub repository

## 🧰 Tech Stack

| Category | Tools |
|---|---|
| Language | Python |
| ML / NLP | scikit-learn, NLTK, joblib |
| Web Framework | Streamlit |
| Document Parsing | pdfplumber, python-docx, pandas |
| Deployment | Streamlit Community Cloud |

## 📂 Repository Structure

```text
.
├── app.py                # Main Streamlit web application code
├── tfidf_model.pkl       # Pre-trained TF-IDF vectorizer model
├── requirements.txt      # Python dependencies for deployment
└── README.md             # Project documentation
```

## 🚀 Getting Started

### Prerequisites
- Python 3.9+
- pip

### Installation

```bash
# Clone the repository
git clone https://github.com/yassersheta-cmyk/<your-repo-name>.git
cd <your-repo-name>

# Install dependencies
pip install -r requirements.txt
```

### Run Locally

```bash
streamlit run app.py
```

The app will open automatically in your browser at `http://localhost:8501`.

## 💡 Usage

1. Choose an input method — paste raw text, or upload a **PDF**, **DOCX**, or **CSV** file
2. For CSV files, select the row/column you want analyzed
3. Run the extraction and view the top 10 keywords ranked by TF-IDF score

## 👨‍💻 Author

**Yasser Sheta**
GitHub: [@yassersheta-cmyk](https://github.com/yassersheta-cmyk)

---

⭐ If you found this project useful, consider giving it a star on GitHub!
