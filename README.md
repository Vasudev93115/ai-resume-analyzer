# AI Resume Analyzer

An AI-powered Resume Analyzer built using Streamlit, NLP, Sentence Transformers, FAISS, and Hugging Face Transformers.

The application analyzes uploaded resumes, extracts skills, recommends matching jobs, calculates ATS compatibility scores, identifies missing skills, and generates AI-based resume feedback.

---

# Features

- Resume PDF Upload
- Skill Extraction
- AI Job Recommendation System
- ATS Compatibility Score
- Missing Skill Detection
- AI Resume Feedback
- Downloadable Analysis Report
- Real-time Job Data using Adzuna API

---

# Tech Stack

- Python
- Streamlit
- Sentence Transformers
- FAISS
- Hugging Face Transformers
- Pandas
- NumPy
- PyPDF
- Adzuna Job API

---

# Project Structure

```bash
AI-RESUME/
│
├── .streamlit/
│   └── config.toml
│
├── app.py
├── scraper.py
├── jobs.csv
├── skills.txt
├── requirements.txt
├── README.md
└── .gitignore
```

---

# Installation

## Clone Repository

```bash
git clone https://github.com/YOUR_USERNAME/ai-resume-analyzer.git
```

## Navigate to Project

```bash
cd ai-resume-analyzer
```

## Create Virtual Environment

### Windows

```bash
python -m venv venv
```

### Activate Environment

```bash
venv\Scripts\activate
```

---

# Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Run Job Scraper

```bash
python scraper.py
```

This generates:

```bash
jobs.csv
```

---

# Run Streamlit App

```bash
streamlit run app.py
```

---

# API Used

- Adzuna Job Search API

---

# Future Improvements

- Resume Ranking System
- Multi-resume comparison
- Interview Question Generator
- LinkedIn Profile Analysis
- Cloud Deployment
- Real-time Skill Gap Analysis

---

# Author

Vasudev Giri
