# KYC Risk Classification using LLaMA and BERT

## Overview
This project develops an automated KYC (Know Your Customer) risk classification system using transformer-based language models like **LLaMA** and **BERT** to streamline the onboarding process for financial services.

## Key Features

- 🔍 **Automated Risk Classification**  
  Classifies onboarding applications into `Approved`, `Review Required`, or `Rejected` based on customer data using NLP models.

- 🌐 **Web Scraping Integration**  
  Augments internal KYC data with publicly available sources (e.g., watchlists, news, company data) for enhanced risk analysis.

- 📊 **Custom Model Training**  
  Fine-tuned LLaMA and BERT models on proprietary KYC datasets using Jupyter notebooks and Python scripts.

- ☁️ **Hugging Face Deployment**  
  Hosted trained models on Hugging Face for real-time and scalable inference via API endpoints.

- ⚙️ **Airflow Workflow Automation**  
  Designed and scheduled an Apache Airflow DAG to automate the model training, validation, and deployment pipeline.

## Tech Stack

- Python, PyTorch, Transformers (Hugging Face)
- BERT, LLaMA (Meta)
- Jupyter Notebook, Pandas, BeautifulSoup, Requests
- Apache Airflow for orchestration
- Hugging Face Hub for model deployment

## Getting Started

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/kyc_risk_classification_llama.git
   cd kyc_risk_classification_llama
