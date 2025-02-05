# kyc_risk_classification_llama
# **KYC Risk Classification using LLaMA or BERT Model**

This repository contains code for **KYC risk classification** using **fine-tuned LLaMA or BERT models**. The project automates KYC application risk evaluation by analyzing customer information and web-scraped public data. The model classifies applications into risk categories like `Approved`, `Review Required`, or `Rejected`.

---

## **Features**
- Fine-tune a **LLaMA model (if access is granted)** or use **BERT (public model)** for classification.
- **Web scraping module** to fetch and analyze publicly available information about businesses.
- **Airflow DAG integration** to automatically detect new KYC applications, process them, and generate classifications.
- **Slack notifications** to alert the team of KYC application decisions.

---

## **System Requirements**
- Python 3.7+
- Hugging Face Transformers
- Hugging Face Hub access
- Airflow for workflow automation

---

## **Project Setup**

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/kyc-risk-classification.git
   cd kyc-risk-classification



