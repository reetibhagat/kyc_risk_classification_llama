from airflow import DAG
from airflow.sensors.filesystem import FileSensor
from airflow.operators.python_operator import PythonOperator
from airflow.providers.slack.operators.slack_webhook import SlackWebhookOperator
from datetime import datetime, timedelta
import pandas as pd
import torch
import requests
from bs4 import BeautifulSoup
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Slack webhook URL
slack_webhook_url = 'https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK'

# Default Airflow DAG arguments
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Define the DAG
dag = DAG(
    'bert_kyc_classification',
    default_args=default_args,
    description='KYC classification using fine-tuned BERT from Hugging Face',
    schedule_interval=None,
    start_date=datetime(2025, 2, 1),
    catchup=False,
)

# === TASK 1: File Sensor to detect new KYC applications ===
file_sensor = FileSensor(
    task_id='wait_for_new_kyc_file',
    filepath='/path/to/new_kyc_input.csv',
    poke_interval=30,
    timeout=3600,
    dag=dag
)

# === TASK 2: Web scraping for public issues ===
def scrape_websites(kyc_file='/path/to/new_kyc_input.csv'):
    kyc_data = pd.read_csv(kyc_file)
    
    def scrape_website(url):
        try:
            response = requests.get(url, timeout=5)
            soup = BeautifulSoup(response.text, 'html.parser')
            text = soup.get_text().lower()
            keywords = ['lawsuit', 'bad debt', 'marijuana', 'illegal']
            return ', '.join([kw for kw in keywords if kw in text]) or 'No issues'
        except Exception as e:
            return f"Error: {e}"

    kyc_data['website_risk'] = kyc_data['Website URL'].apply(scrape_website)
    kyc_data.to_csv('/path/to/kyc_with_website_risk.csv', index=False)

scrape_task = PythonOperator(
    task_id='scrape_websites',
    python_callable=scrape_websites,
    dag=dag
)

# === TASK 3: Classify KYC data using the pre-trained BERT model ===
def classify_kyc_data():
    # Load the BERT model from Hugging Face
    model = AutoModelForSequenceClassification.from_pretrained("your-huggingface-username/bert-kyc-classifier")
    tokenizer = AutoTokenizer.from_pretrained("your-huggingface-username/bert-kyc-classifier")

    kyc_data = pd.read_csv('/path/to/kyc_with_website_risk.csv')
    kyc_data['input_text'] = kyc_data['summary'] + " " + kyc_data['website_risk']

    # Tokenize and predict
    inputs = tokenizer(list(kyc_data['input_text']), padding=True, truncation=True, return_tensors="pt", max_length=512)
    outputs = model(**inputs)
    predictions = torch.argmax(outputs.logits, axis=1).tolist()

    # Map predictions to categories
    label_map = {0: 'Approved', 1: 'Review Required', 2: 'Rejected'}
    kyc_data['decision'] = [label_map[pred] for pred in predictions]

    # Save results
    kyc_data.to_csv('/path/to/final_kyc_results.csv', index=False)

classification_task = PythonOperator(
    task_id='classify_kyc_data',
    python_callable=classify_kyc_data,
    dag=dag
)

# === TASK 4: Send Slack notification ===
def notify_slack():
    kyc_data = pd.read_csv('/path/to/final_kyc_results.csv')
    approved = len(kyc_data[kyc_data['decision'] == 'Approved'])
    review = len(kyc_data[kyc_data['decision'] == 'Review Required'])
    rejected = len(kyc_data[kyc_data['decision'] == 'Rejected'])

    message = f"""
    *KYC Underwriting Summary:*
    ✅ Approved: {approved}
    🔍 Review Required: {review}
    ❌ Rejected: {rejected}
    📁 Results saved at `/path/to/final_kyc_results.csv`.
    """
    return message

slack_notification = SlackWebhookOperator(
    task_id='send_slack_notification',
    http_conn_id=None,
    webhook_token=slack_webhook_url,
    message="{{ task_instance.xcom_pull(task_ids='notify_slack') }}",
    dag=dag
)

# Define task dependencies
file_sensor >> scrape_task >> classification_task >> slack_notification
