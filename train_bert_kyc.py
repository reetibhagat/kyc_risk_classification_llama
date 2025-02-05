
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from datasets import Dataset
import pandas as pd

# === Load and prepare dataset ===
data = {
    "text": [
        "ABC Corp is a business in the healthcare sector. No issues detected.",
        "XYZ Inc operates in the cannabis sector with past lawsuits.",
        "DEF Corp in the insurance sector, low income, low credit score."
    ],
    "label": ["Approved", "Rejected", "Review Required"]
}
df = pd.DataFrame(data)

# Encode labels
label_map = {'Approved': 0, 'Review Required': 1, 'Rejected': 2}
df['label'] = df['label'].map(label_map)

# Split into training and validation sets
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df['text'].tolist(), df['label'].tolist(), test_size=0.2, random_state=42
)

# Tokenize input
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=512)
val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=512)

train_dataset = Dataset.from_dict({'input_ids': train_encodings['input_ids'], 'labels': train_labels})
val_dataset = Dataset.from_dict({'input_ids': val_encodings['input_ids'], 'labels': val_labels})

# Load BERT model
model = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)

# Training setup
training_args = TrainingArguments(
    output_dir="./bert_kyc_model",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

# Train the model
trainer.train()

# Save the model locally
model.save_pretrained('./bert_kyc_model')
tokenizer.save_pretrained('./bert_kyc_model')
