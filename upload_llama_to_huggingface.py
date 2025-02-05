
from huggingface_hub import upload_folder, HfApi

# Define your Hugging Face repository details
repo_name = "your-huggingface-username/llama-kyc-classifier"

# Create a new repository
api = HfApi()
api.create_repo(repo_id=repo_name, private=False)

# Upload the fine-tuned model
upload_folder(repo_id=repo_name, folder_path="./llama_kyc_model")

print(f"Model successfully uploaded to Hugging Face: https://huggingface.co/{repo_name}")
