from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import snapshot_download
import os

# Set your local folder path
local_dir = "/stek/lconcina/SLAM-LLM-DVC-/models/EuroLLM-9B"

# Download the entire model snapshot (including config, tokenizer, weights, etc.)
model_path = snapshot_download(repo_id="utter-project/EuroLLM-9B", local_dir=local_dir, local_dir_use_symlinks=False)

print(f"Model downloaded to: {model_path}")

# (Optional) Load model and tokenizer to test
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

# Quick test
input_text = "Once upon a time"
inputs = tokenizer(input_text, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=20)

print("Generated text:", tokenizer.decode(outputs[0], skip_special_tokens=True))
