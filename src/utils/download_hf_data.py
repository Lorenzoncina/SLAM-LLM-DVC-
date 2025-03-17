"""
date: 6/3/2025 
Author: Lorenzo Concina

This script download a specific split of Speech Massive dataset from HF and save it locally as a parquet file.
Provided the lang and split wanted
"""

from datasets import load_dataset
import os

def download_speech_massive(save_dir, split):
    """Download the 'fr-FR' subset of the 'train' split from FBK-MT/Speech-MASSIVE."""
    
    # Load the dataset
    dataset = load_dataset("FBK-MT/Speech-MASSIVE", "fr-FR", split=split)
    
    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok=True)
    
    # Save dataset as a CSV or JSON file
    if split == "validation":
        split = "dev"
    dataset_name = f"speech_massive_fr-FR_{split}.parquet"
    dataset_path = os.path.join(save_dir, dataset_name)
    #dataset.to_json(dataset_path)
    dataset.to_parquet(dataset_path)

    print(f"Dataset saved to {dataset_path}")

if __name__ == "__main__":
    save_directory = "data/speech_massive_data/hf_parquet_data"
    #split can be 'train', 'validation', 'test'
    split = "validation"
    download_speech_massive(save_directory, split)
