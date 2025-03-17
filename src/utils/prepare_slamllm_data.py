"""
date: 6/3/2025 
Author: Lorenzo Concina

This script prepare a speech Massive dataset split into the format expected by the asr fine-tune recipe of SLAM. For example:
{"key": "3454", "source": "/raid/home/stek/corpora/Speech-MASSIVE/train_dev/fr-FR/audio/dev/cc6cf28a213eca0969c1907b0e3d16d8.wav", "target": "Transcript: va te coucher. Intent class: audio"}

To change task, modify the label construction here

"""

import os
import json
import argparse
import pandas as pd
from pathlib import Path
from datasets import Dataset
from load_params import load_params



def convert_parquet_to_jsonl(params):
    """Convert a Parquet dataset to a JSONL file with a specific structure."""

    lang = params.prepare.lang
    base_dir = params.prepare.base_dir
    train_split = params.prepare.train_split   
    json_slam_files = params.prepare.json_slam_files

    processed_data_dir = Path(json_slam_files)
    processed_data_dir.mkdir(exist_ok=True)

    splits = [train_split, "dev", "test"]

    for split in splits:
        if split == "train" or split == "train-115" or split == "dev":
            hf_split = "train_dev"
        elif split == "test":
            hf_split = "test"
        
        #build the abs path of audio data
        split_dir = os.path.join(base_dir, hf_split, lang, "audio")
        
        #parquet file
        parquet_file = f"data/speech_massive_data/hf_parquet_data/speech_massive_{lang}_{split}.parquet"

        #output file
        output_jsonl = f"data/speech_massive_data/slamllm_json_data/speech_massive_{lang}_{split}.jsonl"

        # Ensure file exists
        if not os.path.exists(parquet_file):
            print(f"Error: File not found at {parquet_file}")
            return
    
        # Load dataset
        dataset = Dataset.from_parquet(parquet_file)
        df = dataset.to_pandas()

        # Check if required columns exist
        required_columns = {"id", "path", "utt", "scenario_str"}
        missing_columns = required_columns - set(df.columns)
        if missing_columns:
            print(f"Error: Missing columns in dataset: {missing_columns}")
            return
    
        # Convert "path" to absolute paths
        df["absolute_path"] = df["path"].apply(lambda x: os.path.abspath(os.path.join(split_dir, x)))

        # Write to JSONL file
        #TODO: move the choise of attributes to be added to the label in the params.yaml
        with open(output_jsonl, "w", encoding="utf-8") as f:
            for _, row in df.iterrows():
                entry = {
                    "key": row["id"],
                    "source": row["absolute_path"],  # Keeping the original path as is
                    "target": f"Transcript: {row['utt']}. Intent class: {row['scenario_str']}"
                    #"target": f"{row['utt']}"  For normal SLAM ASR recipe
                }
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        print(f"✅ JSONL file of {split} split saved at: {output_jsonl}")
    

if __name__ == "__main__":

    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()

    #load parameters from params.yaml file
    params = load_params(params_path=args.config)

    convert_parquet_to_jsonl(params)

    