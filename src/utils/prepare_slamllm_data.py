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
    #Convert a Parquet dataset to a merged JSONL file per split with all languages.

    langs = params.prepare.lang
    if isinstance(langs, str):
        langs = [langs]

    base_dir = params.prepare.base_dir
    train_split = params.prepare.train_split
    json_slam_files = params.prepare.json_slam_files
    task = params.prepare.task

    processed_data_dir = Path(json_slam_files)
    processed_data_dir.mkdir(exist_ok=True)

    splits = [train_split, "dev", "test"]

    for split in splits:
        if split in ["train", "train-115", "dev"]:
            hf_split = "train_dev"
        else:
            hf_split = "test"

        # Output file for the split (all langs combined)
        lang_suffix = "_".join(lang.replace("-", "") for lang in langs)  # Remove dashes if needed
        output_jsonl = os.path.join(json_slam_files, f"speech_massive_{lang_suffix}_{split}_{task}.jsonl")


        with open(output_jsonl, "w", encoding="utf-8") as f_out:
            for lang in langs:
                split_dir = os.path.join(base_dir, hf_split, lang, "audio")
                parquet_file = f"data/speech_massive_data/hf_parquet_data/speech_massive_{lang}_{split}.parquet"

                if not os.path.exists(parquet_file):
                    print(f"⚠️ Warning: File not found at {parquet_file}, skipping.")
                    continue

                dataset = Dataset.from_parquet(parquet_file)
                df = dataset.to_pandas()

                required_columns = {"id", "path", "utt", "scenario_str"}
                missing_columns = required_columns - set(df.columns)
                if missing_columns:
                    print(f"⚠️ Warning: Missing columns in dataset {lang}-{split}: {missing_columns}, skipping.")
                    continue

                df["absolute_path"] = df["path"].apply(lambda x: os.path.abspath(os.path.join(split_dir, x)))

                for _, row in df.iterrows():
                    entry = {
                        "key": row["id"],
                        "source": row["absolute_path"],
                        #"target": f"Transcript: {row['utt']}. Intent class: {row['scenario_str']}. Annotated utterance: {row['annot_utt']}"  #ASR + Intent Classification + Slot Filling Tasks
                        "target": f"Transcript: {row['utt']}. Intent class: {row['intent_str']}."
                        #"target": f"Intent class: {row['intent_str']}" #IC only
                    }
                    f_out.write(json.dumps(entry, ensure_ascii=False) + "\n")

        print(f"✅ Merged JSONL file for {split} split saved at: {output_jsonl}")


if __name__ == "__main__":

    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()

    #load parameters from params.yaml file
    params = load_params(params_path=args.config)

    convert_parquet_to_jsonl(params)

    
