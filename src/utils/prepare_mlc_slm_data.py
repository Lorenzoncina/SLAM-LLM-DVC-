"""
date: 11/4/2025 
Author: Lorenzo Concina

This script prepare the data from the mlc slm challenge, by adding the absolute path to each line of the jsonl file 

"""

import json
from pathlib import Path
import argparse

#specify here the base file path to audio data (change for train and dev)
base_path = Path("/stek/corpora/mlc-slm-data/train_data/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process JSON files and generate JSONL outputs.")
    parser.add_argument("--input_file", type=str, default="/stek/corpora/mlc-slm-data/train_data_italian/raw_data.list", help="Path to input jsonl file")
    parser.add_argument("--output_file", type=str, default="/stek/lconcina/SLAM-LLM-DVC-/data/mlc-slm-data/italian_data/mlc-slm-italian-train.jsonl", help="Path to output json file")
    args = parser.parse_args()
    input_file = args.input_file
    output_file = args.output_file

    with open(input_file, "r", encoding="utf-8") as infile, open(output_file, "w", encoding="utf-8") as outfile:
        for line in infile:
            data = json.loads(line)
            relative_source = Path(data["source"])
            # Replace only the 'train_data' prefix with the desired absolute path
            new_source = base_path / Path(*relative_source.parts[1:])
            data["source"] = str(new_source)
            outfile.write(json.dumps(data, ensure_ascii=False) + "\n")
