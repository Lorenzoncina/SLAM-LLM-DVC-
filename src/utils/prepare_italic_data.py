"""
date: 31/3/2025 
Author: Lorenzo Concina

This script prepare the italic dataset into the format expected by the asr fine-tune recipe of SLAM. For example:
{"key": "3454", "source": "/raid/home/stek/corpora/Speech-MASSIVE/train_dev/fr-FR/audio/dev/cc6cf28a213eca0969c1907b0e3d16d8.wav", "target": "Transcript: va te coucher. Intent class: audio"}

To change task, modify the label construction here

"""

import json
import os
import argparse

def process_json_files(train_file, val_file, test_file, audio_folder, output_folder):
    datasets = {"train": train_file, "val": val_file, "test": test_file}
    
    for split, file_path in datasets.items():
        output_file = os.path.join(output_folder, f"{split}_italic_IC.jsonl")
        
        with open(file_path, "r", encoding="utf-8") as f:
            #data = json.load(f)
            data = [json.loads(line) for line in f]
        
        with open(output_file, "w", encoding="utf-8") as out_f:
            for sample in data:
                speaker_id = sample["id"]
                audio_path = os.path.join(audio_folder, f"{speaker_id}.wav")
                output_entry = {
                    "key": str(speaker_id),
                    "source": audio_path,
                    "target": f"Transcript: {sample['utt']}.  Intent class: {sample['intent']}. " #ASR IC   
                    #"target": f"Intent class: {sample['intent']}" #Intent 
                }
                out_f.write(json.dumps(output_entry, ensure_ascii=False) + "\n")
        
        print(f"Processed {split} set: {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process JSON files and generate JSONL outputs.")
    parser.add_argument("--train_file", type=str, default="/stek/corpora/italic/zenodo_dataset/massive_train.json", help="Path to the training JSON file.")
    parser.add_argument("--val_file", type=str, default="/stek/corpora/italic/zenodo_dataset/massive_validation.json", help="Path to the validation JSON file.")
    parser.add_argument("--test_file", type=str, default="/stek/corpora/italic/zenodo_dataset/massive_test.json", help="Path to the test JSON file.")
    parser.add_argument("--audio_folder", type=str, default="/stek/corpora/italic/zenodo_dataset/recordings", help="Path to the folder containing audio files.")
    parser.add_argument("--output_folder", type=str, default="/stek/lconcina/SLAM-LLM-DVC-/data/speech_massive_data/slamllm_json_data", help="Path to the folder for output JSONL files.")
    
    args = parser.parse_args()
    #os.makedirs(args.output_folder, exist_ok=True)
    process_json_files(args.train_file, args.val_file, args.test_file, args.audio_folder, args.output_folder)
