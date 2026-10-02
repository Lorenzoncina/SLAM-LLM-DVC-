import json
import os
import librosa
import soundfile as sf
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, Shift, ClippingDistortion, ApplyImpulseResponse


# Define the augmentation pipeline
augment = Compose([
    AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
    TimeStretch(min_rate=0.9, max_rate=1.1, p=0.3),
    PitchShift(min_semitones=-3, max_semitones=3, p=0.4),
    ClippingDistortion(p=0.2),
    Shift(min_shift=-0.1, max_shift=0.1, p=0.3)
])

def process_and_augment_jsonl(input_jsonl, output_jsonl, output_audio_dir):

    os.makedirs(output_audio_dir, exist_ok=True)

    with open(input_jsonl, 'r', encoding='utf-8') as infile, \
         open(output_jsonl, 'w', encoding='utf-8') as outfile:

        for line_number, line in enumerate(infile, 1):
            try:
                data = json.loads(line)
                original_key = data.get("key")
                original_path = data["source"]
                transcription = data.get("target")

                #apply the augmentation just for the following set of languages
                #if "Japanese" in original_key or "Portuguese" in original_key or "Thai" in original_key:

                    #load the audio
                y, sr = librosa.load(original_path)

                    # Apply augmentation
                y_augmented = augment(samples=y, sample_rate=sr)

                    # Generate new file name and path
                base_filename = os.path.basename(original_path)
                new_filename = f"aug_{base_filename}"
                new_path = os.path.join(output_audio_dir, new_filename)

                    #   generate new key
                new_key = f"aug_{original_key}"

                    # Save augmented audio
                sf.write(new_path, y_augmented, sr)

                    #write on the output jsonl both the input line and the augmented ones ( I want both in the new train set)
                json.dump(data, outfile, ensure_ascii=False)
                outfile.write('\n')
                    #update the data with augmented ones
                data ["key"] = new_key
                data["source"] = new_path
                    #add the augmented entry
                json.dump(data, outfile, ensure_ascii=False)
                outfile.write('\n')

                print(f"[{line_number}] Augmented and saved: {new_path}")
                #else:
                #    #write on the output jsonljust the input line which has not been augmented
                #    json.dump(data, outfile, ensure_ascii=False)
                #    outfile.write('\n')

            except Exception as e:
                print(f"Error on line {line_number}: {e}")

# Example usage
if __name__ == "__main__":
    input_jsonl = "/stek/lconcina/SLAM-LLM-DVC-/data/mlc-slm-data/full_dataset/mlc-slm-train.jsonl"  # path to the jsonl
    output_jsonl = "/stek/lconcina/SLAM-LLM-DVC-/data/mlc-slm-data/full_dataset/augmented_mlc-slm-train.jsonl"  # Output JSONL file
    output_audio_dir = "/stek/corpora/mlc-slm-data/train_data_augmented"  # Folder to store augmented audio files
    process_and_augment_jsonl(input_jsonl, output_jsonl, output_audio_dir)
