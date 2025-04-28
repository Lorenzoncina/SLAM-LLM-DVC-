import json

def filter_jsonl_by_keyword(input_path, output_path, keyword="Russian"):
    with open(input_path, 'r', encoding='utf-8') as infile, \
         open(output_path, 'w', encoding='utf-8') as outfile:
        for line in infile:
            if keyword in line:
                try:
                    data = json.loads(line)
                    json.dump(data, outfile, ensure_ascii=False)
                    outfile.write('\n')
                except json.JSONDecodeError:
                    print(f"Skipping invalid JSON line: {line.strip()}")

# Example usage
input_file = '/stek/lconcina/SLAM-LLM-DVC-/data/mlc-slm-data/full_dataset/mlc-slm-dev.jsonl'
output_file = '/stek/lconcina/SLAM-LLM-DVC-/data/mlc-slm-data/russian_data/mlc-slm-russian-dev.jsonl'
filter_jsonl_by_keyword(input_file, output_file)
