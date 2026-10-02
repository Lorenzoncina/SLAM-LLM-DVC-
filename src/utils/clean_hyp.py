import argparse

def load_reference_ids(ref_path):
    ids = set()
    with open(ref_path, 'r', encoding='utf-8') as f:
        for line in f:
            utt_id, _ = line.strip().split('\t', 1)
            ids.add(utt_id)
    return ids

def load_hypothesis(hyp_path):
    hyp = {}
    with open(hyp_path, 'r', encoding='utf-8') as f:
        for line in f:
            utt_id, text = line.strip().split('\t', 1)
            hyp[utt_id] = text
    return hyp

def write_cleaned_hyp(hyp, ref_ids, output_path, fill_missing=False):
    with open(output_path, 'w', encoding='utf-8') as f:
        for utt_id in sorted(ref_ids):
            text = hyp.get(utt_id, "" if fill_missing else None)
            if text is not None:
                f.write(f"{utt_id}\t{text}\n")

def main():
    parser = argparse.ArgumentParser(description="Clean hypothesis file based on reference IDs")
    parser.add_argument("--ref", required=True, help="Path to reference .txt file")
    parser.add_argument("--hyp", required=True, help="Path to hypothesis .txt file")
    parser.add_argument("--out", required=True, help="Path to save cleaned hypothesis file")
    parser.add_argument("--fill-missing", action="store_true", help="Add empty lines for missing hypotheses")

    args = parser.parse_args()

    ref_ids = load_reference_ids(args.ref)
    hyp = load_hypothesis(args.hyp)
    write_cleaned_hyp(hyp, ref_ids, args.out, fill_missing=args.fill_missing)

if __name__ == "__main__":
    main()
