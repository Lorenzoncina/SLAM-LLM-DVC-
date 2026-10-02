"""Diagnostic: strip over-generation / hallucination tails from decoded hypotheses.

The SVR-gated model transcribes correctly but often fails to emit EOS, so it
appends filler symbols and hallucinated continuations until max_new_tokens.
That inflates WER even though the leading transcription is correct. This script
truncates each hypothesis so we can quantify the *content* WER (how good the
retention actually is), separate from the stopping problem.

This is a DIAGNOSTIC, not a real decoding fix. With --ref it additionally caps
each hypothesis at a multiple of the reference length, which uses oracle info.

Per hypothesis, two truncations:
  1. cut at the first newline or first "anomalous" character -- a symbol/marker
     that never appears in a normal transcription (e.g. the garbage runs
     ``..fuoco nemico♀♀♀`` or ``... certainty ….the only thing``).
  2. (only if --ref is given) cap to <= length_factor * reference_word_count
     words, catching plain-text hallucinated continuations with no marker.

Files are tab-separated ``utt_id<TAB>text`` (SLAM-LLM decode format).

Usage:
  python src/utils/strip_hallucinations.py --hyp PRED --out PRED.clean \
      [--ref GT] [--length-factor 2.0]
"""
import argparse
import unicodedata

# punctuation that may legitimately appear inside a transcription
ALLOWED_PUNCT = set(".,!?;:'\"()-’‘–—% ")


def is_normal_char(ch):
    """True for letters (incl. accents/Cyrillic), digits, spaces, basic punct."""
    if ch in ("\n", "\r", "\t"):
        return False
    if ch.isspace():
        return True
    if ch in ALLOWED_PUNCT:
        return True
    cat = unicodedata.category(ch)  # L*=letter, M*=combining mark, N*=number
    return cat[0] in ("L", "M", "N")


def truncate_at_garbage(text):
    """Keep the leading run of normal-transcription characters."""
    for i, ch in enumerate(text):
        if not is_normal_char(ch):
            return text[:i].rstrip()
    return text.rstrip()


def cap_to_reference(text, ref_text, factor):
    max_words = int(len(ref_text.split()) * factor) + 5
    words = text.split()
    return " ".join(words[:max_words]) if len(words) > max_words else text


def read_tsv(path):
    rows = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t", 1)
            uid = parts[0]
            rows[uid] = parts[1] if len(parts) == 2 else ""
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--hyp", required=True, help="decoded prediction file (utt_id<TAB>text)")
    p.add_argument("--out", required=True, help="output path for the cleaned prediction file")
    p.add_argument("--ref", default=None, help="ground-truth file; enables length capping")
    p.add_argument("--length-factor", type=float, default=2.0,
                   help="cap hypothesis to factor * reference words (default 2.0)")
    args = p.parse_args()

    hyp = read_tsv(args.hyp)
    ref = read_tsv(args.ref) if args.ref else {}

    n_sym, n_cap = 0, 0
    before_words, after_words = 0, 0
    with open(args.out, "w", encoding="utf-8") as f:
        for uid, text in hyp.items():
            nb = len(text.split())
            before_words += nb
            cleaned = truncate_at_garbage(text)
            if len(cleaned.split()) < nb:
                n_sym += 1
            if uid in ref:
                capped = cap_to_reference(cleaned, ref[uid], args.length_factor)
                if len(capped.split()) < len(cleaned.split()):
                    n_cap += 1
                cleaned = capped
            after_words += len(cleaned.split())
            f.write(f"{uid}\t{cleaned}\n")

    n = max(len(hyp), 1)
    print(f"utterances: {len(hyp)}")
    print(f"truncated at a garbage symbol/newline: {n_sym} ({100*n_sym/n:.1f}%)")
    if args.ref:
        print(f"further capped by reference length:    {n_cap} ({100*n_cap/n:.1f}%)")
    print(f"avg words/hyp: {before_words/n:.1f} -> {after_words/n:.1f}")
    print(f"\nSaved cleaned hypotheses to: {args.out}")
    print("Re-run your WER eval pointing prediction_file at this cleaned file.")


if __name__ == "__main__":
    main()
