#!/usr/bin/env python3
"""
Quantify EOS collapse (unbounded generation) from SLAM-LLM decode directories.

Decode files are 'utt_id<TAB>text', named
    decode_<lang>_test_beam<k>_{gt,pred,pred_stripped}

What is measured
----------------
Plain text carries no EOS token, so "no EOS emitted" is not directly readable.
The observable signature in these decodings is: a correct transcription followed
by a degenerate continuation (filler glyphs, then repeated paraphrases). Four
per-language statistics are reported:

  collapse%  hypotheses whose raw output degenerates after the transcription,
             i.e. raw pred differs from pred_stripped (if available) or shows
             filler glyphs / n-gram loops / excessive length
  filler%    hypotheses containing glyphs that never occur in the references
  loop%      hypotheses with a degenerate repeated n-gram
  ratio      mean length of raw hypothesis / length of reference

With --wer (needs `pip install jiwer`) it also reports WER of the raw output vs
WER after truncation, which separates a termination failure from a
recognition failure.

Usage
-----
  python eos_collapse_stats.py --dir .../hallucinated_decodings --wer
  python eos_collapse_stats.py --dir .../frozen --name "LoRA frozen" \
                               --dir2 .../joint --name2 "joint gate-LoRA" --wer
"""

import argparse, csv, os, re, sys, unicodedata
from collections import Counter

# --------------------------------------------------------------------------- io

def parse(path):
    out = {}
    with open(path, encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f):
            line = line.rstrip("\n")
            if not line.strip():
                continue
            key, sep, text = line.partition("\t")
            if not sep:                      # no tab: treat whole line as text
                key, text = f"line{i}", line
            out[key.strip()] = text.strip()
    return out


def discover(directory):
    found = {}
    for fn in sorted(os.listdir(directory)):
        m = re.match(r"decode_(.+?)_test_beam\d+_(gt|pred)$", fn)
        if not m:
            continue
        found.setdefault(m.group(1), {})[m.group(2)] = os.path.join(directory, fn)
    out = []
    for lang, d in sorted(found.items()):
        if "gt" not in d or "pred" not in d:
            print(f"  [skip] {lang}: missing gt or pred", file=sys.stderr)
            continue
        out.append((lang, d["gt"], d["pred"]))
    return out

# ---------------------------------------------------------------- measurements

def filler_charset(refs, hyps, min_count=3):
    """Glyphs that appear in hypotheses but never in any reference."""
    ref_chars = set()
    for t in refs:
        ref_chars.update(t)
    c = Counter()
    for t in hyps:
        for ch in t:
            if ch not in ref_chars and not ch.isspace() and not ch.isalnum():
                cat = unicodedata.category(ch)
                # keep symbols / marks / private use, drop ordinary punctuation
                if cat[0] in ("S", "C", "M") or cat in ("Po",) and ord(ch) > 0x2000:
                    c[ch] += 1
    return {ch for ch, n in c.items() if n >= min_count}


def ngram_loop(text, n=4, times=3):
    toks = text.split()
    if len(toks) >= n * times:
        for i in range(len(toks) - n * times + 1):
            first = toks[i:i + n]
            if all(toks[i + k*n:i + (k+1)*n] == first for k in range(1, times)):
                return True
    if len(text) >= 10 * times:              # char level, for JA/TH
        for i in range(len(text) - 10 * times + 1):
            first = text[i:i + 10]
            if all(text[i + k*10:i + (k+1)*10] == first for k in range(1, times)):
                return True
    return False


def norm(s):
    return " ".join(s.lower().split())


def truncate(hyp, fillers, run=2):
    """Cut the hypothesis at the first degenerate emission.

    Rule: cut at the first position where `run` filler glyphs occur within a
    3-character window (isolated symbols are left alone), else at the start of
    the first repeated n-gram. Deterministic and easy to state in the paper.
    """
    if fillers:
        for i in range(len(hyp)):
            if hyp[i] in fillers and sum(c in fillers for c in hyp[i:i+3]) >= run:
                return hyp[:i].strip()
    toks = hyp.split()
    for n in (6, 5, 4):
        for i in range(len(toks) - 2*n + 1):
            if toks[i:i+n] == toks[i+n:i+2*n]:
                return " ".join(toks[:i]).strip()
    return hyp.strip()


def analyse(lang, gt_p, pred_p, args):
    gt, pred = parse(gt_p), parse(pred_p)
    keys = [k for k in gt if k in pred]
    if not keys:
        sys.exit(f"{lang}: no shared utterance ids between gt and pred")

    fillers = filler_charset([gt[k] for k in keys], [pred[k] for k in keys])
    # whitespace-free language? decided once per language, from the references
    tot_w = sum(len(gt[k].split()) for k in keys)
    tot_c = sum(len(gt[k]) for k in keys)
    use_chars = (tot_c / max(tot_w, 1)) > 20
    rows = []
    n_fill = n_loop = n_coll = n_diff = 0
    ratios = []
    refs, raws, cuts = [], [], []

    for k in keys:
        ref, hyp = gt[k], pred[k]

        ref_w, hyp_w = len(ref.split()), len(hyp.split())
        ratio = (len(hyp) / max(len(ref), 1)) if use_chars else (hyp_w / max(ref_w, 1))
        ratios.append(ratio)

        f = bool(fillers & set(hyp))
        l = ngram_loop(hyp)
        cut = truncate(hyp, fillers)
        d = norm(cut) != norm(hyp)
        collapsed = f or l or d or ratio >= args.ratio_thr

        n_fill += f; n_loop += l; n_diff += d; n_coll += collapsed
        refs.append(ref); raws.append(hyp); cuts.append(cut or " ")

        rows.append(dict(lang=lang, utt=k, ref_len=len(ref.split()),
                         hyp_len=len(hyp.split()), ratio=round(ratio, 2),
                         filler=int(f), loop=int(l), truncated=int(d),
                         collapsed=int(collapsed), hyp_cut=cut[:200]))

    N = len(keys)
    s = dict(lang=lang, n=N,
             coll=100*n_coll/N, fill=100*n_fill/N, loop=100*n_loop/N,
             diff=100*n_diff/N,
             ratio=sum(ratios)/N, ratio_max=max(ratios),
             metric=("CER" if use_chars else "WER"),
             wer_raw=None, wer_strip=None,
             fillers="".join(sorted(fillers))[:12])

    if args.wer:
        try:
            import jiwer
            meas = jiwer.cer if use_chars else jiwer.wer
            s["wer_raw"] = 100 * meas(refs, raws)
            s["wer_strip"] = 100 * meas(refs, cuts)
        except ImportError:
            print("[warn] jiwer not installed: pip install jiwer", file=sys.stderr)
    return s, rows

# --------------------------------------------------------------------- reporting

def show(name, S):
    print(f"\n=== {name} ===")
    h = (f"{'language':<12}{'N':>6}{'collapse%':>11}{'filler%':>9}{'loop%':>8}"
         f"{'truncated%':>13}{'len.ratio':>11}{'max':>8}{'WER raw':>9}{'WER str':>9}")
    print(h); print("-"*len(h))
    for s in S:
        g = lambda k, f="{:.1f}": "--" if s[k] is None else f.format(s[k])
        print(f"{s['lang']:<12}{s['n']:>6}{s['coll']:>11.1f}{s['fill']:>9.1f}"
              f"{s['loop']:>8.1f}{g('diff'):>13}{s['ratio']:>11.2f}"
              f"{s['ratio_max']:>8.1f}{g('wer_raw'):>9}{g('wer_strip'):>9}")
    tot = sum(s["n"] for s in S)
    if tot:
        w = lambda k: sum(s[k]*s["n"] for s in S if s[k] is not None)/tot
        print("-"*len(h))
        print(f"{'ALL':<12}{tot:>6}{w('coll'):>11.1f}{w('fill'):>9.1f}"
              f"{w('loop'):>8.1f}{'':>13}{w('ratio'):>11.2f}")

def md(A, na, B=None, nb=None):
    print("\n--- markdown ---\n")
    if B is None:
        print("| Language | N | Collapsed % | Mean len. ratio | WER raw | WER truncated |")
        print("|---|---|---|---|---|---|")
        for s in A:
            g = lambda k: "--" if s[k] is None else f"{s[k]:.1f}"
            print(f"| {s['lang']} | {s['n']} | {s['coll']:.1f} | {s['ratio']:.2f} "
                  f"| {g('wer_raw')} | {g('wer_strip')} |")
    else:
        b = {s["lang"]: s for s in B}
        print(f"| Language | N | Collapsed % ({na}) | Collapsed % ({nb}) "
              f"| Len. ratio ({na}) | Len. ratio ({nb}) |")
        print("|---|---|---|---|---|---|")
        for s in A:
            o = b.get(s["lang"])
            if o:
                print(f"| {s['lang']} | {s['n']} | {s['coll']:.1f} | {o['coll']:.1f} "
                      f"| {s['ratio']:.2f} | {o['ratio']:.2f} |")

# -------------------------------------------------------------------------- main

def run(d, args):
    t = discover(d)
    if not t:
        sys.exit(f"no decode files found in {d}")
    S, R = [], []
    for lang, g, p in t:
        s, r = analyse(lang, g, p, args)
        S.append(s); R.extend(r)
    return S, R

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--dir2")
    ap.add_argument("--name", default="condition A")
    ap.add_argument("--name2", default="condition B")
    ap.add_argument("--ratio-thr", type=float, default=1.8)
    ap.add_argument("--wer", action="store_true", help="also compute WER/CER (needs jiwer)")
    ap.add_argument("--csv", default="eos_collapse_per_utt.csv")
    a = ap.parse_args()

    A, rows = run(a.dir, a); show(a.name, A)
    B = None
    if a.dir2:
        B, rb = run(a.dir2, a); show(a.name2, B)
        for r in rb: r["lang"] += f" [{a.name2}]"
        rows += rb
    md(A, a.name, B, a.name2)
    with open(a.csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"\nper-utterance detail -> {a.csv}", file=sys.stderr)

if __name__ == "__main__":
    main()