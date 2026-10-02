"""Build a Singular Value-based Rehearsal (SVR) projector checkpoint.

Implements Stage-2a of "Efficient Rehearsal for Continual Learning in ASR via
Singular Value Tuning" (Vander Eeckt & Van hamme, 2026) for the MEUSLI linear
projector.

Given:
  * BASE  : the projector before adaptation        (W_{t-1}, e.g. MEUSLI base)
  * FT    : the projector after Stage-1 fine-tuning (W~_t,  fine-tuned on the
            new language with the plain `encoder_projector: linear` recipe)

for each linear layer it computes dW = W_ft - W_base, decomposes it with a thin
SVD (dW = U diag(s) Vh), averages the bias, and initialises the gating vector
alpha so that sigmoid(alpha) ~= 0 (i.e. the layer starts at W_base).

The resulting checkpoint matches `EncoderProjectorConcatSVR` and can be loaded
into the model via `ckpt_path=...` (model.load_state_dict(strict=False)). Then
train only the alpha gates with rehearsal data using
`encoder_projector: linear-svr`.

Usage:
    python src/utils/build_svr_projector.py \
        --base   path/to/base/model.pt \
        --ft     path/to/finetuned/model.pt \
        --out    path/to/svr_init/model.pt \
        [--alpha-init -10.0] [--layers linear1 linear2]
"""
import argparse
from collections import OrderedDict

import torch

PREFIX = "encoder_projector."


def _load_projector_sd(path):
    """Load a checkpoint and return a state dict keyed `encoder_projector.*`."""
    sd = torch.load(path, map_location="cpu")
    if isinstance(sd, dict) and "model" in sd and isinstance(sd["model"], dict):
        sd = sd["model"]
    out = OrderedDict()
    for key, value in sd.items():
        if key.startswith(PREFIX):
            out[key] = value
        else:
            # checkpoint may already have the prefix stripped (extract_projector.py)
            out[PREFIX + key] = value
    return out


def _load_full_sd(path):
    """Load a checkpoint and return its raw state dict (no key filtering)."""
    sd = torch.load(path, map_location="cpu")
    if isinstance(sd, dict) and "model" in sd and isinstance(sd["model"], dict):
        sd = sd["model"]
    return sd


def build(base_path, ft_path, out_path, alpha_init=-10.0, layers=("linear1", "linear2"),
          lora_source="base"):
    base = _load_projector_sd(base_path)
    ft = _load_projector_sd(ft_path)

    svr = OrderedDict()
    for name in layers:
        w_key = f"{PREFIX}{name}.weight"
        b_key = f"{PREFIX}{name}.bias"
        if w_key not in base or w_key not in ft:
            raise KeyError(f"missing '{w_key}' in base or fine-tuned checkpoint")

        w_base = base[w_key].to(torch.float32)
        w_ft = ft[w_key].to(torch.float32)
        delta = w_ft - w_base
        U, s, Vh = torch.linalg.svd(delta, full_matrices=False)

        svr[f"{PREFIX}{name}.weight_base"] = w_base
        svr[f"{PREFIX}{name}.U"] = U
        svr[f"{PREFIX}{name}.s"] = s
        svr[f"{PREFIX}{name}.Vh"] = Vh
        svr[f"{PREFIX}{name}.alpha"] = torch.full((s.numel(),), float(alpha_init))

        if b_key in base:
            b_base = base[b_key].to(torch.float32)
            b_ft = ft[b_key].to(torch.float32) if b_key in ft else b_base
            svr[f"{PREFIX}{name}.bias"] = (b_base + b_ft) / 2.0

        kept = (torch.sigmoid(torch.tensor(alpha_init)) * s).sum().item()
        print(f"[{name}] in={w_base.shape[1]} out={w_base.shape[0]} rank={s.numel()} "
              f"||dW||_nuc={s.sum().item():.3f} initial gated energy={kept:.3e}")

    if lora_source != "none":
        # Non-gated params (e.g. LoRA) are NOT decomposed. They are carried over
        # and frozen during Stage 2.
        #   "base"    -> use the base model's params (default). This makes the KD
        #                teacher (gates off) EXACTLY the base MEUSLI, and keeps
        #                the gates-closed model identical to the base model.
        #   "average" -> (base+ft)/2, the paper's Eq. (7) for non-gated params.
        base_full = _load_full_sd(base_path)
        ft_full = _load_full_sd(ft_path) if lora_source == "average" else {}
        n = 0
        for key, vb in base_full.items():
            if key.startswith(PREFIX):
                continue  # projector handled above
            if lora_source == "average" and key in ft_full and ft_full[key].shape == vb.shape:
                svr[key] = (vb.to(torch.float32) + ft_full[key].to(torch.float32)) / 2.0
            else:
                svr[key] = vb
            n += 1
        if n:
            print(f"[others] carried {n} non-projector tensors (e.g. LoRA) from '{lora_source}'")

    torch.save(svr, out_path)
    print(f"\nSaved SVR-init projector checkpoint to: {out_path}")
    print("Train Stage 2 with `encoder_projector: linear-svr` and this as ckpt_path.")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base", required=True, help="base projector checkpoint (W_{t-1})")
    p.add_argument("--ft", required=True, help="Stage-1 fine-tuned projector checkpoint (W~_t)")
    p.add_argument("--out", required=True, help="output path for the SVR-init checkpoint")
    p.add_argument("--alpha-init", type=float, default=-10.0,
                   help="initial gate logit; sigmoid(alpha-init) ~= 0 (default: -10.0)")
    p.add_argument("--layers", nargs="+", default=["linear1", "linear2"],
                   help="linear layer names to gate (default: linear1 linear2)")
    p.add_argument("--lora-source", choices=["base", "average", "none"], default="base",
                   help="how to carry non-projector params (LoRA): base (default, "
                        "clean KD teacher), average (paper Eq.7), or none")
    args = p.parse_args()
    build(args.base, args.ft, args.out, args.alpha_init, tuple(args.layers),
          lora_source=args.lora_source)


if __name__ == "__main__":
    main()
