"""Fold a trained SVR projector into a plain (dense) linear projector.

After Stage-2 training, the effective weight of each gated layer is

    W = W_base + U diag(sigmoid(alpha) * s) Vh

This script reconstructs that dense weight and writes a checkpoint compatible
with `EncoderProjectorConcat` (`encoder_projector: linear`), so the result can
be evaluated/served through the standard inference path with no SVR code.

It needs the SVD buffers (W_base, U, s, Vh, bias) and the trained alpha gates.
Depending on how you saved Stage 2:

  * If you trained with `save_trainable_only: false`, a single Stage-2 model.pt
    contains everything: pass it as --svr.

  * If you trained with `save_trainable_only: true` (only alpha is saved), pass
    the SVR-init checkpoint (buffers) as --svr and the trained model.pt
    (alpha) as --alpha; the two are merged.

Usage:
    python src/utils/export_svr_to_dense.py --svr STAGE2_OR_INIT.pt \
        [--alpha TRAINED_ALPHA.pt] --out dense_projector/model.pt
"""
import argparse
from collections import OrderedDict

import torch

PREFIX = "encoder_projector."


def _load(path):
    """Load a checkpoint, returning its raw state dict (keys left untouched).

    Keys are not rewritten here: projector keys are looked up prefix-tolerantly
    below, while non-projector keys (e.g. LoRA, named ``llm.*``) must be carried
    through verbatim.
    """
    sd = torch.load(path, map_location="cpu")
    if isinstance(sd, dict) and "model" in sd and isinstance(sd["model"], dict):
        sd = sd["model"]
    return OrderedDict(sd)


def _get(sd, name):
    for key in (PREFIX + name, name):
        if key in sd:
            return sd[key]
    raise KeyError(name)


def export(svr_path, out_path, alpha_path=None, layers=("linear1", "linear2")):
    sd = _load(svr_path)
    if alpha_path is not None:
        sd.update(_load(alpha_path))  # overwrite / add the trained alpha gates

    dense = OrderedDict()
    for name in layers:
        base = f"{PREFIX}{name}"
        try:
            w_base = _get(sd, f"{name}.weight_base").to(torch.float32)
            U = _get(sd, f"{name}.U").to(torch.float32)
            s = _get(sd, f"{name}.s").to(torch.float32)
            Vh = _get(sd, f"{name}.Vh").to(torch.float32)
            alpha = _get(sd, f"{name}.alpha").to(torch.float32)
        except KeyError as e:
            raise KeyError(f"missing SVR buffer/param {e} for layer '{name}'. "
                           "Did you pass the SVR-init checkpoint (and --alpha if "
                           "you saved trainable-only)?")
        gated_s = torch.sigmoid(alpha) * s
        weight = w_base + (U * gated_s) @ Vh
        dense[f"{base}.weight"] = weight
        try:
            dense[f"{base}.bias"] = _get(sd, f"{name}.bias").to(torch.float32)
        except KeyError:
            pass

        open_gates = ((torch.sigmoid(alpha) > 0.05) & (torch.sigmoid(alpha) < 0.95)).sum().item()
        accepted = (torch.sigmoid(alpha) >= 0.95).sum().item()
        print(f"[{name}] rank={s.numel()} accepted(sig>=.95)={accepted} "
              f"intermediate={open_gates} mean_sigma={torch.sigmoid(alpha).mean().item():.3f}")

    # Carry every non-projector param (e.g. LoRA) into the dense checkpoint, so
    # decode reproduces base projector + its adaptation. Without this the LLM
    # loses the LoRA it was trained with and only echoes the prompt.
    svr_buffer_suffixes = (".weight_base", ".U", ".s", ".Vh", ".alpha")
    n_carried = 0
    for key, val in sd.items():
        if key.startswith(PREFIX) and key.endswith(svr_buffer_suffixes):
            continue  # SVR internals, already folded into dense weight
        if key.startswith(PREFIX) and (key.endswith(".weight") or key.endswith(".bias")):
            continue  # projector weight/bias already produced above
        dense[key] = val
        n_carried += 1
    if n_carried:
        print(f"[others] carried {n_carried} non-projector tensors (e.g. LoRA) into dense ckpt")

    torch.save(dense, out_path)
    print(f"\nSaved dense projector to: {out_path}")
    print("Evaluate it with `encoder_projector: linear` and this as ckpt_path.")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--svr", required=True,
                   help="Stage-2 model.pt (full) or the SVR-init checkpoint (buffers)")
    p.add_argument("--alpha", default=None,
                   help="trained alpha-only model.pt (if Stage 2 saved trainable-only)")
    p.add_argument("--out", required=True, help="output path for the dense projector")
    p.add_argument("--layers", nargs="+", default=["linear1", "linear2"])
    args = p.parse_args()
    export(args.svr, args.out, args.alpha, tuple(args.layers))


if __name__ == "__main__":
    main()
