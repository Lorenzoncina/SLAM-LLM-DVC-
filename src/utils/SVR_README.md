# SVR (Singular Value-based Rehearsal) for the MEUSLI projector

Integration of *"Efficient Rehearsal for Continual Learning in ASR via Singular
Value Tuning"* (Vander Eeckt & Van hamme, 2026) into the SLAM-LLM MEUSLI
projector, for continual-learning fine-tuning of a new language without
catastrophic forgetting.

## What was added

| File | Purpose |
|------|---------|
| `SLAM-LLM/src/slam_llm/models/projector.py` | `GatedSVDLinear` and `EncoderProjectorConcatSVR` |
| `SLAM-LLM/src/slam_llm/models/slam_model.py` | registers `encoder_projector: linear-svr` |
| `src/utils/build_svr_projector.py` | Stage-1 ckpts → SVR-init checkpoint (SVD of ΔW) |
| `src/utils/export_svr_to_dense.py` | trained SVR ckpt → plain dense projector (for eval) |

`GatedSVDLinear` replaces a linear layer's weight with
`W = W_base + U·diag(σ(α)·s)·Vᵀ`, where `U,s,Vᵀ = SVD(W_ft − W_base)`. Only the
gate vector `α` (one scalar per singular value) is trainable; `W_base, U, s, Vᵀ`
and the (averaged) bias are frozen buffers. `α` starts at −10 so `σ(α)≈0` and the
projector begins exactly at the base model, preserving the 28 base languages.

## Workflow (per new language Tᵢ)

**Stage 0 — base.** The MEUSLI base projector checkpoint `W_{t-1}` (e.g. the
projector saved from the multilingual run).

**Stage 1 — fine-tune (plain).** Fine-tune on the new language with the normal
`encoder_projector: linear` recipe → produces `W̃_t` (`finetuned/model.pt`).
This is your existing per-language fine-tuning; no replay needed here.

**Stage 2a — build the gated projector.**
```bash
python src/utils/build_svr_projector.py \
    --base   <base>/model.pt \
    --ft     <finetuned>/model.pt \
    --out    <svr_init>/model.pt
```

**Stage 2b — train only the gates with rehearsal.** Run training with:
```yaml
model_config:
    encoder_projector: linear-svr
    # svr_alpha_init: -10.0   # optional
train_config:
    ckpt_path: <svr_init>/model.pt   # loads the SVD buffers + averaged LoRA + α
    freeze_peft: true                # freeze LoRA -> ONLY the gates α train
    lr: 1e-3                         # higher than the usual 1e-4 (see below)
    max_grad_norm: 1.0               # gradient clipping (prevents fp16 NaN divergence)
```
Feed the dataloader a mix of the new-language data and the rehearsal buffer
(the same replay data used in the uniform / FBS experiments).

Two settings are essential and were the cause of an early failure:

* **`lr: 1e-3` + `max_grad_norm: 1.0`.** With the usual `lr=1e-4` the gates
  barely move. The paper uses `lr=0.01`, but that was a small from-scratch model
  with batch size 64; here (1.7B LLM, fp16, batch 4) `lr=0.01` diverges to NaN.
  Use **`lr=1e-3`** with **gradient clipping** instead. Note: with **Adam** the
  near-zero sigmoid gradient at `α=−10` does *not* throttle the step (Adam
  normalises it), so `α` still moves ≈ `lr`/step and opens the gates over an
  epoch. `finetune.sh` hard-codes `++train_config.lr=...`; set it to `1e-3`.
* **`freeze_peft: true`.** MEUSLI trains a **LoRA** alongside the projector, and
  the base projector only works together with its LoRA. The build step averages
  the base/fine-tuned LoRA (Eq. 7) into `<svr_init>` and `freeze_peft` keeps it
  fixed, so only the projector gates adapt. Without this the LoRA is retrained
  from scratch (and then dropped), and decode degrades to prompt-echo
  hallucination.

**Stage 3 — export to a dense projector for evaluation.** With `freeze_peft`,
Stage 2 saves only `α` (it is the only `requires_grad` tensor), so pass the
SVR-init file (which holds the SVD buffers **and** the averaged LoRA) via
`--svr` and the trained `α` via `--alpha`:
```bash
python src/utils/export_svr_to_dense.py \
    --svr   <svr_init>/model.pt \
    --alpha <stage2>/asr_epoch_*/model.pt \
    --out   <dense>/model.pt
```
The dense checkpoint contains the folded projector weights **and** the LoRA
tensors. Evaluate it with `encoder_projector: linear` — no SVR code needed at
inference.

> **Decode gotcha.** In `dvc.yaml` the decode stage reuses
> `${train.encoder_projector}`, so set `encoder_projector: linear` in
> `params.yaml` before decoding the dense projector. If it is left at
> `linear-svr`, the SVD buffers load as zeros and the projector emits only its
> bias → constant embeddings → prompt-echo hallucination.

## The Stage-2 KD loss (implemented)

The paper's Stage-2 loss (Eq. 8) is now implemented:

```
L = CE(new) + svr_mem_weight · ( CE(memory) + KD(memory; θ_base) )
```

Crucially this is **not** data-mixing. A **separate memory minibatch is drawn
every step** from `memory_data_path` (cycling), so even a 1-sample/language
buffer is replayed on 100% of steps and actually constrains the gates. The KD
teacher is the base model, obtained for free by forcing the gates to 0
(`svr_base_projector`), which is exactly base projector + base LoRA — hence the
build now uses **`--lora-source base`** by default.

### Data layout (important — different from before)
Split your data into two jsonl files:
- `train_data_path` → **new task only** (e.g. Ukrainian).
- `memory_data_path` → **rehearsal buffer only** (1 / 25 / 500 samples per base
  language, **no** Ukrainian).

Previously these were concatenated into one file; that is what made the
rehearsal ineffective at small buffers.

### Stage-2 config (finetune.sh / params.yaml)
```yaml
encoder_projector: linear-svr
ckpt_path: <svr_init>/model.pt        # built with --lora-source base
freeze_peft: false                     # LoRA trains WITH the gates (see below)
lr: 1e-3                               # gates (alpha)
svr_lora_lr: 1e-4                      # LoRA (separate optimizer group)
max_grad_norm: 1.0
svr_kd: true
svr_mem_weight: 1.0                    # = (t-1)/2; raise for more retention
svr_kd_temperature: 1.0
dataset_config.memory_data_path: <rehearsal-only>.jsonl
```

### Why the LoRA must train in Stage 2 (`freeze_peft: false`)

MEUSLI's Stage-1 solution is a *coordinated pair* (projector, LoRA) trained
jointly. Gating only the projector delta while pinning the LoRA at base creates
a hybrid the network never saw: transcription content survives (high margin)
but the LLM's EOS calibration breaks for the old languages → endless
hallucinated tails after a correct transcription. This was observed at every
buffer size / memory weight / epoch. Letting the 1.38M-param LoRA re-calibrate
alongside the gates (anchored by the same memory CE+KD) restores stopping —
the same reason plain data-replay (which trains projector+LoRA jointly) never
hallucinates.

With trainable LoRA the KD teacher must stay the *base* model: the trainer
snapshots the base LoRA at startup (the svr-init carries base LoRA) and swaps
it in for every teacher forward, so teacher = base projector (gates off) +
base LoRA throughout training. The Stage-2 checkpoint then contains alpha +
trained LoRA, and the export step picks the trained LoRA up automatically.

`svr_mem_weight` is the stability/plasticity knob: higher → more retention, less
new-language learning. Start at 1.0; try 2–5 if forgetting persists, lower if UK
WER is too high.

> **Cost:** KD adds two extra forwards on the memory batch per step (teacher +
> student), so Stage 2 is ~2–3× slower. The memory forwards run in `eval()` mode
> so LoRA dropout is off and the teacher is deterministic.
