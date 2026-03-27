# Fort Knox: Zero Val-Adaptation Baseline

**val_bpb: TBD** | **~8 MB** artifact | 8xH100 SXM

## Summary

Fort Knox is a deliberately ultra-conservative submission designed to establish a legality baseline. It uses **zero adaptation on validation data** — no incremental cache, no phrase cache, no TTT, no alpha calibration, no two-pass rescoring. The only information available at eval time is what was serialized into the artifact during training: model weights + a packed n-gram frequency table from training data.

If Fort Knox is ruled illegal, then every submission in the competition is illegal, because every submission uses at least model weights trained on training data.

## Method

1. **Training (600s on 8xH100):**
   - Train a 6L/256d transformer (4.2M params, FP16)
   - Every 10th step, update a 32K-bucket order 2-9 n-gram count table from the training batch tokens
   - Serialize model weights (FP16) + n-gram count table (~2.3MB) into a single artifact via LZMA

2. **Eval:**
   - Load artifact (model + training n-gram table). No training data accessed.
   - For each chunk of validation tokens:
     - Score with the neural model (frozen weights, inference mode)
     - Score against the packed training n-gram table (frozen, no updates)
     - Blend: `p = (1 - 0.85) * p_neural + 0.85 * p_training_ngram` for matched tokens
     - Apply temperature sharpening (T=0.85) to model logits before softmax
   - **No val cache updates. No phrase cache. No TTT. No alpha calibration.**
   - Report the single-pass scores directly.

## Legality Analysis

### What Fort Knox Does NOT Do

| Technique | Fort Knox | Legal Status |
|-----------|-----------|-------------|
| Two-pass full rescore | **No** | Debated ([PR #846](https://github.com/openai/parameter-golf/pull/846)) |
| Incremental val n-gram cache | **No** | Legal per [PR #913](https://github.com/openai/parameter-golf/pull/913), but conservative exclusion |
| Phrase cache from val data | **No** | Legal per PR #913, excluded |
| Score-first TTT | **No** | Legal per [Issue #677](https://github.com/openai/parameter-golf/issues/677), excluded |
| Online alpha calibration | **No** | Gray area, excluded |
| Oracle/min(NLL) selection | **No** | Illegal per [PR #573](https://github.com/openai/parameter-golf/pull/573) |
| GPTQ calibration at eval time | **No** | Illegal per Issue #677 |
| Any val data touching any cache | **No** | — |

### What Fort Knox DOES Do

| Technique | Legal Basis |
|-----------|------------|
| Train neural model on training data (600s) | Core competition rule |
| Build n-gram counts from training data (during training) | Same as training model weights — learning from training data |
| Serialize both into artifact (<16MB) | FAQ: "you aren't allowed to access any training data during evaluation, **unless you pay for those bits in the <16MB limit**" |
| Load artifact at eval start | Core competition rule |
| Score val tokens with frozen model | Core competition rule |
| Blend with frozen training n-gram table | The table is part of the artifact, no different from model weights |
| Temperature sharpening (T=0.85) | Stateless transform of model logits; used in accepted [PR #913](https://github.com/openai/parameter-golf/pull/913) |

### Rule-by-Rule Compliance (Issue #677)

**"You can't cheat by training on the validation set before you evaluate on the validation set."**
Fort Knox never trains on the validation set. The n-gram table is built entirely from `fineweb_train_*` during the 600s training budget.

**"You are only allowed to test-time train on validation set tokens you've already evaluated your model on."**
Fort Knox does not test-time train at all. The model and n-gram table are frozen throughout eval.

**"No external downloads, training dataset access, or network calls are allowed during evaluation. The artifact must be fully self-contained."**
Fort Knox loads only the artifact. No `fineweb_train_*` files are opened during eval. The artifact is self-contained.

**"GPTQ/Hessian calibration uses fineweb_train_* during evaluation" — ILLEGAL**
Fort Knox does not run GPTQ. The n-gram table was built during training, not eval.

**"People are trying to sneak in extra compute between training and eval by arguing it's part of 'artifact construction'."**
Fort Knox builds the n-gram table *during* the 600s training budget, not in a separate phase. The wallclock covers both neural training and n-gram construction.

### Precedent

The packed training n-gram approach is used by multiple accepted/pending top submissions:

- [PR #962](https://github.com/openai/parameter-golf/pull/962) (0.0214 BPB): "The packed n-gram cache in the artifact is derived from training data only and is produced within the 600 second training budget."
- [PR #931](https://github.com/openai/parameter-golf/pull/931) (0.0498 BPB): "The packed n-gram cache in the artifact is derived from training data only."
- [PR #944](https://github.com/openai/parameter-golf/pull/944) (0.0165 BPB): "Added packed causal n-gram memory path (built from train shards, loaded at eval start)."
- [PR #945](https://github.com/openai/parameter-golf/pull/945) (0.0274 BPB): "Pre-filled from all training shards at startup."

Fort Knox is strictly MORE conservative than all of these — it does not use any incremental val cache or TTT that those submissions use.

### The Strongest Possible Argument Against Fort Knox

"The packed training n-gram table gives the model access to training data statistics during eval, which could be considered 'training data access during evaluation'."

**Rebuttal:** The model weights themselves ARE training data statistics. Every parameter in the transformer was learned from training data. The n-gram table is no different — it is a compressed statistical summary of training data, serialized into the artifact, counted against the 16MB budget. The FAQ explicitly permits this: "unless you pay for those bits in the <16MB limit."

If packed training statistics in the artifact are illegal, then model weights are illegal, and the competition has no valid submissions.

## Architecture

- 6L / 256d / 4 heads / 2 KV heads / 3x MLP (768 hidden)
- 4.2M params, FP16 (zero quantization penalty)
- Packed training n-gram: 32K buckets, order 2-9, ~2.3MB
- Total artifact: ~8 MB (well under 16MB)
- Temperature sharpening: T=0.85

## Reproduction

```bash
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Expected Performance

Fort Knox will have significantly worse BPB than submissions using incremental val caches (~0.3-0.8 BPB estimated vs ~0.06-0.09 for cache-augmented submissions). This is by design — it sacrifices performance for unassailable legality.

The performance gap between Fort Knox and cache-augmented submissions quantifies exactly how much BPB comes from val-data adaptation vs pure training-time knowledge.
