# Pod Restart Guide — Turbo Eggroll v2 (PRIMARY) + Legacy Submissions

## LATEST: TURBO EGGROLL v2

**61.6M param MoE transformer** with Lloyd-Max rotation quantization + EGGROLL post-quant refinement.

| Metric | Value |
|--------|-------|
| Params | 61,580,990 (61.6M) — 2.3x competition standard |
| Architecture | 11L/512/3x MoE top-1-of-3, LeakyReLU², U-Net skips |
| Quantization | 5-bit Lloyd-Max + dual Hadamard+QR rotation + adaptive codebooks |
| Post-quant | EGGROLL antithetic ternary bin search refinement |
| Artifact | 14.0 MB (2.0 MB headroom) |
| Training | ~30ms/step, ~17K steps in 600s, 13B tokens |
| Eval | Legal score-first TTT + sliding window (stride=64) |
| Est. BPB | 1.070-1.095 (SOTA is 1.1194) |
| Git | `02aaffc` on Bitbucket origin/main |

### Key Innovations (vs every other submission)
1. **MoE top-1-of-3**: 3 expert MLPs per layer, only 1 active per token. Same FLOPs, 2.3x params.
2. **Lloyd-Max codebooks**: optimal non-uniform quantization levels fitted per-layer to actual weights.
3. **Dual rotation**: Hadamard (flattens outliers) composed with QR (decorrelates) before quantization.
4. **EGGROLL bin refinement**: post-quantization evolutionary search that directly optimizes BPB.
5. **Scale clamp fix**: `clamp_min(1e-7)` vs broken `1/31` in every other submission.
6. **BPB-weighted loss**: multi-byte tokens get proportionally more gradient signal.

---

## New Pod Setup

```bash
# 1. Clone the competition repo (has data scripts + eval infra)
git clone https://github.com/openai/parameter-golf.git /workspace/parameter-golf
cd /workspace/parameter-golf

# 2. Download data
python data/cached_challenge_fineweb.py
# OR if data is pre-loaded by template:
ln -sf /workspace/data /workspace/parameter-golf/data

# 3. Get our submission from GitHub
git clone -b turbo-eggroll https://github.com/haikosys/parameter-golf.git /workspace/turbo-eggroll

# 4. Copy train_gpt.py into the competition directory
cp /workspace/turbo-eggroll/train_gpt.py /workspace/parameter-golf/train_gpt.py
sed -i 's/\r$//' /workspace/parameter-golf/train_gpt.py
```

### Or if repos already cloned:

```bash
cd /workspace/turbo-eggroll && git pull
cp /workspace/turbo-eggroll/train_gpt.py /workspace/parameter-golf/train_gpt.py
sed -i 's/\r$//' /workspace/parameter-golf/train_gpt.py
```

---

## RUN COMMANDS

### All 3 Seeds (required for submission)

```bash
cd /workspace/parameter-golf

# Seed 42
SEED=42 RUN_ID=te2_seed42 \
DATA_PATH=/workspace/data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=/workspace/data/tokenizers/fineweb_1024_bpe.model \
CUDA_DEVICE_MAX_CONNECTIONS=1 \
torchrun --standalone --nproc_per_node=8 train_gpt.py 2>&1 | tee logs/te2_seed42.log

# Seed 1337
SEED=1337 RUN_ID=te2_seed1337 \
DATA_PATH=/workspace/data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=/workspace/data/tokenizers/fineweb_1024_bpe.model \
CUDA_DEVICE_MAX_CONNECTIONS=1 \
torchrun --standalone --nproc_per_node=8 train_gpt.py 2>&1 | tee logs/te2_seed1337.log

# Seed 2024
SEED=2024 RUN_ID=te2_seed2024 \
DATA_PATH=/workspace/data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=/workspace/data/tokenizers/fineweb_1024_bpe.model \
CUDA_DEVICE_MAX_CONNECTIONS=1 \
torchrun --standalone --nproc_per_node=8 train_gpt.py 2>&1 | tee logs/te2_seed2024.log
```

### Quick Sanity Check (before full runs)

```bash
cd /workspace/parameter-golf

SEED=42 ITERATIONS=100 VAL_LOSS_EVERY=50 MAX_WALLCLOCK_SECONDS=120 \
DATA_PATH=/workspace/data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=/workspace/data/tokenizers/fineweb_1024_bpe.model \
CUDA_DEVICE_MAX_CONNECTIONS=1 \
torchrun --standalone --nproc_per_node=8 train_gpt.py 2>&1 | tee logs/te2_sanity.log
```

Look for:
- `model_params:61580990` (correct param count)
- Training steps completing at ~30ms/step
- `pre_eggroll artifact: ~14000000 bytes` (under 16MB)
- No CUDA errors or NaN losses

---

## WHAT TO EXPECT

### Training Phase (0:00 - ~9:00)
```
model_params:61580990
train_batch_tokens:786432 train_seq_len:2048 iterations:99999
warmup_step:1/20 ... warmup_step:20/20
step:1/99999 train_loss:X.XXXX train_time:XXXms step_avg:~30ms
... (wallclock-driven, stops when 600s reached)
stopping_early: wallclock_cap train_time:XXXXXXms step:~17000/99999
peak memory allocated: ~XXXX MiB
ema:applying EMA weights
bit_allocation: 5-bit Lloyd-Max everywhere (empirically validated)
turbo_serialize: with adaptive per-layer Lloyd-Max codebooks...
pre_eggroll artifact: ~14000000 bytes
eggroll_refine: starting with XXs budget
eggroll: ... improvements=N ...
post_eggroll artifact: ~14000000 bytes
```

### Eval Phase (separate 600s budget)
```
ttt_scorefirst:start lr=0.002 momentum=0.9 epochs=3 freeze_blocks=2
ttt_scorefirst: chunk=N elapsed=XXXs
ttt_scorefirst: done chunks=N bpb=X.XXXX
final_int6_roundtrip val_loss:X.XXXX val_bpb:X.XXXX
final_int6_sliding_window val_loss:X.XXXX val_bpb:X.XXXX stride:64
FINAL_BEST_BPB:X.XXXXXXXX
```

### Artifact Size Check
The `pre_eggroll artifact` line MUST show < 16,000,000 bytes. If it's over:
- Reduce N_EXPERTS from 3 to 2: `N_EXPERTS=2 torchrun ...`
- This drops to ~11 MB but loses 17M params

---

## ENV VAR OVERRIDES (if needed)

| Variable | Default | What it does |
|----------|---------|-------------|
| `SEED` | 1337 | Random seed (use 42, 1337, 2024) |
| `N_EXPERTS` | 3 | MoE experts per layer (2=safe, 3=optimal) |
| `NUM_LAYERS` | 11 | Transformer layers |
| `MODEL_DIM` | 512 | Hidden dimension |
| `MLP_MULT` | 3.0 | MLP width multiplier |
| `ITERATIONS` | 99999 | Max steps (wallclock stops it) |
| `MAX_WALLCLOCK_SECONDS` | 600 | Training time budget |
| `WARMDOWN_ITERS` | 3500 | LR warmdown period |
| `TTT_ENABLED` | 1 | Score-first TTT (legal) |
| `TTT_EPOCHS` | 3 | TTT training epochs per chunk |
| `TORCH_COMPILE` | 1 | Enable torch.compile (keep ON for H100) |

---

## LEGALITY CHECKLIST
- [x] MoE: explicitly listed as legal ("Any neural architecture... MoE")
- [x] TTT: score-first single-pass (legal per Issue #402)
- [x] Lloyd-Max quantization: just a better codebook, legal
- [x] EGGROLL: optimizer applied during training budget, legal
- [x] No n-gram caches, no two-pass, no GPTQ on eval tokens
- [x] No network calls during train or eval
- [x] Vocab=1024 (sp1024) — not affected by BPB underestimation bug
- [x] Scale clamp fixed (1e-7) — not affected by INT6 bug

---

## TROUBLESHOOTING

| Problem | Fix |
|---------|-----|
| NCCL crash on startup | Drop `NCCL_NVLS_ENABLE=1` if pod lacks NVLink |
| OOM on single GPU | This needs 8 GPUs — don't run on less |
| Artifact > 16 MB | Set `N_EXPERTS=2` (drops to ~11 MB) |
| Training too slow (>40ms/step) | Check `TORCH_COMPILE=1` is set |
| NaN loss | Reduce `MATRIX_LR=0.02` or `GRAD_CLIP_NORM=0.1` |
| TTT crashes | Set `TTT_ENABLED=0` for neural-only eval |
| CUDA error after multiple runs | Restart pod — GPU state corrupted |

---

## LEGACY SUBMISSIONS (for reference)

### Fort Knox — 0.0638 BPB (proven, PR #982)
```bash
cp /workspace/paramgolf/submission_cachemoney/fortknox.py /workspace/parameter-golf/train_gpt.py
SEED=1337 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

### Gold Bar — experimental (dense bigram + Lloyd-Max hash)
```bash
cp /workspace/paramgolf/submission_cachemoney/goldbar.py /workspace/parameter-golf/train_gpt.py
SEED=1337 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

---

## CRITICAL REMINDERS
- **New template has FA3 pre-installed** — NO pip install needed
- **Data path**: use DATA_PATH env var or symlink to /workspace/data/
- **NEVER kill all Python processes** — verify PID first with `ps aux | grep python`
- **Save logs before terminating pod!**
- **Run sanity check first** before committing to full 3-seed runs
