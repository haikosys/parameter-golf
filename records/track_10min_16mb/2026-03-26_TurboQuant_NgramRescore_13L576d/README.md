# Record: TurboQuant + Full-Rescore N-gram Cache (11L/576d/3.5x)

**val_bpb: TBD** (3-seed mean) | **~14.8 MB** artifact | 8xH100 SXM, 600s

## Summary

TurboQuant rotation-based Lloyd-Max codebook quantization replaces int6, enabling 39% more parameters (37.6M vs 27.0M) in the same 16MB budget. Combined with PR #870's two-pass full-rescore n-gram cache for eval.

## Architecture
- 11L / 576d / 8 heads / 4 KV heads / 3.5x MLP (2016 hidden)
- 37.6M params (39% more than PR #870's 27.0M)
- LeakyReLU(0.5)^2 activation, XSA last 4 layers
- BigramHash(2048), ValueEmbedding on layers 9-10
- SmearGate, U-Net skip connections, partial RoPE(16)
- Tied embeddings, logit softcap=30

## Quantization: TurboQuant
- Rotation-based Lloyd-Max codebooks with deterministic QR rotation matrix
- Per-component bit allocation: 2-bit MLP up, 3-bit attn/MLP down, 4-bit embeddings
- Progressive QAT during warmdown: 4-bit -> 3-bit -> 2-bit (STE)
- LZMA compression -> ~14.8 MB artifact (1.2 MB headroom)

## Eval: Two-Pass Full-Rescore N-gram Cache (from PR #870)
- Pass 1: Sliding-window neural eval (stride=64), store per-token model_p and entropy
- Build: Complete order 2-12 n-gram cache from all val tokens (numpy vectorized, np.bincount)
- Pass 2: Rescore ALL ~62M tokens against full cache with entropy-adaptive alpha
- No TTT required

## Training
- Muon optimizer (matrices, lr=0.025) + AdamW (embeddings lr=0.035, scalars lr=0.025)
- EMA(0.997), SWA during warmdown, gradient clipping 0.3
- 786K tokens/batch, seq_len=2048, warmdown 3500 steps
- 600s wall clock on 8xH100 SXM

## Results

TBD — awaiting 3-seed runs.

| Seed | val_bpb (neural) | val_bpb (n-gram rescore) | Artifact | Train time | Eval time |
|------|------------------|--------------------------|----------|------------|-----------|
| 1337 | TBD | TBD | TBD | TBD | TBD |
| 42   | TBD | TBD | TBD | TBD | TBD |
| 2024 | TBD | TBD | TBD | TBD | TBD |

## Reproduction
```bash
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Lineage
- PR #870 (BROADSIDE): Full-rescore n-gram cache, two-pass eval
- PR #549: LeakyReLU^2, parallel Muon
- PR #287: Partial RoPE, LN Scale, EMA, XSA
- TurboQuant: Rotation-based quantization with Lloyd-Max codebooks
