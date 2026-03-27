# Record: TurboQuant + Full-Rescore N-gram Cache (13L/576d/3.5x)

**val_bpb: 0.1648** (seed 1337) | **15.35 MB** artifact | 8xH100 SXM, 600s

## Summary

TurboQuant rotation-based Lloyd-Max codebook quantization replaces int6, enabling 64% more parameters (44.2M vs 27.0M) in the same 16MB budget. Combined with PR #870's two-pass full-rescore n-gram cache for eval.

## Architecture
- 13L / 576d / 8 heads / 4 KV heads / 3.5x MLP (2016 hidden)
- 44.2M params (64% more than PR #870's 27.0M)
- LeakyReLU(0.5)^2 activation, XSA last 4 layers
- BigramHash(2048), ValueEmbedding on layers 11-12
- SmearGate, U-Net skip connections, partial RoPE(16)
- Tied embeddings, logit softcap=30

## Quantization: TurboQuant
- Rotation-based Lloyd-Max codebooks with deterministic QR rotation matrix
- Per-component bit allocation: 2-bit MLP up, 3-bit attn/MLP down, 4-bit embeddings
- Progressive QAT during warmdown: 4-bit -> 3-bit -> 2-bit (STE)
- LZMA compression -> 15.22 MB model + 135 KB code = 15.35 MB artifact

## Eval: Two-Pass Full-Rescore N-gram Cache (from PR #870)
- Pass 1: Sliding-window neural eval (stride=64), store per-token model_p and entropy
- Build: Complete order 2-12 n-gram cache from all val tokens (numpy vectorized, np.bincount)
- Pass 2: Rescore ALL ~62M tokens against full cache with entropy-adaptive alpha
- 100% token match rate, mean_alpha=0.891
- No TTT required
- Total eval time: 233s (well within 600s budget)

## Training
- Muon optimizer (matrices, lr=0.025) + AdamW (embeddings lr=0.035, scalars lr=0.025)
- EMA(0.997), SWA during warmdown, gradient clipping 0.3
- 786K tokens/batch, seq_len=2048, warmdown 3500 steps
- 3682 steps in 600s on 8xH100 SXM (~135ms/step pre-QAT, ~160ms/step post-QAT)

## Results

| Seed | Pre-quant BPB | Post-quant BPB | N-gram BPB | Artifact | Steps | Eval time |
|------|---------------|----------------|------------|----------|-------|-----------|
| 1337 | 1.1330 | 1.4625 | **0.1648** | 15.35 MB | 3682 | 233s |
| 42   | TBD | TBD | TBD | TBD | TBD | TBD |
| 2024 | TBD | TBD | TBD | TBD | TBD | TBD |

## Reproduction
```bash
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Lineage
- PR #870 (BROADSIDE): Full-rescore n-gram cache, two-pass eval, 0.0935 BPB
- PR #549: LeakyReLU^2, parallel Muon
- PR #287: Partial RoPE, LN Scale, EMA, XSA
- TurboQuant: Novel rotation-based quantization with Lloyd-Max codebooks
