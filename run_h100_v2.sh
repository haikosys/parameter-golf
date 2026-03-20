#!/bin/bash
set -e
cd /workspace/parameter-golf

# Get the v2 script (SOTA #1 base + LoRA TTT + stride=32)
curl -sL "https://api.github.com/repos/haikosys/parameter-golf/contents/records/track_10min_16mb/2026-03-20_LoRATTT_Stride32_EMA_Seq2048/train_gpt.py?ref=submission/seq2048-ema-council" | python3 -c "import sys,json,base64; print(base64.b64decode(json.load(sys.stdin)['content']).decode())" > our_v2.py
echo "Downloaded v2: $(wc -l < our_v2.py) lines"

for SEED in 42 1337 7; do
  echo "========== V2 SEED $SEED =========="
  RUN_ID=v2_seed${SEED} \
  SEED=$SEED \
  TRAIN_SEQ_LEN=2048 \
  TRAIN_BATCH_TOKENS=524288 \
  WARMDOWN_ITERS=5000 \
  MUON_MOMENTUM=0.95 \
  MUON_MOMENTUM_WARMUP_START=0.85 \
  MUON_MOMENTUM_WARMUP_STEPS=500 \
  MATRIX_LR=0.02 \
  SCALAR_LR=0.02 \
  TIED_EMBED_LR=0.03 \
  GRAD_CLIP_NORM=1.0 \
  EMA_DECAY=0.999 \
  INT6_LAYERS="" \
  TTT_LORA_RANK=0 \
  EVAL_STRIDE=32 \
  torchrun --standalone --nproc_per_node=8 our_v2.py
done

echo "========== V2 RESULTS =========="
grep "final_int8_zlib_roundtrip_exact\|final_ttt_lora_exact" logs/v2_seed*.txt
