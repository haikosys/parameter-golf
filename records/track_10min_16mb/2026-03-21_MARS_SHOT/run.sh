#!/bin/bash
# MARS SHOT: PR#338 base (PR#315 + TTT) = 1.1256 BPB
# This is the EXACT PR#338 command + our PCA-driven additions
set -e
cd /workspace/parameter-golf

for SEED in 42 1337 7; do
  echo "========== MARS SHOT SEED $SEED =========="
  RUN_ID=mars_seed${SEED} \
  SEED=$SEED \
  NUM_LAYERS=11 \
  BIGRAM_VOCAB_SIZE=2048 \
  XSA_LAST_N=4 \
  EMA_ENABLED=1 \
  EMA_DECAY=0.997 \
  SWA_ENABLED=0 \
  LATE_QAT=1 \
  QAT_THRESHOLD=0.1 \
  ROPE_DIMS=16 \
  LN_SCALE=1 \
  TTT_ENABLED=1 \
  TTT_LR=0.002 \
  TTT_EPOCHS=3 \
  TTT_MOMENTUM=0.9 \
  TTT_FREEZE_BLOCKS=2 \
  MUON_WD=0.04 \
  ADAM_WD=0.04 \
  MATRIX_LR=0.025 \
  SCALAR_LR=0.025 \
  TIED_EMBED_LR=0.035 \
  MUON_MOMENTUM=0.99 \
  WARMDOWN_ITERS=3000 \
  ITERATIONS=9000 \
  MAX_WALLCLOCK_SECONDS=600 \
  EVAL_STRIDE=64 \
  torchrun --standalone --nproc_per_node=8 train_gpt.py
done

echo "========== RESULTS =========="
grep "final_int6_sliding_window_exact\|final_int6_roundtrip_exact\|Serialized model int6\|Total submission" logs/mars_seed*.txt
