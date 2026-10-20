#!/usr/bin/env bash
# SDS ablation on guidance_scale.
# Each value gets its own directory, so runs never overwrite each other:
#   outputs/ablation/sds_guidance/gs_<value>/{<prompt>.png, eval.json, logs/}
#
# Usage: bash scripts/ablation_sds.sh <device> <guidance> [<guidance> ...]
#   e.g. bash scripts/ablation_sds.sh 1 7.5 15 25 50 100
set -euo pipefail
cd "$(dirname "$0")/.."

DEVICE="$1"; shift
GUIDANCES=("$@")
STEPS=500
LR=1e-2
NEGATIVE_PROMPT="low quality"
ROOT="./outputs/ablation/sds_guidance"

# Same prompts as eval.sh
PROMPTS=(
  "A red bus driving on a desert road"
  "a boat in a river"
  "A cabin surrounded by forests"
  "A church beside a lake"
  "A villa close to the pool"
  "A castle next to a river"
  "A burger on the table"
  "A dog sitting on grass"
  "a cat sitting on a table"
  "A car on the road"
)

for gs in "${GUIDANCES[@]}"; do
  SAVE_DIR="$ROOT/gs_${gs}"
  mkdir -p "$SAVE_DIR/logs"
  echo "[*] SDS | guidance=$gs | steps=$STEPS | lr=$LR | device=$DEVICE -> $SAVE_DIR"
  for prompt in "${PROMPTS[@]}"; do
    key="${prompt// /_}"
    if [[ -f "$SAVE_DIR/$key.png" ]]; then
      echo "    skip (done): $prompt"; continue
    fi
    echo "    $prompt"
    python main.py \
      --prompt "$prompt" \
      --negative_prompt "$NEGATIVE_PROMPT" \
      --loss_type sds \
      --guidance_scale "$gs" \
      --steps "$STEPS" \
      --lr "$LR" \
      --device "$DEVICE" \
      --save_dir "$SAVE_DIR" \
      > "$SAVE_DIR/logs/$key.log" 2>&1
  done

  # main.py dumps intermediate <step>.png into save_dir; eval.py would score them as prompts
  find "$SAVE_DIR" -maxdepth 1 -type f -name '[0-9]*.png' -delete
  python eval.py --fdir1 "$SAVE_DIR" --device "$DEVICE" > "$SAVE_DIR/logs/eval.log" 2>&1
  python -c "import json; print('[*] gs=$gs CLIP =', json.load(open('$SAVE_DIR/eval.json'))['score'])"
done
