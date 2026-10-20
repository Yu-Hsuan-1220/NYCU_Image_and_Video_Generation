#!/usr/bin/env bash
# VSD main run (Task 2) on all eval prompts, written to its own directory.
#
# Usage: bash scripts/run_vsd.sh <device> [save_dir] [prompt_index ...]
#   e.g. bash scripts/run_vsd.sh 1                              # all prompts -> outputs/vsd
#        bash scripts/run_vsd.sh 1 outputs/vsd 0 2 4 6 8        # subset, for splitting across processes
# Run eval separately once all prompts are done:
#   python eval.py --fdir1 outputs/vsd --device 1
set -euo pipefail
cd "$(dirname "$0")/.."

DEVICE="$1"
SAVE_DIR="${2:-./outputs/vsd}"
shift $(( $# >= 2 ? 2 : 1 ))
GUIDANCE=7.5
STEPS=500
LR=1e-2
LORA_LR=1e-4
LORA_LOSS_WEIGHT=1.0
LORA_RANK=4
NEGATIVE_PROMPT="low quality"

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
IDX=("$@")
if [[ ${#IDX[@]} -eq 0 ]]; then IDX=($(seq 0 $(( ${#PROMPTS[@]} - 1 )))); fi

mkdir -p "$SAVE_DIR/logs"
echo "[*] VSD | guidance=$GUIDANCE | steps=$STEPS | lr=$LR | lora_lr=$LORA_LR | rank=$LORA_RANK | device=$DEVICE -> $SAVE_DIR"
for i in "${IDX[@]}"; do
  prompt="${PROMPTS[$i]}"
  key="${prompt// /_}"
  if [[ -f "$SAVE_DIR/$key.png" ]]; then
    echo "    skip (done): $prompt"; continue
  fi
  echo "    $prompt"
  # Each prompt gets a private scratch dir so parallel processes don't clash on <step>.png
  tmp="$SAVE_DIR/tmp_$key"
  python main.py \
    --prompt "$prompt" \
    --negative_prompt "$NEGATIVE_PROMPT" \
    --loss_type vsd \
    --guidance_scale "$GUIDANCE" \
    --steps "$STEPS" \
    --lr "$LR" \
    --device "$DEVICE" \
    --save_dir "$tmp" \
    --lora_lr "$LORA_LR" \
    --lora_loss_weight "$LORA_LOSS_WEIGHT" \
    --lora_rank "$LORA_RANK" \
    > "$SAVE_DIR/logs/$key.log" 2>&1
  mv "$tmp/$key.png" "$SAVE_DIR/$key.png"
  mv "$tmp/run_config.json" "$SAVE_DIR/logs/$key.config.json"
  rm -rf "$tmp"
  echo "    done: $prompt"
done
