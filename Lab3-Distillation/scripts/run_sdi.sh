#!/usr/bin/env bash
# SDI on all eval prompts with a given (inversion_n_steps, sdi_update_interval),
# written to its own directory, then CLIP-evaluated.
#
# Usage: bash scripts/run_sdi.sh <device> <save_dir> <inversion_n_steps> <sdi_update_interval>
#   e.g. bash scripts/run_sdi.sh 0 outputs/sdi 10 25
set -euo pipefail
cd "$(dirname "$0")/.."

DEVICE="$1"
SAVE_DIR="$2"
INVERSION_N_STEPS="$3"
SDI_UPDATE_INTERVAL="$4"
GUIDANCE=7.5
STEPS=1000
LR=0.005
INVERSION_GUIDANCE_SCALE=-7.5
INVERSION_ETA=0.3
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

mkdir -p "$SAVE_DIR/logs"
echo "[*] SDI | n_steps=$INVERSION_N_STEPS | interval=$SDI_UPDATE_INTERVAL | guidance=$GUIDANCE | lr=$LR | steps=$STEPS | device=$DEVICE -> $SAVE_DIR"
for prompt in "${PROMPTS[@]}"; do
  key="${prompt// /_}"
  if [[ -f "$SAVE_DIR/$key.png" ]]; then
    echo "    skip (done): $prompt"; continue
  fi
  python main.py \
    --prompt "$prompt" \
    --negative_prompt "$NEGATIVE_PROMPT" \
    --loss_type sdi \
    --guidance_scale "$GUIDANCE" \
    --steps "$STEPS" \
    --lr "$LR" \
    --device "$DEVICE" \
    --save_dir "$SAVE_DIR" \
    --inversion_n_steps "$INVERSION_N_STEPS" \
    --inversion_guidance_scale "$INVERSION_GUIDANCE_SCALE" \
    --inversion_eta "$INVERSION_ETA" \
    --sdi_update_interval "$SDI_UPDATE_INTERVAL" \
    > "$SAVE_DIR/logs/$key.log" 2>&1
  echo "    done: $prompt"
done

# main.py dumps intermediate <step>.png into save_dir; eval.py would score them as prompts
find "$SAVE_DIR" -maxdepth 1 -type f -name '[0-9]*.png' -delete
python eval.py --fdir1 "$SAVE_DIR" --device "$DEVICE" > "$SAVE_DIR/logs/eval.log" 2>&1
python -c "import json; print('[*] CLIP $SAVE_DIR =', json.load(open('$SAVE_DIR/eval.json'))['score'])"
