#!/usr/bin/env bash
# SDI main run + ablations, one process at a time:
#   outputs/sdi/                                   main (n_steps=10, interval=25)
#   outputs/ablation/sdi_inv_steps/n_<v>/          inversion_n_steps in {1,3,5,10,25}, interval=25
#   outputs/ablation/sdi_update_interval/i_<v>/    sdi_update_interval in {25,75,200,500,1000}, n_steps=10
# n_10 and i_25 share the main config, so they are copied from outputs/sdi instead of re-run.
#
# Usage: bash scripts/ablation_sdi.sh <device>
set -euo pipefail
cd "$(dirname "$0")/.."
DEVICE="$1"

bash scripts/run_sdi.sh "$DEVICE" outputs/sdi 10 25

for d in outputs/ablation/sdi_inv_steps/n_10 outputs/ablation/sdi_update_interval/i_25; do
  mkdir -p "$d" && cp outputs/sdi/*.png outputs/sdi/eval.json "$d/"
done

for n in 1 3 5 25; do
  bash scripts/run_sdi.sh "$DEVICE" "outputs/ablation/sdi_inv_steps/n_$n" "$n" 25
done
for i in 75 200 500 1000; do
  bash scripts/run_sdi.sh "$DEVICE" "outputs/ablation/sdi_update_interval/i_$i" 10 "$i"
done
echo "[*] ALL SDI RUNS FINISHED"
