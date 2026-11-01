#!/bin/bash
# Task 2: train image FM with CFG, then sweep CFG scales and inference steps for the report.
# Usage: bash scripts/run_task2.sh [GPU_ID]
set -e
cd "$(dirname "$0")/.."
export PATH=/home_nfs/YuHsuanLee/miniconda3/envs/ddpm/bin:$PATH
# torch>=2.6 defaults torch.load to weights_only=True, but the ckpt pickles the whole network.
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export CUDA_VISIBLE_DEVICES=${1:-1}

python -m task2_image_flow_matching.train --use_cfg --gpu 0
RUN=$(ls -td results/cfg_fm-* | head -1)
echo "run dir: $RUN"

eval_one() {  # cfg_scale num_steps
    local out=$RUN/eval/cfg$1_steps$2
    local start=$(date +%s.%N)
    python -m image_common.sampling --use_cfg --gpu 0 --ckpt_path $RUN/last.ckpt \
        --save_dir $out --cfg_scale $1 --num_inference_steps $2 > /dev/null
    local secs=$(echo "$(date +%s.%N) - $start" | bc)
    local fid=$(python -m fid.measure_fid data/afhq/eval $out 2>/dev/null | grep '^FID:' | awk '{print $2}')
    echo "cfg=$1 steps=$2 fid=$fid sample_time_500imgs=${secs}s" | tee -a $RUN/eval/fid_results.txt
}

mkdir -p $RUN/eval
for cfg in 1.0 3.0 7.5; do eval_one $cfg 20; done
for steps in 10 50; do eval_one 7.5 $steps; done
echo "done"
