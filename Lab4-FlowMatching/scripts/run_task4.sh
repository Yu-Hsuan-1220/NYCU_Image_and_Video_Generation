#!/bin/bash
# Task 4: distill the Task 3 1-rectified flow into a one-step InstaFlow student.
# Waits for the Task 3 pipeline (tmux session lab4_task3) to finish first.
# Usage: bash scripts/run_task4.sh [GPU_ID]
set -e
cd "$(dirname "$0")/.."
export PATH=/home_nfs/YuHsuanLee/miniconda3/envs/ddpm/bin:$PATH
# torch>=2.6 defaults torch.load to weights_only=True, but the ckpt pickles the whole network.
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export CUDA_VISIBLE_DEVICES=${1:-1}
DISTILL_DATA=data/afhq_instaflow

while tmux has-session -t lab4_task3 2>/dev/null; do sleep 60; done
grep -q "^done" task3_run.log || { echo "Task 3 did not finish cleanly"; exit 1; }
RF1_CKPT=$(ls -td results/rectified_fm_1-* | head -1)/last.ckpt
echo "1-RF teacher: $RF1_CKPT"

if [ ! -f $DISTILL_DATA/metadata.json ]; then
    python -m task4_instaflow.generate_instaflow_data --rf1_ckpt_path $RF1_CKPT \
        --num_samples 50000 --save_dir $DISTILL_DATA --use_cfg --cfg_scale 1.5 --save_images
fi

python -m task4_instaflow.train_instaflow --distill_data_path $DISTILL_DATA \
    --use_cfg --use_lpips --train_num_steps 100000 --init_ckpt $RF1_CKPT --gpu 0
RUN=$(ls -td results/instaflow-* | head -1)
echo "run dir: $RUN"

python -m task4_instaflow.evaluate_instaflow --rf1_ckpt_path $RF1_CKPT \
    --instaflow_ckpt_path $RUN/last.ckpt --save_dir $RUN/eval --gpu 0

for d in $RUN/eval/*/; do
    fid=$(python -m fid.measure_fid data/afhq/eval $d 2>/dev/null | grep '^FID:' | awk '{print $2}')
    echo "$(basename $d) fid=$fid" | tee -a $RUN/eval/fid_results.txt
done
echo "done"
