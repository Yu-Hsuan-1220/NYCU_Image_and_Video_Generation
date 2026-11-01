#!/bin/bash
# Task 3: generate reflow pairs from the Task 2 model, train 1-rectified flow,
# then compare base FM vs rectified flow at 5/10/20 steps (speed + FID).
# Usage: bash scripts/run_task3.sh BASE_CKPT [GPU_ID]
set -e
cd "$(dirname "$0")/.."
export PATH=/home_nfs/YuHsuanLee/miniconda3/envs/ddpm/bin:$PATH
# torch>=2.6 defaults torch.load to weights_only=True, but the ckpt pickles the whole network.
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export CUDA_VISIBLE_DEVICES=${2:-1}
BASE_CKPT=$1
REFLOW_DATA=data/afhq_reflow

if [ ! -f $REFLOW_DATA/metadata.json ]; then
    python -m task3_rectified_flow.generate_reflow_data --ckpt_path $BASE_CKPT \
        --num_samples 50000 --save_dir $REFLOW_DATA --use_cfg --num_inference_steps 20 --save_images
fi

python -m task3_rectified_flow.train_rectified --reflow_data_path $REFLOW_DATA \
    --use_cfg --reflow_iteration 1 --gpu 0
RUN=$(ls -td results/rectified_fm_1-* | head -1)
echo "run dir: $RUN"

python -m task3_rectified_flow.evaluate_rectified --base_ckpt_path $BASE_CKPT \
    --rectified_ckpt_path $RUN/last.ckpt --save_dir $RUN/eval --step_counts 5 10 20 --gpu 0

for d in $RUN/eval/*_steps*/; do
    fid=$(python -m fid.measure_fid data/afhq/eval $d 2>/dev/null | grep '^FID:' | awk '{print $2}')
    echo "$(basename $d) fid=$fid" | tee -a $RUN/eval/fid_results.txt
done
echo "done"
