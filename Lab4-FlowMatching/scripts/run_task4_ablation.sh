#!/bin/bash
# Task 4 ablation: retrain the InstaFlow student with one setting changed, then evaluate.
# Usage: bash scripts/run_task4_ablation.sh {no_lpips|no_init} [GPU_ID]
set -eo pipefail
cd "$(dirname "$0")/.."
export PATH=/home_nfs/YuHsuanLee/miniconda3/envs/ddpm/bin:$PATH
# torch>=2.6 defaults torch.load to weights_only=True, but the ckpt pickles the whole network.
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export CUDA_VISIBLE_DEVICES=${2:-1}
DISTILL_DATA=data/afhq_instaflow
RF1_CKPT=results/rectified_fm_1-11-01-151854/last.ckpt

case $1 in
    no_lpips) EXTRA="--init_ckpt $RF1_CKPT" ;;
    no_init)  EXTRA="--use_lpips" ;;
    *) echo "unknown ablation: $1"; exit 1 ;;
esac

LOG=task4_ablation_$1.log
python -m task4_instaflow.train_instaflow --distill_data_path $DISTILL_DATA \
    --use_cfg --train_num_steps 100000 --gpu 0 $EXTRA 2>&1 | tee $LOG
# Several runs may write to results/ at once, so take the dir this run printed.
RUN=$(grep -m1 '^save_dir:' $LOG | awk '{print $2}')
echo "$1" > $RUN/ablation.txt
echo "run dir: $RUN"

python -m task4_instaflow.evaluate_instaflow --rf1_ckpt_path $RF1_CKPT \
    --instaflow_ckpt_path $RUN/last.ckpt --save_dir $RUN/eval --gpu 0

d=$RUN/eval/instaflow_1step
fid=$(python -m fid.measure_fid data/afhq/eval $d 2>/dev/null | grep '^FID:' | awk '{print $2}')
echo "$(basename $d) fid=$fid" | tee -a $RUN/eval/fid_results.txt
echo "done"
