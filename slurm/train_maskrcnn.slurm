#!/bin/sh
#SBATCH --account=<account>
#SBATCH --job-name=maskrcnn_train
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
#SBATCH --mail-user=<email>
#SBATCH --mail-type=END,FAIL

WORKDIR=${SLURM_SUBMIT_DIR}
cd "${WORKDIR}"

echo "Job info:"
echo "  Directory : ${WORKDIR}"
echo "  Job name  : ${SLURM_JOB_NAME}"
echo "  Job ID    : ${SLURM_JOB_ID}"
echo "  Nodes     : ${SLURM_JOB_NODELIST}"
echo "  GPU       : ${SLURM_JOB_GPUS}"

module purge
source /cluster/work/eliasls/tdt17/task4/.venv/bin/activate

python mask_r_cnn.py \
    --mode train \
    --images-root data-mask-r-cnn/images \
    --train-ann data-mask-r-cnn/annotations/train.json \
    --val-ann data-mask-r-cnn/annotations/val.json \
    --output-dir runs/maskrcnn \
    --epochs 50 \
    --batch-size 2 \
    --num-workers 8 \
    --device cuda
