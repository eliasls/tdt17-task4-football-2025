#!/bin/sh
#SBATCH --account=ie-idi
#SBATCH --job-name=maskrcnn_debug
#SBATCH --time=0-02:00:00

#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --output=output.txt      
#SBATCH --error=output.err
#SBATCH --mail-user=eliasls@stud.ntnu.no
#SBATCH --mail-type=END,FAIL

WORKDIR=${SLURM_SUBMIT_DIR}
cd "${WORKDIR}"

echo "Debug job"
echo "  Directory : ${WORKDIR}"
echo "  Job name  : ${SLURM_JOB_NAME}"
echo "  Job ID    : ${SLURM_JOB_ID}"
echo "  Nodes     : ${SLURM_JOB_NODELIST}"

module purge
module load Anaconda3/2024.02-1
source /cluster/work/eliasls/tdt17/task4/.venv/bin/activate

python mask_r_cnn.py \
    --mode train \
    --images-root data-mask-r-cnn/images \
    --train-ann data-mask-r-cnn/annotations/train.json \
    --val-ann data-mask-r-cnn/annotations/val.json \
    --output-dir runs/debug_maskrcnn \
    --epochs 1 \
    --batch-size 1 \
    --num-workers 4 \
    --lr 0.001 \
    --device cuda
