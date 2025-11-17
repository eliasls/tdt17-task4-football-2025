#!/bin/bash
#SBATCH --partition=GPUQ
#SBATCH --account=ie-idi
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100"
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16
#SBATCH --job-name=yolo_inference
#SBATCH --output=logs/elias/train_%j.out
#SBATCH --error=logs/elias/train_%j.err


WORKSPACE=/cluster/work/eliasls/tdt17/task4

# Inference configuration - EDIT THESE
MODEL="runs/train/elias/yolo11s_1920_rect_e150/weights/best.pt"
INFER_ARGS="--data datasets/data.yaml"

# ============================================================

echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_JOB_NODELIST"
echo "Start: $(date)"
echo "=========================================="

cd "$WORKSPACE" || exit 1
source .venv/bin/activate

echo -e "\nGPU Info:"
nvidia-smi

echo -e "\n=========================================="
echo "Starting Inference"
echo "Model: $MODEL"
echo "Arguments: $INFER_ARGS"
echo "=========================================="

python scripts/elias/inference.py --model "$MODEL" $INFER_ARGS

EXIT_CODE=$?

echo -e "\n=========================================="
echo "End: $(date)"
echo "Exit code: $EXIT_CODE"
echo "=========================================="

if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ SUCCESS!"
else
    echo "✗ FAILED"
fi

exit $EXIT_CODE
