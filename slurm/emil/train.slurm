#!/bin/bash
#SBATCH --partition=GPUQ
#SBATCH --account=ie-idi
#SBATCH --gres=gpu:1
#SBATCH --constraint="v100|a100"
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16
#SBATCH --job-name=yolo_train
#SBATCH --output=logs/emil/train_%j.out
#SBATCH --error=logs/emil/train_%j.err

WORKSPACE=/cluster/work/emilkl/tdt17-task4-football-2025

MODEL="yolo11n.pt"
TRAIN_ARGS="--epochs 50 --imgsz 1920 --batch 8 --name yolo11n_1920"

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
echo "Starting YOLO Training"
echo "Model: $MODEL"
echo "Arguments: $TRAIN_ARGS"
echo "=========================================="

python scripts/emil/train.py --model "$MODEL" $TRAIN_ARGS

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
