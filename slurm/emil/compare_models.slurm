#!/bin/bash
#SBATCH --partition=GPUQ
#SBATCH --account=ie-idi
#SBATCH --gres=gpu:1
#SBATCH --constraint="v100|a100"
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --job-name=compare_models
#SBATCH --output=logs/emil/compare_%j.out
#SBATCH --error=logs/emil/compare_%j.err

WORKSPACE=/cluster/work/emilkl/tdt17-task4-football-2025

MODEL1="runs/train/emil/new_data/yolo11n_640/weights/best.pt"
MODEL2="runs/train/emil/new_data/yolo11n_1920/weights/best.pt"
NAME1="YOLO11n-640"
NAME2="YOLO11n-1920"
SPLIT="val"
NUM_SAMPLES=20
OUTPUT_DIR="model_comparison/640v1920"

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
echo "Starting Model Comparison"
echo "Model 1: $NAME1 ($MODEL1)"
echo "Model 2: $NAME2 ($MODEL2)"
echo "=========================================="

python scripts/emil/compare_models.py \
    --model1 "$MODEL1" \
    --model2 "$MODEL2" \
    --name1 "$NAME1" \
    --name2 "$NAME2" \
    --split "$SPLIT" \
    --num-samples "$NUM_SAMPLES" \
    --output-dir "$OUTPUT_DIR"

EXIT_CODE=$?

echo -e "\n=========================================="
echo "End: $(date)"
echo "Exit code: $EXIT_CODE"
echo "=========================================="

if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ SUCCESS!"
    echo "Results saved to: $OUTPUT_DIR"
else
    echo "✗ FAILED"
fi

exit $EXIT_CODE
