#!/bin/bash
#SBATCH --partition=GPUQ
#SBATCH --account=ie-idi
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100"
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16
#SBATCH --job-name=yolo_train
#SBATCH --output=logs/elias/train_%j.out
#SBATCH --error=logs/elias/train_%j.err

# ============================================================
# YOLO Training Job - Simple and Reusable
# ============================================================
#
# To customize training, edit the MODEL and TRAIN_ARGS below:
#
# MODEL examples:
#   - yolo11n.pt (nano, fastest)
#   - yolo11s.pt (small)
#   - yolo11m.pt (medium)
#   - yolo11l.pt (large)
#
# Common arguments:
#   --epochs 50         Number of training epochs
#   --imgsz 1280        Image size (640, 1280, etc.)
#   --batch 16          Batch size (adjust for GPU memory)
#   --name exp_name     Experiment name
#   --patience 10       Early stopping patience
#   --cache             Cache images in RAM for faster training
#
# ============================================================

WORKSPACE=/cluster/work/eliasls/tdt17/task4

# Training configuration - EDIT THESE
MODEL="yolo11s.pt"
TRAIN_ARGS="--data datasets/data.yaml --epochs 150 --imgsz 1920 --batch 4 --rect --scale 0.2 --patience 50 --workers 8 --name yolo11s_1920_rect_e150"




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
echo "Starting YOLO Training"
echo "Model: $MODEL"
echo "Arguments: $TRAIN_ARGS"
echo "=========================================="

python scripts/elias/train.py --model "$MODEL" $TRAIN_ARGS

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
