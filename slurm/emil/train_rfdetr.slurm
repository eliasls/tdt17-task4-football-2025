#!/bin/bash
#SBATCH --job-name=rfdetr_train
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100"
#SBATCH --time=10:00:00
#SBATCH --mem=64G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/emil/rfdetr_%j.out
#SBATCH --error=logs/emil/rfdetr_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=emilkl@stud.ntnu.no

echo "=========================================="
echo "RF-DETR Training Job"
echo "=========================================="
echo "Job started at: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "=========================================="
echo

# Navigate to project directory
cd /cluster/work/emilkl/tdt17-task4-football-2025 || exit 1

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# Verify environment
echo "Python: $(which python)"
echo "Python version: $(python --version)"
echo

# Check GPU
echo "Checking GPU availability..."
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
echo

# Configure distributed training environment for single GPU
# These variables are required by RF-DETR even for single-GPU training
echo "Configuring distributed training environment..."
export MASTER_ADDR=localhost
export MASTER_PORT=29500
export WORLD_SIZE=1
export RANK=0
export LOCAL_RANK=0
echo "  MASTER_ADDR: $MASTER_ADDR"
echo "  MASTER_PORT: $MASTER_PORT"
echo "  WORLD_SIZE: $WORLD_SIZE"
echo "  RANK: $RANK"
echo

# Optimize PyTorch CUDA memory allocation
echo "Configuring PyTorch memory optimization..."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "  PYTORCH_CUDA_ALLOC_CONF: $PYTORCH_CUDA_ALLOC_CONF"
echo

# Run training
echo "Starting RF-DETR training..."
echo "=========================================="
python scripts/emil/train_rfdetr.py \
    --model medium \
    --dataset-dir football_dataset_coco \
    --output-dir runs/rfdetr/football_medium \
    --epochs 100 \
    --batch-size 16 \
    --lr 1e-4 \
    --device cuda \
    --grad-accum-steps 1 \
    --checkpoint-interval 5 \
    --early-stopping-patience 15 \
    --num-workers 4

echo
echo "=========================================="
echo "Job completed at: $(date)"
echo "=========================================="
