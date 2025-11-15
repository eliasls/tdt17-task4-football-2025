#!/bin/bash
#SBATCH --partition=GPUQ
#SBATCH --account=ie-idi
#SBATCH --gres=gpu:A100:1        # Request 1 A100 GPU
#SBATCH --time=10:00:00           # 10 hours (safe for 11k images)
#SBATCH --nodes=1
#SBATCH --mem=80G                 # 80GB RAM for large dataset
#SBATCH --cpus-per-task=16        # 16 CPU cores
#SBATCH --job-name=yolo11_football_complete
#SBATCH --output=logs/yolo11_complete_%j.out
#SBATCH --error=logs/yolo11_complete_%j.err
#SBATCH --mail-user=emilkl@stud.ntnu.no
#SBATCH --mail-type=BEGIN,END,FAIL

# Print job information
echo "=========================================="
echo "Job Information"
echo "=========================================="
echo "Job ID:        $SLURM_JOB_ID"
echo "Job name:      $SLURM_JOB_NAME"
echo "Node:          $SLURM_JOB_NODELIST"
echo "Start time:    $(date)"
echo "CPUs:          $SLURM_CPUS_PER_TASK"
echo "Memory:        ${SLURM_MEM_PER_NODE}MB"
echo "=========================================="

# Navigate to project
cd /cluster/work/emilkl/tdt17-task4-football-2025 || exit 1

# Create logs directory
mkdir -p logs

# Activate environment
echo -e "\nActivating virtual environment..."
source .venv/bin/activate

# Check GPU
echo -e "\n=========================================="
echo "GPU Information"
echo "=========================================="
nvidia-smi
echo ""

# Print environment
echo "=========================================="
echo "Python Environment"
echo "=========================================="
which python
python --version
echo ""
pip list | grep -E "torch|ultralytics|numpy|opencv"

# Run training
echo -e "\n=========================================="
echo "Starting YOLO11 Training"
echo "=========================================="
echo "Dataset: 11,168 images (5 matches)"
echo "Expected time: 5-6 hours on A100"
echo "=========================================="
echo ""

python train_yolo11.py

EXIT_CODE=$?

# Print completion info
echo -e "\n=========================================="
echo "Job Completed"
echo "=========================================="
echo "End time:   $(date)"
echo "Exit code:  $EXIT_CODE"
echo "=========================================="

if [ $EXIT_CODE -eq 0 ]; then
    echo -e "\n✓ SUCCESS! Training completed"
    echo "Results: football_yolo11/complete_dataset_v1/"
    echo ""
    echo "Best model: football_yolo11/complete_dataset_v1/weights/best.pt"
    echo "Metrics: football_yolo11/complete_dataset_v1/results.csv"
    echo "Plots: football_yolo11/complete_dataset_v1/results.png"
else
    echo -e "\n✗ FAILED with exit code $EXIT_CODE"
    echo "Check error log: logs/yolo11_complete_${SLURM_JOB_ID}.err"
fi

exit $EXIT_CODE