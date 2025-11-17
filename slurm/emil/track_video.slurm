#!/bin/bash
#SBATCH --job-name=track_video
#SBATCH --partition=GPUQ
#SBATCH --account=ie-idi
#SBATCH --gres=gpu:1
#SBATCH --constraint="v100|a100"
#SBATCH --time=02:00:00
#SBATCH --mem=16G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/emil/track_%j.out
#SBATCH --error=logs/emil/track_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=emilkl@stud.ntnu.no

echo "=========================================="
echo "Video Tracking Job"
echo "=========================================="
echo "Job started at: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
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

# Example: Track RBK-HamKam match (validation set)
INPUT_VIDEO="/cluster/projects/vc/courses/TDT17/other/Football2025/RBK-HamKam/rbk_hamkam.mp4"
OUTPUT_VIDEO="track_output/rbk_hamkam_tracked.mp4"

# Create output directory
mkdir -p track_output

# Run tracking
echo "Running tracking..."
echo "=========================================="
python scripts/emil/track_video.py \
    --input "$INPUT_VIDEO" \
    --output "$OUTPUT_VIDEO" \
    --model runs/train/emil/new_data/yolo11n_1920/weights/best.pt

echo
echo "=========================================="
echo "Job completed at: $(date)"
echo "=========================================="
