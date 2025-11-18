#!/bin/bash
#SBATCH --partition=GPUQ
#SBATCH --account=ie-idi
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|v100"
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --job-name=track_eval
#SBATCH --output=logs/elias/track/track_eval_%j.out
#SBATCH --error=logs/elias/track/track_eval_%j.err


WORKSPACE=/cluster/work/eliasls/tdt17/task4
TRACKEVAL=/cluster/work/eliasls/tdt17/task4/TrackEval

SEQ_NAME="RBK-AALESUND"
TRACKER_NAME="yolo_botsort"

INPUT_VIDEO=/cluster/projects/vc/courses/TDT17/other/Football2025/RBK-AALESUND/aalesund.mp4
MODEL_PATH=$WORKSPACE/models/yolo11s_1920_rect_e150_best.pt
TRACKER_CFG=$WORKSPACE/configs/botsort.yaml

MOT_PRED_PATH=$TRACKEVAL/data/trackers/mot_challenge/FOOTBALL-test/$TRACKER_NAME/data/${SEQ_NAME}.txt
JSON_PRED_PATH=runs/track/elias/json/${SEQ_NAME}.json

echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $SLURM_JOB_NODELIST"
echo "Start:  $(date)"
echo "=========================================="

# Tracking
cd "$WORKSPACE" || exit 1
source .venv/bin/activate

echo -e "\nGPU Info:"
nvidia-smi

echo -e "\n=== RUNNING TRACKING ==="
echo "Video:   $INPUT_VIDEO"
echo "Model:   $MODEL_PATH"
echo "Tracker: $TRACKER_CFG"
echo "json out: $JSON_PRED_PATH"
echo "MOT out: $MOT_PRED_PATH"


python scripts/elias/tracking.py \
  --input "$INPUT_VIDEO" \
  --model "$MODEL_PATH" \
  --tracker "$TRACKER_CFG" \
  --project runs/track/elias \
  --name "$SEQ_NAME" \
  --save-predictions-json "$JSON_PRED_PATH" \
  --save-predictions-mot "$MOT_PRED_PATH"

TRACK_EXIT=$?
echo "Tracking exit code: $TRACK_EXIT"
if [ $TRACK_EXIT -ne 0 ]; then
    echo "Tracking failed, aborting."
    exit $TRACK_EXIT
fi

# HOTA evaluation with TrackEval
echo -e "\n=== RUNNING TRACKEVAL (HOTA) ==="

cd "$TRACKEVAL" || exit 1

python scripts/run_mot_challenge.py \
  --BENCHMARK FOOTBALL \
  --SPLIT_TO_EVAL test \
  --TRACKERS_TO_EVAL yolo_botsort \
  --METRICS HOTA CLEAR Identity \
  --DO_PREPROC False

EVAL_EXIT=$?
echo "TrackEval exit code: $EVAL_EXIT"

echo -e "\n=== DONE ==="
echo "End: $(date)"
echo "=========================================="

exit $EVAL_EXIT
