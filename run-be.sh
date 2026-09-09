#!/usr/bin/env bash
# Start (or point you at) the skyform-backend tmux session running uvicorn.
#
# Pass tuning knobs as env vars, they are forwarded into the session:
#   SKYFORM_MAX_FRAMES=600 SKYFORM_FUSION_IMAGE_SIZE=512 ./run-be.sh
#
# Dynamic-object filtering (YOLOv8n masking of vehicles/people/animals
# before COLMAP) defaults OFF here: it loads torch + ultralytics inside
# the COLMAP SfM stage and, on a small-RAM box, that extra pressure is
# enough to get COLMAP OOM-killed mid-stage (pipeline dies at ~24%).
# Turn it back on for a run with:
#   SKYFORM_FILTER_DYNAMIC=1 ./run-be.sh
set -euo pipefail

SESSION="skyform-backend"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "Session '$SESSION' is already running."
  echo "Attach with:  tmux attach -t $SESSION"
  exit 0
fi

# Forward only the SKYFORM_* knobs that were actually passed in.
EXTRA_ENV=""
for v in SKYFORM_FUSION_IMAGE_SIZE SKYFORM_PATH_DIST_MULTIPLIER \
         SKYFORM_SFM_MAX_IMAGE_SIZE SKYFORM_SFM_NUM_THREADS; do
  if [ -n "${!v:-}" ]; then
    EXTRA_ENV="$EXTRA_ENV $v=${!v}"
    echo "  forwarding $v=${!v}"
  fi
done

# Keep the frame count bounded by default. Exhaustive COLMAP matching is
# O(n^2) in pairs and this box runs it on the CPU with only a few GB of
# RAM; 60 frames from a 60 s pass is 1 fps - ample overlap for SfM - and
# keeps stage 3 well inside its memory + time budget. Bump it for a
# higher-quality run on a bigger machine: SKYFORM_MAX_FRAMES=200 ./run-be.sh
MAX_FRAMES="${SKYFORM_MAX_FRAMES:-60}"
echo "  SKYFORM_MAX_FRAMES=$MAX_FRAMES"

# COLMAP SfM low-memory profile (capped features / matches / threads,
# no affine-shape or domain-size pooling). On by default; override with
# SKYFORM_SFM_LOW_MEM=0 ./run-be.sh on a machine with RAM to spare.
LOW_MEM="${SKYFORM_SFM_LOW_MEM:-1}"
echo "  SKYFORM_SFM_LOW_MEM=$LOW_MEM"

# Dynamic-object filtering defaults OFF (see header); an explicit value
# from the environment still wins (SKYFORM_FILTER_DYNAMIC=1 ./run-be.sh).
FILTER_DYNAMIC="${SKYFORM_FILTER_DYNAMIC:-0}"
echo "  SKYFORM_FILTER_DYNAMIC=$FILTER_DYNAMIC"

CMD="cd '$REPO_ROOT/backend' \
&& source venv/bin/activate \
&& HF_HUB_DISABLE_XET=1 QT_QPA_PLATFORM=offscreen \
SKYFORM_FILTER_DYNAMIC=$FILTER_DYNAMIC \
SKYFORM_MAX_FRAMES=$MAX_FRAMES \
SKYFORM_SFM_LOW_MEM=$LOW_MEM${EXTRA_ENV} \
uvicorn app.main:app --host 127.0.0.1 --port 8000"

tmux new-session -d -s "$SESSION" -c "$REPO_ROOT/backend"
tmux send-keys -t "$SESSION" "$CMD" C-m

echo "Started '$SESSION'."
echo "Attach with:  tmux attach -t $SESSION"
