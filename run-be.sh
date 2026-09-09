#!/usr/bin/env bash
# Start (or point you at) the skyform-backend tmux session running uvicorn.
#
# Pass tuning knobs as env vars, they are forwarded into the session:
#   SKYFORM_MAX_FRAMES=600 SKYFORM_FUSION_IMAGE_SIZE=512 ./run-be.sh
#
# Dynamic-object filtering (YOLOv8n masking of vehicles/people/animals
# before COLMAP) defaults ON here; disable for a run with:
#   SKYFORM_FILTER_DYNAMIC=0 ./run-be.sh
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
for v in SKYFORM_MAX_FRAMES SKYFORM_FUSION_IMAGE_SIZE SKYFORM_PATH_DIST_MULTIPLIER; do
  if [ -n "${!v:-}" ]; then
    EXTRA_ENV="$EXTRA_ENV $v=${!v}"
    echo "  forwarding $v=${!v}"
  fi
done

# Dynamic-object filtering defaults on; an explicit value from the
# environment still wins (e.g. SKYFORM_FILTER_DYNAMIC=0 ./run-be.sh).
FILTER_DYNAMIC="${SKYFORM_FILTER_DYNAMIC:-1}"
echo "  SKYFORM_FILTER_DYNAMIC=$FILTER_DYNAMIC"

CMD="cd '$REPO_ROOT/backend' \
&& source venv/bin/activate \
&& HF_HUB_DISABLE_XET=1 QT_QPA_PLATFORM=offscreen \
SKYFORM_FILTER_DYNAMIC=$FILTER_DYNAMIC${EXTRA_ENV} \
uvicorn app.main:app --host 127.0.0.1 --port 8000"

tmux new-session -d -s "$SESSION" -c "$REPO_ROOT/backend"
tmux send-keys -t "$SESSION" "$CMD" C-m

echo "Started '$SESSION'."
echo "Attach with:  tmux attach -t $SESSION"
