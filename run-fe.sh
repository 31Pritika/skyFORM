#!/usr/bin/env bash
# Start (or point you at) the skyform-frontend tmux session running the Vite dev server.
set -euo pipefail

SESSION="skyform-frontend"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "Session '$SESSION' is already running."
  echo "Attach with:  tmux attach -t $SESSION"
  exit 0
fi

tmux new-session -d -s "$SESSION" -c "$REPO_ROOT/frontend"
tmux send-keys -t "$SESSION" "cd '$REPO_ROOT/frontend' && npm run dev" C-m

echo "Started '$SESSION'."
echo "Attach with:  tmux attach -t $SESSION"
