#!/bin/zsh
# Double-click to start B-Roll Librarian and open it in your browser.
# Close this Terminal window (or press Ctrl-C) to stop it.
cd /Users/nathan/broll-librarian || exit 1
PORT=8000
if lsof -nP -iTCP:$PORT -sTCP:LISTEN >/dev/null 2>&1; then
  echo "B-Roll Librarian is already running - opening it."
  open "http://127.0.0.1:$PORT"
  exit 0
fi
( sleep 6 && open "http://127.0.0.1:$PORT" ) &
echo "Starting B-Roll Librarian on http://127.0.0.1:$PORT ..."
echo "Leave this window open while you use it. Close it to stop."
exec ./.venv/bin/broll serve --port $PORT 2>&1 | grep -viE "loading weights|unauthenticated requests"
