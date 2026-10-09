#!/bin/sh
# Create the library on first start, then run the web app and the ingest worker.
set -eu

# On a public domain with no login in front (Cloudflare Access), the shared password is the only lock.
# Refuse to start without one rather than serve the library to anyone who finds the address.
if [ -n "${BROLL_DOMAIN:-}" ] && [ -z "${BROLL_ACCESS_PASSWORD:-}" ] && [ "${BROLL_NO_PASSWORD_OK:-}" != "1" ]; then
    echo "BROLL_DOMAIN is set but BROLL_ACCESS_PASSWORD is empty. Set a password in .env (or BROLL_NO_PASSWORD_OK=1 if a login sits in front)." >&2
    exit 1
fi

WORKSPACE="${BROLL_WORKSPACE:-library}"
CONFIG="${BROLL_HOME:-/data}/workspaces/${WORKSPACE}/config.yaml"

if [ ! -f "$CONFIG" ]; then
    echo "First start: creating the '${WORKSPACE}' library."
    broll init \
        --name "${BROLL_LIBRARY_NAME:-B-Roll Library}" \
        --id "$WORKSPACE" \
        --provider "${BROLL_PROVIDER:-gemini}" \
        --embedder "${BROLL_EMBEDDER:-gemini}" \
        --hosted
fi

# Listens on all interfaces inside the container. The compose file only publishes it to the server's
# own loopback address, so the only way in from outside is the tunnel (or the reverse proxy).
exec broll serve -w "$WORKSPACE" --host 0.0.0.0 --port 8000
