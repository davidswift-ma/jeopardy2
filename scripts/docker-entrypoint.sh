#!/bin/sh
# Starts as root only long enough to make the memory directory writable, then
# drops to the unprivileged `app` user for the actual server.
#
# Why this is needed: a Fly.io volume (and a fresh Docker named volume) is
# mounted root-owned, over whatever the image had at that path, so a chown in
# the Dockerfile does not survive the mount. Without this the app user cannot
# create memory.sqlite3 and every memory write fails.
set -e

if [ "$(id -u)" = "0" ]; then
    mem_dir="$(dirname "${MEMORY_DB_PATH:-/app/data/memory.sqlite3}")"
    mkdir -p "$mem_dir"
    chown app:app "$mem_dir"
    exec setpriv --reuid=app --regid=app --init-groups "$@"
fi
exec "$@"
