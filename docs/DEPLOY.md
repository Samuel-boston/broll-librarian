# Running this on a server

## What it needs

This is a long-running application with local state, not a set of serverless
functions. It will run on any box you can SSH into; it will not run on shared
web hosting, Vercel, Netlify, or anything that expects a stateless request
handler.

| Requirement | Why |
|---|---|
| Python 3.11+ | The app |
| ffmpeg and ffprobe on `PATH` | Frame extraction, probing, HEIC decoding |
| ~2 GB RAM | The local embedding model (all-MiniLM-L6-v2) sits in memory |
| ~3 GB disk for the install | Mostly PyTorch, pulled in by sentence-transformers |
| Persistent disk for `$BROLL_HOME` | SQLite database and thumbnails. Losing it loses the index, not the footage |
| One always-on process | The web UI and the ingest worker run together |
| Outbound HTTPS | Google Drive and the vision provider |

No inbound ports need opening to the world. The UI binds to `127.0.0.1` by
default; put it behind a reverse proxy with authentication if it must be
reachable from outside, because the app has no login of its own.

Footage is never stored on the server. Files are staged, analysed, uploaded to
Drive and then deleted; the database holds metadata and Drive file IDs.

## Install

```bash
git clone <repo-url> broll-librarian
cd broll-librarian
python3 -m venv .venv
.venv/bin/pip install -e ".[gemini,drive,shots,embeddings-local,web]"
.venv/bin/broll doctor          # checks ffmpeg, the database and credentials
```

## Configure

```bash
.venv/bin/broll init -n "Client Name"
.venv/bin/broll drive login -w <workspace-id>
.venv/bin/broll serve -w <workspace-id> --host 127.0.0.1 --port 8000
```

Keys live in `$BROLL_HOME/.env` (owner-only permissions) and never in the
database or `config.yaml`. `$BROLL_HOME` defaults to `~/.broll`.

## Keeping it running

Anything that restarts a process will do - systemd, pm2, Docker, launchd. One
instance per machine per workspace: two workers on the same database is
supported (jobs are claimed atomically) but buys nothing.

```ini
# /etc/systemd/system/broll.service
[Unit]
Description=B-Roll Librarian
After=network-online.target

[Service]
User=broll
WorkingDirectory=/opt/broll-librarian
Environment=BROLL_HOME=/var/lib/broll
ExecStart=/opt/broll-librarian/.venv/bin/broll serve -w client --host 127.0.0.1 --port 8000
Restart=always

[Install]
WantedBy=multi-user.target
```

## Sharing a box with other software

Fine. It is one Python process plus ffmpeg subprocesses, and it is idle
whenever nothing is being indexed. The things to watch:

* RAM - the embedding model is the only heavy resident, about 2 GB with room
  to work.
* CPU during ingest - ffmpeg decoding is the busy part. Lower
  `ingest.concurrency` if it crowds out whatever else lives on the box.
* Disk for `$BROLL_HOME` - the database and thumbnails, a few MB per hundred
  clips. Staging is transient.
