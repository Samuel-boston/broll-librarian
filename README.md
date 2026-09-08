# B-Roll Librarian

Turns a pile of unsorted B-roll into a searchable, auto-organised library backed
by Google Drive.

Drop in videos (or point it at a Drive folder). In the background it splits each
video into shots, analyses them with a vision model, writes rich structured
metadata into a local database, uploads the files into a meticulously organised
Drive tree, and then lets you find footage two ways:

1. **Search** — natural language plus filters, returning thumbnails and Drive links.
2. **Transcript matching** — paste a transcript, get suggested B-roll per beat,
   exportable as a timeline for Premiere Pro or DaVinci Resolve.

**The video files never live in the database.** The database stores metadata and
a Google Drive file ID. Everything else follows from that constraint.

---

## Status

| Milestone | Scope | State |
|---|---|---|
| **M1** | Config, registry + workspace schema, provider abstraction, frame extraction, analysis, `broll analyse` | **done** |
| **M2** | Shot detection, job queue, batch indexing, embeddings, FTS5 + vector search | **done** |
| **M3** | Drive OAuth, upload, taxonomy, shortcut tree, `reorganise`, `--dry-run` | **built; verified against a mock Drive, not yet against live Drive** |
| M4 | Web UI: ingest, queue view, search | not started |
| M5 | Transcript matching, FCP7 XML / EDL / CSV export | not started |
| M6 | Review queue, remaining providers, cost reporting, vocabulary management | not started |

---

## Requirements

- Python 3.11+
- **ffmpeg and ffprobe** on `PATH` (`brew install ffmpeg`, `apt install ffmpeg`,
  `winget install Gyan.FFmpeg`). `broll doctor` checks for them.
- An API key for one vision provider (Gemini, Anthropic, or OpenAI).

## Install

```bash
git clone <this repo> && cd broll-librarian
python3 -m venv .venv && source .venv/bin/activate
pip install -e '.[gemini]'          # or '.[anthropic]' / '.[openai]'
```

Optional extras, added as you need them:

| Extra | Pulls in | Needed for |
|---|---|---|
| `gemini` / `anthropic` / `openai` | that provider's SDK | analysis |
| `shots` | PySceneDetect + OpenCV | shot detection (M2) |
| `embeddings-local` | sentence-transformers (**~2GB, pulls torch**) | local embeddings (M2) |
| `drive` | Google API client | Drive (M3) |
| `web` | FastAPI + uvicorn | web UI (M4) |
| `dev` | pytest | tests |

If `embeddings-local` is not installed, the tool falls back to the configured
provider's embedding API and says so at startup.

## Quickstart

```bash
cp .env.example .env        # then put your provider key in it
broll init --name "My Library" --provider gemini
broll doctor

broll analyse path/to/clip.mp4         # print the analysis as JSON, write nothing
broll index path/to/footage --dry-run  # show the plan, touch nothing
broll index path/to/footage            # queue it and work the queue
broll status                           # queue, counts, cost so far and remaining
broll search "calm beach wide shot golden hour" --limit 20
broll search "" --shot-type aerial --clean --json
```

`broll index` queues one job per file and then works the queue with four
concurrent workers. **Kill it at any point and run `broll index` or `broll work`
again** — jobs are claimed atomically in SQLite, a job left mid-flight is
requeued, and a source that was half-ingested resumes at the first shot that is
not yet indexed. Nothing is analysed twice.

Every UI action is also a CLI command. The CLI is the real interface; the web UI
is a client of it.

## Where things live

```
~/.broll/
  registry.db                     # the workspace list, and nothing else
  .env                            # optional, machine-wide secrets
  workspaces/<id>/
    config.yaml                   # non-secret settings (see config.example.yaml)
    library.db                    # this workspace's metadata
    thumbnails/                   # 640px JPEGs for the search UI
    tmp/, staging/                # working files, safe to delete
    drive_token.json              # OAuth token (M3)
```

Override the root with `BROLL_HOME`. Pick a workspace per command with
`--workspace`, or set `BROLL_WORKSPACE`; with exactly one workspace it is implied.

**Secrets are never written to `config.yaml` or the database**, and are never
logged. They come from the environment or a `.env` file.

## Choosing a provider

| Provider | Default model | Notes |
|---|---|---|
| `gemini` | `gemini-2.5-flash` | **The default.** Lowest cost per image, which is what dominates the bill when you are analysing thousands of shots at 3 frames each. |
| `anthropic` | `claude-opus-5` | Tends to return the richest structured descriptions, at the highest cost. Set `provider.vision_model: claude-haiku-4-5` for a cheaper Claude. |
| `openai` | `gpt-4.1-mini` | Middle of the road. |
| `mock` | — | Deterministic offline output. For tests and pipeline smoke runs with no key. |

Switching is one line of config plus a key. Nothing in the codebase imports a
provider module directly — everything goes through the provider registry.

**Embeddings are separate and local by default** (`all-MiniLM-L6-v2`, 384
dimensions). Search quality therefore does not depend on which vision provider
you pay for, and re-embedding the library after a prompt change is free.

### What it costs

Cost is dominated by images: 3 frames per shot, each downscaled to 768px on the
long edge. Per-shot estimates at list prices, with the shared prompt (~2.4k
input tokens, mostly the controlled vocabularies) and ~350 output tokens:

| Provider / model | ≈ per shot | ≈ 1,000 shots | ≈ 5,000 shots |
|---|---|---|---|
| gemini-2.5-flash | $0.0019 | $1.86 | $9.32 |
| gpt-4.1-mini | $0.0025 | $2.48 | $12.40 |
| claude-haiku-4-5 | $0.0055 | $5.50 | $27.50 |
| claude-opus-5 | $0.0275 | $27.50 | $137.50 |

These are estimates from published list prices, computed by
`analysis/providers/*.PRICING`. Real costs are logged per job and shown by
`broll status` — trust those over this table.

## How search works

Hybrid retrieval, because neither half is good enough alone:

* **Keyword** — FTS5 over a single denormalised `shots.search_text` column
  (caption + facets + tags), recomputed in exactly one place. Every user token is
  quoted before it reaches FTS5, so `person's wide-shot`, `NEAR(a b)` and a stray
  `"` are searches, not syntax errors.
* **Vector** — KNN over shot embeddings. Because a vector index cannot pre-filter
  against a join, the vector side over-fetches `limit x 10` and applies the
  filters afterwards; the keyword side applies them as ordinary SQL.
* **Fusion** — reciprocal rank fusion (k=60). Simpler than normalising raw
  scores, and better.

An empty query with filters is a browse, newest first.

### Vector backends

The spec calls for `sqlite-vec`, which is a loadable SQLite extension. Some
Python builds — including the python.org macOS framework build this was
developed against — compile `sqlite3` **without** extension loading, so
`sqlite-vec` cannot be loaded at all, and a client machine is not where you want
to discover that. So both backends ship and the right one is chosen
automatically:

| Backend | When | Notes |
|---|---|---|
| `sqlite-vec` | the interpreter can load extensions | a `vec0` virtual table, as specced |
| `numpy` | it cannot | exact cosine KNN over blobs; ~milliseconds at tens of thousands of shots |

`broll status` reports which one is in use. Both fix the dimension at creation,
so changing the embedding model needs `broll reembed`, which drops and rebuilds
the table.

## The analysis contract

Search quality is entirely determined by this. Every provider must return an
`AnalysisResult` (see `src/broll/analysis/schema.py`), using its native
structured-output mode — no free-text parsing anywhere.

Closed enums: `shot_type`, `camera_movement`, `time_of_day`, `colour_profile`,
`people_count`, `pace`. Controlled vocabularies (130 subjects, 107 actions, 99
settings, 82 moods, plus editorial uses and quality flags) for `subjects`,
`action`, `setting`, `mood`, `usable_for`, `quality_flags`. One open field,
`tags`, where synonyms and specifics belong — it is what keyword search hits.

Out-of-vocabulary terms are **kept on the row** and also recorded in
`vocabulary_candidates`, so you can promote the ones that keep coming up rather
than losing them. Without controlled vocabularies the Drive tree becomes a swamp
of near-synonyms (`meditating` / `meditation` / `person meditating`) and
everything downstream degrades.

The prompt is one shared template in `src/broll/analysis/prompt.py`, versioned
by `PROMPT_VERSION` and stored on each row as `analysis_version`, so stale rows
can be found and re-analysed when the prompt improves.

## How the Drive tree is organised

One canonical copy of each file, plus a faceted tree of **shortcuts**. A shortcut
is a native Drive object pointing at a file without duplicating its bytes, so one
clip can appear in a dozen browsable places at zero storage cost.

```
B-Roll/
├── _Library/2026-09/beach_meditating_golden_hour_wide_a1b2c3d4.mp4   ← the real files
├── By Subject/Person/
├── By Action/Meditating/Beach/          ← third level, once 5+ clips share it
├── By Setting/Beach/Golden Hour/
├── By Mood/Calm/
├── By Shot Type/Wide/
├── By Camera Movement/Static/
├── By Time of Day/Golden Hour/
├── By Colour/Warm/
├── By Use/Establishing Shots/
└── _Needs Review/                       ← low confidence or quality-flagged
```

* **Third-level folders are earned.** A combination gets its own subfolder only
  once `taxonomy.min_clips_for_subfolder` (default 5) clips share it; below that,
  clips sit at the second level and are promoted later as the library grows.
  Without this you get thousands of one-item folders.
* **Levels are capped** at `taxonomy.max_folders_per_level` (default 40); the long
  tail is grouped under `Other/`.
* **A multi-shot source** gets the union of its shots' facets, and a shortcut
  whose name carries the timecode when it is not the primary shot —
  `..._at_0m42s.mp4` — so a browsing editor knows where to look inside the file.
* **v1 never splits or re-encodes footage.** Editors want the original file.
* `broll organise` is idempotent: it reconciles desired against actual state, so
  a second run performs zero writes. `broll reorganise` rebuilds the whole tree
  from the database — the escape hatch when the taxonomy changes.
* Every Drive-writing command takes `--dry-run`, which prints the full plan
  (every file, every destination, every shortcut) and touches nothing.
* The organiser **never deletes footage**. The only thing it deletes is a
  shortcut it created that the taxonomy no longer justifies, and it refuses if
  the object turns out not to be a shortcut.

```bash
broll drive login                 # once per workspace
broll index ~/footage --organise  # index, then file into Drive
broll organise --dry-run          # show what would change
broll reorganise                  # rebuild the tree from the database
```

## Google Drive setup (needed from M3)

This is the biggest onboarding hurdle, so it gets a full walkthrough.

**Why it is fiddly.** The app needs the full `https://www.googleapis.com/auth/drive`
scope. The narrower `drive.file` scope only grants access to files the app itself
created, which means it could not list an existing folder, adopt a Drive folder
you already have, or create shortcuts in folders it did not create. Google
classes the full scope as restricted, so *publishing* an app that uses it
requires their verification and a security assessment. Self-hosting sidesteps
that entirely: **you create your own Google Cloud project and use the app in
testing mode against your own account.** No verification, no review.

### Step by step, for someone who has never opened a cloud console

1. Go to <https://console.cloud.google.com/> and sign in with the Google account
   that owns the Drive you want to organise.
2. Top-left, click the project dropdown → **New Project**. Name it something like
   `broll-librarian`. Click **Create**, then select it in the dropdown.
3. In the search bar at the top, type **Google Drive API** and open it. Click
   **Enable**. Wait for it to finish.
4. In the left menu, open **APIs & Services → OAuth consent screen**.
   - User type: **External**. Click **Create**.
   - App name: `B-Roll Librarian`. User support email: your address.
   - Developer contact: your address. **Save and Continue**.
   - On the **Scopes** step, click **Add or Remove Scopes**, paste
     `https://www.googleapis.com/auth/drive` into the filter box, tick it, then
     **Update** and **Save and Continue**.
   - On the **Test users** step, click **Add Users** and add your own Google
     address. This is what lets you use a restricted scope without verification.
     **Save and Continue**, then **Back to Dashboard**.
   - Leave the app in **Testing**. Do not click "Publish app".
5. Left menu → **APIs & Services → Credentials** → **Create Credentials** →
   **OAuth client ID**.
   - Application type: **Desktop app**. Name it anything. **Create**.
   - Click **Download JSON** on the dialog. Keep this file safe — it identifies
     your app, though it is not by itself enough to read your Drive.
6. Put the client ID and secret from that file into your `.env` as
   `GOOGLE_OAUTH_CLIENT_ID` and `GOOGLE_OAUTH_CLIENT_SECRET`.
7. Run `broll drive login` (M3). A browser window opens; sign in with the same
   account and accept the "unverified app" warning by clicking **Advanced →
   Go to B-Roll Librarian (unsafe)**. That warning is expected for a testing-mode
   app you built yourself.

Tokens are stored per workspace at `~/.broll/workspaces/<id>/drive_token.json`.

> **Testing-mode caveat:** refresh tokens for an app in testing mode expire after
> 7 days. Re-run `broll drive login` when that happens. Moving the project to
> "In production" removes the expiry but triggers Google's verification
> requirement for the restricted scope.

## Editing exports (M5)

An NLE cannot link media from a Google Drive URL — it needs a local file path.
Exports therefore map each Drive file ID to its path inside your **Google Drive
for Desktop** mount, using `drive_local_mount_path` in the workspace config, e.g.

```
/Users/you/Library/CloudStorage/GoogleDrive-you@example.com/My Drive
```

If it is not set, the export is still produced but the media will import offline
and need relinking. This is the single most common reason an editor thinks the
product is broken.

## Development

```bash
pip install -e '.[dev]'
pytest -q
```

Tests never call a live API: provider behaviour is exercised through recorded
fixtures and the deterministic `mock` provider.

Fixture clips in `tests/fixtures/clips/` are generated with ffmpeg and cover the
*structural* cases — single shot, multi-shot with hard cuts, static camera,
moving camera, deliberately out of focus, fades to black, high motion, vertical
low-resolution. They cannot cover the *semantic* cases (people / no people), so
those are covered by recorded provider responses in `tests/fixtures/responses/`.
Dropping a handful of real clips into `tests/fixtures/clips/` makes the
acceptance run meaningfully stronger.

Run the M1 acceptance gate against a real provider:

```bash
BROLL_ACCEPTANCE_PROVIDER=gemini GEMINI_API_KEY=... pytest tests/test_m1_acceptance.py -s
```

Run the M2 gates:

```bash
pytest tests/test_search.py::test_relevance_smoke_test -s   # 8 queries, top-3
pytest tests/test_queue.py -q                                # resumability
```

The relevance gate runs against `tests/fixtures/library.json` — 14 recorded
analyses of a plausible small library — so it is deterministic and offline. It
was written before the search code, as the spec asks.
