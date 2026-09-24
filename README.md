# B-Roll Librarian

Turns a pile of unsorted B-roll into a searchable, auto-organised library backed
by Google Drive.

Drop in videos and photographs (or point it at a Drive folder). In the background
it splits each video into shots, analyses them with a vision model, writes rich
structured metadata into a local database, uploads the files into a meticulously
organised Drive tree, and then lets you find footage two ways:

1. **Search** — natural language plus filters, returning thumbnails and Drive links.
2. **Transcript matching** — paste a transcript, get suggested B-roll per beat,
   exportable as a timeline for Premiere Pro or DaVinci Resolve.

**The video files never live in the database.** The database stores metadata and
a Google Drive file ID. Everything else follows from that constraint.

**Photographs are footage too.** A still runs the same pipeline with the shot
detection removed: one "shot", one frame, no duration and no camera movement.
It is captioned, tagged, filed and searched exactly like a clip, and can be cut
under a line of narration - where it holds the frame for as long as the beat
needs. `.jpg .jpeg .png .heic .heif .webp .tif .tiff .avif .bmp` are indexed;
HEIC is read through ffmpeg, so there is nothing extra to install.

---

**Setting it up for a client?** [SETUP-PROMPT.md](SETUP-PROMPT.md) is a prompt
to hand to their Claude, which does the install and the Google setup with them.
[docs/DEPLOY.md](docs/DEPLOY.md) covers running it on a server.

---

## Status

| Milestone | Scope | State |
|---|---|---|
| **M1** | Config, registry + workspace schema, provider abstraction, frame extraction, analysis, `broll analyse` | **done** — verified on real footage with Gemini 3.6 Flash |
| **M2** | Shot detection, job queue, batch indexing, embeddings, FTS5 + vector search | **done** |
| **M3** | Drive OAuth, upload, taxonomy, shortcut tree, `reorganise`, `--dry-run` | **done** — verified against live Drive; second pass performs zero writes |
| **M4** | Web UI: ingest, queue view, search | **done** |
| **M5** | Transcript matching, FCP7 XML / EDL / CSV export | **done** (Premiere import is a manual gate, see below) |
| **M6** | Review queue, settings, cost reporting, vocabulary management | **done** |

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
| `gemini` | `gemini-3.6-flash` | **The default.** Cheap per image, which is what dominates the bill at 3 frames per shot. (`gemini-2.5-flash` is closed to new API projects — Google returns a 404 pointing at 3.6.) |
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
| gemini-3.6-flash | $0.0038 | $3.79 | $18.95 |
| gpt-4.1-mini | $0.0025 | $2.48 | $12.40 |
| claude-haiku-4-5 | $0.0055 | $5.50 | $27.50 |
| claude-opus-5 | $0.0275 | $27.50 | $137.50 |

Gemini 3.x Flash prices are a promotional rate to 31 Dec 2026 and double on 1 Jan 2027. These are estimates from published list prices, computed by
`analysis/providers/*.PRICING`. Real costs are logged per job; `broll costs`
reports what a workspace has actually spent, per file and per shot, and
`--project N` extrapolates from that measured rate rather than from this table.
Trust the measurement over the table.

## Web UI

```bash
broll serve                    # http://127.0.0.1:8000
broll serve --no-worker        # UI only; run `broll work --follow` separately
```

One process: FastAPI + Jinja2 + HTMX + Tailwind from a CDN. No SPA, no npm, no
build step. The ingest worker runs inside the same process by default, so
dropping files in the browser is all it takes — no CLI step.

* **Ingest** — drag and drop, or point at a folder (files there are never moved,
  copied or deleted). The queue panel polls every two seconds while work is
  outstanding and shows queued/running/done/failed, shots indexed, cost so far
  and estimated cost remaining.
* **Search** — query box, filter sidebar built from the facets the library
  actually contains, thumbnail grid with captions, timecodes, quality flags,
  a Drive link and a copy-link button.

* **Transcript** — paste or upload, see suggestions per beat, swap any of them,
  export FCP7 XML / EDL / CSV.
* **Review** — the `needs_review` queue at *shot* level, so a multi-shot file
  shows exactly which shot needs attention. A shot lands here when the model's
  own confidence is below `ingest.review_below_confidence` (0.7 by default;
  clean footage scores 0.95+), when it reports a defect in the footage, or when
  the analysis failed twice. Corrections are ground truth: saving one recomputes
  the shot's search text and embedding immediately.
* **Settings** — provider and model, API keys, Drive connection, taxonomy
  thresholds, the Drive-for-Desktop mount path, and vocabulary promotion.

API keys typed into Settings are written to `$BROLL_HOME/.env` with owner-only
permissions and are never rendered back to the page, never stored in the
database, and never written to `config.yaml`.

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

## Keeping the library honest

```bash
broll review                       # what needs a human look, and why
broll fix <shot-id> --set setting=beach --set "tags=surf, ocean, dawn"
broll vocab --min-count 3          # terms the model keeps inventing
broll vocab --promote subjects=hydrofoil
broll costs --project 5000         # measured cost, and what 5,000 more would cost
broll reanalyse --dry-run          # re-run rows analysed with an older prompt
```

`broll reanalyse` is deliberately manual. A new provider model never triggers a
re-analysis on its own — that would spend the client's money without asking.

A shot a human has corrected is marked as such, and `reanalyse` skips it unless
you pass `--overwrite-corrections`. Re-analysis must never quietly undo human
judgement.

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

## Client profiles and a client's own folder structure

Each workspace can describe who the library belongs to and how they organise
footage. `examples/adam-kunder.yaml` is a complete, real example - merge it into
`~/.broll/workspaces/<id>/config.yaml` and edit freely; the brief, themes and
folder notes all go straight into the analysis prompt.

**Client profile** (`client:`)

* `brief` and `themes` - what the client does, so tags and emotions reflect
  their world ("nervous system", "breathwork", "9-to-5" for a past career).
* `featured_person` - the recurring person to name in captions ("Adam meditates
  on the beach at sunrise"). **This is an instruction to the model, not face
  recognition** (identifying people stays a v1 non-goal): it names whoever is
  clearly the lone featured subject, so a shot of someone else alone can be
  misnamed. The Review screen fixes it in one click. `featured_person_description`
  ("a man") helps it tell them from others.
* `emotions` - the client's emotion vocabulary. **Emotions are separate from
  mood**: mood is how a shot *looks* (cinematic, moody, clean); emotions are what
  the person feels and what a viewer feels. Editors cut to feeling, so emotions
  are shown on every card and are a search filter.

**Their own folder tree** (`taxonomy.mode: tree`)

* Each file lives in **one** best-fit folder, renamed to
  `taxonomy.filename_template` (e.g. `{leaf}_{action}_{emotion}` →
  `meditation_stillness_meditating_calm_f54362f4.mov`). Other folders it also
  fits get a shortcut. `★ Top Picks` holds shortcuts to starred clips.
* A folder's note is inherited by everything under it, so a rule written once on
  a parent ("only for shots with no clear activity") reaches the model.
* **New folders:** when nothing fits, the model can propose one under the right
  existing parent. It is created, saved to the workspace config and offered to
  every later clip; a near-duplicate of an existing sibling ("Beach and Water"
  next to "Beach & Water") reuses it instead. Turn off with
  `taxonomy.allow_new_folders: false`.
* **Splitting video from photographs** (`taxonomy.media_split: true`): the same
  tree exists under `Videos/` and `Images/`, and a source is filed under the one
  that matches it. The tree is written once and used for both.
* A `00_START HERE` Google Doc explains the naming format, the folders and the
  emotion list. It is written once, so edit it by hand if you like.
* The whole structure is created up front, empty folders included - editors
  learn where things go by browsing it.

Changing the tree is safe: `broll reorganise` moves files and shortcuts to match
the database, and never deletes footage.

## Running it day to day

Double-click **B-Roll Librarian** (`scripts/B-Roll Librarian.command`; a copy
can live on the Desktop). It starts the app and opens it in the browser; close
the Terminal window to stop it. **Restart it after updating the code** - a
running app keeps the code and settings it started with.

The web app and CLI commands can share a workspace safely: each job records the
process holding it, and a starting worker only reclaims jobs whose owner has
actually died.

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

## Connecting to the Content Ops dashboard

If the client also uses the Content Ops dashboard, the library's Footage index
(Library > Footage index) can fill by itself. The librarian keeps its working
index in SQLite and mirrors it into the dashboard's Supabase project, so it
works the same whether the librarian runs on a Mac or on a server.

One-time setup, on the machine that runs the librarian:

```bash
broll connect-dashboard
```

It asks for the dashboard's Supabase **Project URL** and **service_role** key
(Supabase > Project Settings > API; the key is typed hidden). It checks both,
saves them, and sends the first sync. The same form is under **Settings >
Content Ops dashboard**. For a headless server, set `DASHBOARD_SUPABASE_URL`
and `DASHBOARD_SUPABASE_KEY` in the environment and run `broll connect-dashboard`
once.

After that nothing else needs running. `broll serve` pushes changes every
minute (set `dashboard.interval_s` in the workspace config to change it), and
`broll work` pushes when it finishes a run. Only shots that changed are sent,
thumbnails are uploaded once, and shots removed from the library are removed
from the dashboard. Video files are never uploaded: the dashboard holds the
metadata, a thumbnail and the Drive link.

`broll sync` pushes immediately; `broll sync --force` re-sends everything;
`broll doctor` reports whether the connection works. If it fails, the message
says why (wrong key, or the dashboard database is missing its `library_shots`
table - run the dashboard's `supabase/setup_all.sql`).

The service_role key is stored only in `BROLL_HOME/.env` (owner-only), like the
provider keys.

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

Google renamed this flow: there is no longer an "OAuth consent screen" wizard
under APIs & Services. It is now **Google Auth Platform**, and the steps below
match that. Swap `YOUR-PROJECT` in the links for your project id.

1. **Create a project** at <https://console.cloud.google.com/projectcreate>, if
   you do not already have one. Name it something you will recognise.
2. **Enable the Drive API**:
   `https://console.cloud.google.com/apis/library/drive.googleapis.com?project=YOUR-PROJECT`
   → **Enable**. Wait for it to finish.
3. **Configure Google Auth Platform**:
   `https://console.cloud.google.com/auth/overview?project=YOUR-PROJECT`
   → **Get started**, then fill in the short form:
   - *App name* — anything, e.g. `B-Roll Librarian`. *User support email* — yours.
   - *Audience* — **External**. (Internal only exists for Workspace orgs.)
   - *Contact information* — your email. Then agree and **Create**.
4. **Add yourself as a test user**:
   `https://console.cloud.google.com/auth/audience?project=YOUR-PROJECT`
   → under **Test users**, **Add users**, enter your own Google address, save.
   Leave the publishing status as **Testing**. This is what lets you use a
   restricted scope without Google's verification review. Do not click
   "Publish app".
5. **Add the Drive scope**:
   `https://console.cloud.google.com/auth/scopes?project=YOUR-PROJECT`
   → **Add or remove scopes**, paste `https://www.googleapis.com/auth/drive`
   into the filter, tick it, **Update**, then **Save**.
6. **Create the OAuth client**:
   `https://console.cloud.google.com/auth/clients?project=YOUR-PROJECT`
   → **Create client** → *Application type*: **Desktop app** → **Create**.
   Copy the **Client ID** and **Client secret** from the dialog.
7. **Put them in your `.env`**:

   ```
   GOOGLE_OAUTH_CLIENT_ID=...apps.googleusercontent.com
   GOOGLE_OAUTH_CLIENT_SECRET=...
   ```

8. **Connect the workspace**: `broll drive login`. A browser window opens; sign
   in with the same account. You will see an "unverified app" warning — that is
   expected for a testing-mode app you built yourself. Click **Advanced → Go to
   B-Roll Librarian (unsafe)** and allow access.

Tokens are stored per workspace at `~/.broll/workspaces/<id>/drive_token.json`.

> **Testing-mode caveat:** refresh tokens for an app in testing mode expire after
> 7 days. Re-run `broll drive login` when that happens. Moving the project to
> "In production" removes the expiry but triggers Google's verification
> requirement for the restricted scope.

## Transcript matching

```bash
broll transcript script.srt --out ./exports              # all three formats
broll transcript notes.txt --out ./exports --format fcp7 # plain text is timed by estimate
broll transcript script.vtt --json                       # machine-readable, no files
```

Or use the Transcript screen: paste or upload, swap any suggestion for another
candidate, then export.

1. **Parse** — `.srt` and `.vtt` carry real timecodes; plain text is timed from
   `transcript.words_per_minute` (default 150) and flagged as estimated.
2. **Segment into beats** — 3-15 second spans, split on sentence boundaries then
   merged and divided to land in that window.
3. **Retrieve** — the same hybrid search, top 8 candidates per beat.
4. **Rerank** — a text model picks the best 3 with a one-line reason each. It is
   told that the literal match is usually the wrong answer (narration about
   "slowing down" wants a calm, slow-paced visual, not necessarily a clock), and
   that returning **"no good match"** is better than forcing one.
5. **Enforce variety** — a shot already used in the timeline is penalised, so the
   same clip does not carry five beats.
6. **Output** — beat timecode, narration, ranked suggestions with Drive links and
   reasons, plus a list of the footage you should go and shoot.

If the reranker is unavailable — no key, an outage — the timeline is still
produced in hybrid-search order rather than failing.

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

Timebase and trimming rules, stated once because they are not obvious:

* The sequence timebase comes from config, defaulting to the modal fps across
  the suggested clips; every source's timecodes are converted into it. Mixed
  frame rates in one library are normal, so the conversion is explicit and
  reported.
* Each clip is placed at its beat's start, trimmed to the beat's duration,
  starting from the shot's `start_s`.
* If the shot is shorter than the beat, the remainder is left as a **gap** and
  flagged in the report — never stretched or frozen.

FCP7 XML (`.xml`) is the export to reach for: it imports into both Premiere Pro
and DaVinci Resolve. CMX3600 EDL is the universal fallback (one timebase, so
cross-rate sources are conformed). CSV is for anyone who just wants the list.

**Automated coverage:** golden-file tests for all three formats, including a
mixed-frame-rate sequence and a clip too short for its beat. **Manual gate not
yet run:** importing a real export into Premiere with media linked needs a
Premiere licence and a synced Drive mount.

## Command reference

| Command | What it does |
|---|---|
| `broll init --name X --provider gemini` | Create a workspace |
| `broll connect-dashboard` | Connect to the Content Ops dashboard and sync (asks for the key, hidden) |
| `broll sync` | Push the index to the dashboard now (`--force` re-sends everything) |
| `broll doctor` | Check ffmpeg, SQLite features, embeddings, credentials |
| `broll analyse <clip>` | Analyse one clip, print JSON, write nothing |
| `broll index <path> [--dry-run] [--organise]` | Queue footage and work the queue |
| `broll index --drive-folder <id>` | Index footage already in Drive |
| `broll work [--follow]` | Work the queue; safe to kill and restart |
| `broll status` | Queue, counts, vector backend, cost |
| `broll search "..." [--json]` | Hybrid search with filters |
| `broll reembed` | Rebuild the vector index after changing the embedder |
| `broll drive login / logout / status` | Drive connection for this workspace |
| `broll organise [--dry-run]` | Upload and file footage into the Drive tree |
| `broll reorganise [--dry-run]` | Rebuild the whole tree from the database |
| `broll transcript <file> --out DIR` | Match a transcript and export a timeline |
| `broll review` / `broll fix <shot>` | The review queue, and corrections |
| `broll vocab [--promote field=term]` | Out-of-vocabulary terms, and promotion |
| `broll costs [--project N]` | Measured spend, and a projection |
| `broll reanalyse [--stale/--all]` | Re-run analysis after a prompt change |
| `broll serve` | The web UI plus the ingest worker, one process |

Every screen in the UI has a CLI equivalent, deliberately: the CLI is the real
interface and the UI is a client of it.

## Not in v1

Said out loud so they do not creep in: audio analysis or transcription of the
B-roll itself (B-roll is used muted), face recognition, splitting or re-encoding
video, hover-preview proxies, native-video (rather than keyframe) analysis,
multi-user permissions inside a workspace, a mobile app, and automatic
re-analysis when a provider ships a new model.

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
