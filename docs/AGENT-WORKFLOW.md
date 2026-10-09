# Using the librarian from an agent (ChatGPT agent mode, Claude, a script)

Two ways in, with the same result: the web app (an agent that can use a browser, such as ChatGPT's agent
mode, can click through it), or the JSON API (`/api/...`, for anything that can make HTTP requests).
This is the whole path from raw footage to clips in a timeline.

## 1. Put footage in

* Web: **Upload**, drop files in. Video and photos, including camera RAW (.NEF and others).
* API: `POST /api/upload` with the files, or `POST /api/ingest` with paths on the server.
* Footage already in Google Drive: `broll index --drive-folder <link>` (run `--dry-run` first to see how many
  files, how many GB, and how many would be left for a person to decide).

Indexing is automatic. Watch it on the **Upload** page, or `GET /api/status` (`queued`, `running`,
`needs_attention`).

## 2. Deal with what was set aside

Files that are too long to be B-roll (default: over 10 minutes), too big for the server (over 8 GB), unreadable, or
whose download didn't finish are **not** indexed and **not** lost. They are on **Needs attention**
(`GET /api/attention`), each with the reason and a link to open it in Drive.

For each one: open it, then **Index anyway** (the library looks through it for usable stretches) or
**Dismiss** (it is never queued again). `POST /api/attention/{id}/index` and `.../dismiss` do the same.

The same list appears in Drive, as shortcuts under `_Needs Attention/`, one folder per reason. Nothing is moved.

## 3. Check what the model decided

* **Review** lists shots a person should look at, with the reason. Fix the caption, tags or folder and save.
  Saving moves the file to the right Drive folder.
* **Suggested folders** (top of Review): approve to create a folder the model suggested.
* `GET /api/audit/tags` says whether the tagging is healthy (tags on too many clips, themes in use).

## 4. Find clips

`GET /api/search?q=...&client=...` for a phrase. For a whole script, `POST /api/shortlist` (candidates per
line, free) or `POST /api/suggest` with `"rerank": true` (the model picks and gives a reason per line).

Each result carries what an editor needs:

| field | meaning |
|---|---|
| `drive_link`, `local_path` | where the file is |
| `start_s`, `end_s` | the usable stretch of the file (setup and dead air already cut off) |
| `best_start_s`, `best_end_s` | the strongest part inside it: cut from here first |
| `caption`, `tags`, `themes`, `emotions` | what is in it, what it feels like |
| `relevance`, `keyword_coverage` | whether it is really about the line (`match`, `near`, `weak`) |
| `status`, `review_reasons` | `needs_review` means a person hasn't confirmed it |

Take `match` results, prefer ones outside `needs_review`, and say no when nothing is a `match`: a gap is better
than a wrong clip. Ask for a kind of shot in plain words ("close up of coffee", "aerial shot of the coast"); it
nudges the ranking.

## 5. Get the clip into the edit

* `POST /api/fetch` turns a shot into a local file, cut down to the shot (or its best part) when asked.
* The **Script to B-roll** page exports a Premiere (FCP7 XML), Resolve (EDL) or CSV timeline.
* `POST /api/usage` records which video a shot went into, so the next shortlist doesn't offer it again.

## Authentication

On the hosted server the web app asks for the team password. The API needs `BROLL_API_TOKEN`
(`Authorization: Bearer <token>`); without one it only answers requests from the server itself.
