# How clips get their tags, and how to check they're right

The aim: a clip turns up when you search for what is in it, and **does not** turn up when you search for
something else. A missing tag costs you one clip. A wrong tag costs you trust in every search.

## What the model is asked to do

For each shot the model sees a handful of frames, each labelled with its time, and fills in the result in
this order:

1. **observations** - what it can literally see: "man sitting cross-legged", "sand", "orange sky". Written first,
   so everything after is grounded in it.
2. **caption** - one sentence, how an editor would say it out loud.
3. **subjects, action, setting, mood, emotions** - from fixed vocabularies, so the same thing is always
   called the same thing.
4. **tags** - search keywords for things that are visibly there. Each must be one of the observations or an
   everyday synonym of one. Feelings, ideas and "what it represents" are not allowed here.
5. **themes** - only from the client's own list (`client.themes` in the library's config), and only when the footage clearly
   shows one. Most clips have none. This is where "nervous system", "breathwork" and the like live.
6. **confidence** and, when the client has folders, **which folder** and how sure it is of that.

After the model answers, the library checks its work (`analysis/analyzer.py`): a feeling or idea that
slipped into `tags` is dropped (a theme is moved to `themes`), filler like "footage" or "cinematic" is dropped,
the client's own name is removed, and themes not on the client's list are discarded.

## What is searchable

Search looks at two columns (`searchtext.py`):

* **the picture** - caption, subjects, action, place, time of day, tags. A match here counts in full.
* **concepts** - themes, mood, emotions, **search phrases**, and the name of every folder the clip is filed
  under (so a clip in Speaking > Keynotes is also found by "speaking"). A match here counts for 70%, and ranks lower.

**Search phrases** are written by the model as it describes the clip: 3 to 8 short phrases an editor might type to
find it, in their words rather than the model's ("staring blankly", "head in hands", "low point"). This is what
lets a search for how a clip feels or what the person is doing find it, even when the caption says something
else. They only ever describe what is true of the clip.

**What a folder is for** also counts. The client's own note on a folder ("Low points: sad, crying, staring blankly,
head in hands...") is added to the meaning-based search text of every clip filed there, so a search for any word
of the note reaches all of them. Change a note with `broll folders note`, then `broll reembed` so existing clips pick it up.

Not searchable as words: shot type, camera movement, colour, people count, pace, "usable for". They sit
on most clips, so as words they only add noise ("slow", "wide", "static", "one"). They are still filters,
and a query that *asks* for one ("close up of coffee", "handheld walk", "aerial shot") nudges clips with that
shot type up the list. It never hides the rest, because the model's shot-type label can be wrong.

Meaning-based search (embeddings) is on top of that. A clip is kept if enough of the query's words are
literally in it, or if it means the same thing and scores clearly above how the query scores on the rest of the
library. That floor is measured per query on a sample of the library, so it works whichever embedding model is in
use. A library with fewer than 60 shots uses fixed numbers for its model instead (`search/query.py`, `CALIBRATION`).

## Checking it: `broll audit-tags`

```bash
broll audit-tags            # or: GET /api/audit/tags
```

It reads the index (no model calls) and reports:

* **tags on too many clips** (over 12% of the library, once there are 25 clips). This is the main warning sign:
  such a tag makes every search that uses it return the same pile.
* the **themes** in use and how many clips have none (most should).
* **how sure the model said it was**, overall and about the folder. If everything is 0.9+, the review
  threshold isn't catching anything and should be raised.
* **why shots are waiting for review**, and the clips whose tags the description never mentions.

Run it after the first 40 or so clips and again after a big batch. If a tag is on too many clips, don't
delete it by hand: tighten the client's theme list or the prompt and re-run `broll reanalyse` on those clips.

## When a person should look

## Changing the folders

```bash
broll folders rename "08_Life Chapters" "08_Adam's Journey"   # clips follow
broll folders add "08_Adam's Journey" "Landscaping Business" -n "What belongs in it. The model reads this."
broll folders note "07_Mood/Struggle & Upset" "What the folder is for."
broll folders remove "07_Mood/Night & City" --into "07_Mood/Reflective"
```

A folder's note is read by the model when it files a clip, so it is where "this, not that" goes. Clips remember their folder
by path, so rename and remove carry them along (a removed folder's clips are flagged for review unless `--into` says where).
Re-describing a folder changes where *new* clips go; `broll reanalyse` re-files the ones already indexed.

A shot goes to **Review** when: the model's confidence is under 0.7; it was under 0.6 sure of the folder; the
folder it named is not one of the client's; nothing in the file looked usable; or the footage is technically
poor (shaky, out of focus...). The first four mean the folder may be wrong, so the file is put in the
`_Needs Review` folder in Drive instead of being guessed into a folder. Confirming it in Review moves it to
the right place. A shaky clip stays in its folder.

When nothing in the client's folders fits, the model suggests a new one. It is **not** created: it appears
under "Suggested folders" on the Review page (and `broll folders`). Approve it and the clips that suggested it
move in. Set `taxonomy.auto_create_folders: true` to let the model create folders on its own (not recommended).

## Looking inside a clip

A clip eight seconds or longer is first looked through as a whole (`analysis/segmentation.py`): frames spread
across it, each with its time, and the model splits it into stretches that are **usable**, **setup** (someone
positioning the camera, walking up to start it, a slate, a countdown, focus being set) or **dead** (black,
lens covered, the floor). Only usable stretches are indexed, each as its own shot with its own in and out
point. A take with two separate scenes becomes two shots. For each, the model also names the **best part**:
the strongest 3-10 seconds, which is what an editor cuts first (`best_start_s` / `best_end_s` in the API).

What was left out is kept on the file (`broll show <source id>` shows the plan), so you can see why. If the model
can't be asked, the whole file is treated as one shot, as before. If it finds nothing usable at all, the whole
file is kept and flagged for review rather than lost.

A shot gets one frame about every 4 seconds (3 to 8), not three however long it is.

## Files that are not B-roll

See "Files that need a decision" in `HOSTING.md`.
