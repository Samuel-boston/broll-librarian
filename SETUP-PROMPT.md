# Setting this up with Claude

Give Claude (Claude Code, with the Claude in Chrome extension connected) the
prompt below, together with the link to this repository. It will do the parts
that can be automated and stop to ask you for the parts that cannot.

Read "What only you can do" at the bottom first, so you know what is coming.

---

## The prompt

> I want to set up the B-Roll Librarian on this machine. The code is at
> <REPO URL>. Read its README.md and docs/DEPLOY.md first, then set it up for
> me end to end.
>
> Use the Claude in Chrome extension for anything that happens in a browser,
> and do as much as you can without me. Drive the browser yourself for
> navigating the Google Cloud console, opening the right pages, filling in
> non-sensitive forms, and reading back values that are shown on screen. Stop
> and ask me whenever a step needs my Google password, two-factor code, payment
> details, a terms-of-service acceptance, or the final "Allow" on a permissions
> screen - those are mine to click, not yours. If Google shows a bot check,
> hand it back to me rather than retrying.
>
> Work in this order, checking each step before moving on:
>
> 1. Clone the repository, create a virtualenv, install it with the
>    gemini, drive, shots, embeddings-local and web extras, and run
>    `broll doctor`. Install ffmpeg if it is missing.
> 2. Walk me through creating a Google Cloud project, enabling the Gemini API
>    and the Google Drive API on it, and linking billing. Open each page for me
>    and tell me exactly what to click. The free Gemini tier is capped at 20
>    requests a day, so billing has to be linked before this is usable.
> 3. Walk me through creating a Gemini API key in that project. Do not read,
>    copy, or type the key yourself - tell me where to paste it once the app is
>    running (Settings, API keys) and let me handle the value.
> 4. Walk me through the Google Drive setup in README.md ("Google Drive setup"),
>    which uses Google Auth Platform: the consent screen, adding me as a test
>    user, adding the full drive scope, and creating a Desktop app OAuth client.
>    Leave the app in Testing - publishing it to Production triggers Google's
>    verification review for that scope. Tell me where to put the client ID and
>    secret. Warn me that in Testing mode the Drive login has to be redone every
>    7 days, unless my Google account is on a Workspace domain, in which case
>    tell me how to set the audience to Internal instead, which has no expiry.
> 5. Create the workspace with `broll init`, then start the app with
>    `broll serve` and open it in the browser for me.
> 6. Run `broll drive login` and hand me the consent screen to approve.
> 7. Ask me about the client this library is for: their name, what they do, who
>    is in front of camera, and the folder structure they want their footage
>    organised into. Write it into the workspace config the way
>    examples/adam-kunder.yaml does, and explain what you changed.
> 8. Test it with two or three files - at least one video and one photo. Confirm
>    each one is captioned, tagged, filed into the right Drive folder, and comes
>    back from a search. Show me the result.
>
> 9. If I also use the Content Ops dashboard, connect the two so its Library >
>    Footage index fills by itself. Ask me for the dashboard's Supabase Project
>    URL and tell me to run `broll connect-dashboard` in my own terminal, where
>    it asks for the service_role key with the input hidden - do not ask me to
>    paste that key into this chat and do not read it. When it prints
>    "Connected", run `broll doctor` and confirm it says the dashboard is
>    connected. After that nothing else needs running: the app pushes changes
>    to the dashboard every minute while it is running.
>
> Tell me at each stage what it is doing and what it costs. If something fails,
> show me the actual error rather than guessing.

---

## What only you can do

Claude cannot do these, by design, and will stop and ask:

* Signing in to Google, passwords, two-factor codes.
* Entering card or billing details, and accepting terms of service.
* The final "Allow" on any Google permissions screen.
* Any bot check or CAPTCHA.
* Pasting the API key itself - Claude will never handle the value.
* Typing the dashboard's service_role key into `broll connect-dashboard` (or
  into Settings > Content Ops dashboard). It is a master key for the dashboard's
  database.

Everything else - installing, configuring, testing, explaining - it can do.

## What it costs to run

* The analysis, per file: roughly 0.3p for a video shot, 0.1p to 0.3p for a
  photo. A library of several hundred files is a few pounds, once.
* Google Drive storage for the footage itself, on your existing plan.
* The machine it runs on. It is one small always-on process; if you already pay
  for a server, it can live there. See docs/DEPLOY.md.

## Why billing matters beyond the cap

On the free Gemini tier, Google may use the content you send for product
improvement, including human review. On a paid key it does not. Your footage
stops being training data.
