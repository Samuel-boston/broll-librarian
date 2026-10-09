# Hosting the librarian for a team

Run it once on a small server and everyone uses the same library from their browser: drag files in,
search, browse and review. Nothing to install on anyone's computer, and no local copy to keep in sync.
It also keeps the Content Ops dashboard's Footage index up to date on its own.

This guide is for a person setting it up. `docs/DEPLOY.md` covers the same app on a server you manage by hand.

## What you get

* One always-on server (about 2 GB of RAM) running the web app and the indexing worker in one container.
* Search embeddings from the Gemini API instead of a local model. That is what makes a small server
  enough: no PyTorch, no 2 GB model in memory. It costs a fraction of a cent per shot on top of the analysis.
* A shared password on the app, HTTPS, and a daily copy of the database.
* Google Drive connected from the Settings page (no terminal needed).

Footage is never kept on the server. An uploaded file is staged, analysed, filed into Drive, and removed.

## Cost

Roughly **$8 to $9 a month** in total, paid by whoever owns the library.

| Part | Cost |
|---|---|
| Server: 2 vCPU, 4 GB RAM, 40 GB disk (Hetzner CX23, or similar) | about €5.5 a month |
| Domain name (skip if the team already has one: use a subdomain of it) | about $10 a year |
| Server backups or snapshots (recommended) | about €1 a month |
| Gemini API, one-off for indexing | about $19 per 5,000 shots at the current promotional rate, which doubles on 1 Jan 2027 |

Prices move; check them when you order. Fly.io and Railway also work but cost more, because every uploaded
file counts as outbound traffic on the way to Drive.

## How people get in

Pick one.

**A. Your own domain with a shared password (recommended).** Caddy gets and renews an HTTPS certificate,
and the app asks for a password once per browser (30 days). Uploads have no size limit.

**B. Cloudflare Tunnel with Cloudflare Access.** People sign in with their email, and no ports are open.
The catch is that Cloudflare's proxy caps a single upload at **100 MB on the free plan**, so large video
clips fail. It suits a library that is mostly photos and short clips, or one that is filled by pointing the
app at footage already in Drive. Use A if people will drag in big files.

## Setting it up

You need: a server, a domain name pointing at it, Adam's Gemini key, and a Google Cloud project.

### 1. The server

Create an Ubuntu 24.04 server. When it asks for **Cloud config** (Hetzner: *Create server → Cloud config*),
paste in [`deploy/cloud-init.yaml`](../deploy/cloud-init.yaml). It installs Docker, downloads this repo to
`/opt/broll-librarian`, opens the firewall for SSH, HTTP and HTTPS only, and starts the app on the server's
loopback address. Add an SSH key so you can log in.

Without cloud config: install Docker and Docker Compose, then
`git clone https://github.com/Samuel-boston/broll-librarian.git /opt/broll-librarian`.

### 2. The domain

Add a DNS **A record** for a name such as `library.example.com` pointing at the server's IP address. If it
is on Cloudflare, set it to **DNS only** (grey cloud), not proxied, or the 100 MB cap applies.

### 3. Set it up and start it

```bash
cd /opt/broll-librarian
./deploy/setup.sh
```

It asks for the domain, the team password (or generates one), and the Gemini and Google details — any of
which can be left empty and added later in Settings — then writes `.env` and starts the app. Open the
address in a browser: it asks for the password, then shows the library.

To do it by hand instead: `cp deploy/env.hosted.example .env`, edit it, then
`docker compose --profile caddy up -d --build`.

### 5. Connect the pieces in Settings

* **Gemini key**: under the provider section.
* **Google Drive**: create an OAuth client in Google Cloud (*APIs & Services → Credentials → Create
  credentials → OAuth client ID → Web application*) and add
  `https://library.example.com/drive/callback` as an **Authorised redirect URI**. Paste the client ID and
  secret into Settings → Google Drive, press *Save client*, then *Connect Google Drive* and sign in with the
  account that owns the folder. Enable the Drive API on that project, and set the consent screen to *In
  production* so the login does not expire after 7 days.
* **Drive root folder ID**: the folder the library is filed into.
* **Content Ops dashboard**: the Supabase Project URL and service_role key, so the Footage index fills.

Then try one file on the Upload page before the real run.

### Option B: Cloudflare Tunnel instead of Caddy

Create a tunnel in Cloudflare (*Zero Trust → Networks → Tunnels*), copy its token into
`CLOUDFLARE_TUNNEL_TOKEN`, and add a public hostname pointing at `http://librarian:8000`. Under *Access →
Applications*, add a self-hosted application for that hostname with a policy allowing the team's emails.
Leave `BROLL_ACCESS_PASSWORD` out so people don't log in twice. Start it with
`docker compose --profile tunnel up -d`. Cloudflare may ask for a payment card even on the free plan.

## Day to day

* **Update**: `cd /opt/broll-librarian && git pull && docker compose --profile caddy up -d --build`
* **Logs**: `docker compose logs -f librarian`
* **Change the password**: edit `.env`, then `docker compose --profile caddy up -d`. Everyone is signed out.
* **Backups**: the database is copied every 24 hours to the `backups` folder of the library's data (the last
  7 are kept). That protects against a bad edit, not against losing the server, so also switch on the
  host's own backups or snapshots. List them with `docker compose exec librarian broll backup --list`.
* **Restore**: stop the app, copy a backup over `library.db` in the data volume, and start it again. The
  Drive files are untouched either way.
* **Everything is in one volume**, `broll-data`: the index, thumbnails, backups, and any keys saved from
  Settings.

## Good to know

* The password is shared. Anyone who has it can use the whole library, including deleting things. Give it
  to people you trust, and change it when someone leaves.
* A file being indexed is never interrupted by deleting or clearing the queue: it finishes first.
* Each upload needs disk for a moment. A 40 GB server handles ordinary use; if someone uploads many
  large clips at once, the queue works through them and frees the space as it goes.
* Deleting a clip removes it from the library, not from Drive.
* The switch to Gemini embeddings is made when the library is created. To move an existing library from
  local embeddings to Gemini, run `broll reembed` after changing the setting; the sizes differ, so the
  vector index is rebuilt.

## Editing files inside the data volume

The container runs as user 10001, not root. If you edit a file in the volume by hand (for example
`config.yaml`), run `chown -R 10001:10001` on the volume folder afterwards. A root-owned file makes the
Settings pages fail with a permission error.

Deleting the whole library keeps a snapshot of the database in the `backups` folder first, and keeps any
staged uploads, because one may be the only copy of a clip that has not reached Drive yet.
