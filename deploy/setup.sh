#!/bin/sh
# Interactive first-time setup on the server: writes .env, then starts the app.
#   cd /opt/broll-librarian && ./deploy/setup.sh
set -eu
cd "$(dirname "$0")/.."

if [ -f .env ]; then
    printf '.env already exists. Overwrite it? [y/N] '
    read -r answer
    case "$answer" in y|Y) ;; *) echo "Left .env alone. Edit it by hand, then run: docker compose --profile caddy up -d --build"; exit 0 ;; esac
fi

echo "B-Roll Librarian setup"
echo "----------------------"
printf 'Domain (e.g. library.example.com), or leave empty to use Cloudflare Tunnel instead: '
read -r domain

if [ -n "$domain" ]; then
    way_in="caddy"
    printf 'Shared password for the team (leave empty to generate one): '
    read -r password
    if [ -z "$password" ]; then
        password=$(python3 -c "import secrets; print(secrets.token_urlsafe(9))")
    fi
    public_url="https://${domain}"
else
    way_in="tunnel"
    printf 'Cloudflare Tunnel token (Zero Trust -> Networks -> Tunnels): '
    read -r tunnel_token
    printf "This app's public address (the hostname you set in the tunnel, e.g. https://library.example.com): "
    read -r public_url
    password=""
fi

printf 'Gemini API key (leave empty to add later in Settings): '
read -r gemini_key
printf 'Google OAuth client ID (leave empty to add later in Settings): '
read -r google_id
if [ -n "$google_id" ]; then
    printf 'Google OAuth client secret: '
    read -r google_secret
else
    google_secret=""
fi

{
    echo "BROLL_PUBLIC_URL=${public_url}"
    [ -n "${domain:-}" ] && echo "BROLL_DOMAIN=${domain}"
    [ -n "$password" ] && echo "BROLL_ACCESS_PASSWORD=${password}"
    [ -n "$gemini_key" ] && echo "GEMINI_API_KEY=${gemini_key}"
    [ -n "$google_id" ] && echo "GOOGLE_OAUTH_CLIENT_ID=${google_id}"
    [ -n "$google_secret" ] && echo "GOOGLE_OAUTH_CLIENT_SECRET=${google_secret}"
    [ "$way_in" = "tunnel" ] && echo "CLOUDFLARE_TUNNEL_TOKEN=${tunnel_token}"
} > .env
chmod 600 .env

echo
echo "Wrote .env. Starting the app (this builds the image the first time, a minute or two)..."
docker compose --profile "$way_in" up -d --build

echo
echo "Done. Open ${public_url} in a browser."
if [ -n "$password" ]; then
    echo "Team password: ${password}"
    echo "(Also saved in .env on this server. Share it privately, not in a group chat.)"
fi
if [ -z "$gemini_key" ] || [ -z "$google_id" ]; then
    echo "Still needed, in Settings once the site loads: $( [ -z "$gemini_key" ] && printf 'the Gemini key' )$( [ -z "$gemini_key" ] && [ -z "$google_id" ] && printf ', ' )$( [ -z "$google_id" ] && printf 'the Google OAuth client, then Connect Google Drive' )."
fi
