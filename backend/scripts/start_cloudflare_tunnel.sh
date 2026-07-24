#!/bin/sh
set -eu

PORT="${PORT:-80}"
TARGET_HOST="${TARGET_HOST:-canting-web}"

if ! command -v cloudflared >/dev/null 2>&1; then
    echo "cloudflared not found in PATH" >&2
    exit 1
fi

TARGET_URL="http://${TARGET_HOST}:${PORT}"

echo "Using cloudflared: $(command -v cloudflared)"
cloudflared --version

echo "Starting Cloudflare tunnel to ${TARGET_URL}"
exec cloudflared tunnel --no-autoupdate --url "${TARGET_URL}"
