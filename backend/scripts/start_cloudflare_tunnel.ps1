# this file is used to start the backend API and then start a cloudflare tunnel to it
# it is intended to be run from the root of the repository
# TODO: path is for my local machine, should be changed to a more generic path or passed as an argument

param(
    [int]$Port = 8000,
    [string]$CloudflaredPath = "$HOME\Downloads\cloudflared-windows-amd64.exe"
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path -LiteralPath $CloudflaredPath)) {
    throw "cloudflared not found on a path: $CloudflaredPath"
}

$resolvedCloudflaredPath = (Resolve-Path -LiteralPath $CloudflaredPath).Path
$targetUrl = "http://localhost:$Port"

Write-Host "Using cloudflared:" $resolvedCloudflaredPath
& $resolvedCloudflaredPath --version

Write-Host "Starting Cloudflare tunnel to $targetUrl"
& $resolvedCloudflaredPath tunnel --url $targetUrl
