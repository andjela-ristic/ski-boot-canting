param(
    [int]$Port = 8000,
    [string]$CloudflaredPath = "$HOME\Downloads\cloudflared-windows-amd64.exe"
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path -LiteralPath $CloudflaredPath)) {
    throw "cloudflared nije pronadjen na putanji: $CloudflaredPath"
}

$resolvedCloudflaredPath = (Resolve-Path -LiteralPath $CloudflaredPath).Path
$targetUrl = "http://localhost:$Port"

Write-Host "Koristim cloudflared:" $resolvedCloudflaredPath
& $resolvedCloudflaredPath --version

Write-Host "Pokrecem Cloudflare tunnel ka $targetUrl"
& $resolvedCloudflaredPath tunnel --url $targetUrl
