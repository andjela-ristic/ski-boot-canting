# this file is used to start the backend API and then start a cloudflare tunnel to it
# it is intended to be run from the root of the repository
# TODO: path is for my local machine, should be changed to a more generic path or passed as an argument

param(
    [int]$Port = 8000,
    [string]$CloudflaredPath = "$HOME\Downloads\cloudflared-windows-amd64.exe", # should be changed 
    [ValidateSet("Docker", "SystemPython", "Existing")]
    [string]$Backend = "Docker",
    [string]$PythonPath = "python"
)

$ErrorActionPreference = "Stop"

$backendRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path

function Get-ListeningConnection {
    param([int]$TargetPort)

    return Get-NetTCPConnection -LocalPort $TargetPort -ErrorAction SilentlyContinue |
        Where-Object { $_.State -eq "Listen" } |
        Select-Object -First 1
}

function Wait-ForPort {
    param(
        [int]$TargetPort,
        [int]$TimeoutSeconds = 20
    )

    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    do {
        $connection = Get-ListeningConnection -TargetPort $TargetPort
        if ($connection) {
            return $connection
        }

        Start-Sleep -Milliseconds 500
    } while ((Get-Date) -lt $deadline)

    return $null
}

function Test-OriginFrontend {
    param([int]$TargetPort)

    try {
        $response = Invoke-WebRequest -UseBasicParsing -Uri "http://localhost:$TargetPort/" -TimeoutSec 5
    } catch {
        Write-Warning "Origin on port $TargetPort is reachable, but GET / could not be verified: $($_.Exception.Message)"
        return
    }

    $contentType = [string]$response.Headers["Content-Type"]
    $body = [string]$response.Content
    $looksLikeHtml = $contentType -like "text/html*" -or $body.TrimStart().StartsWith("<!DOCTYPE html") -or $body.TrimStart().StartsWith("<html")
    $looksLikeServiceIndex = $body -match '"service"\s*:\s*"ski-boot-canting-api"'

    if ($looksLikeHtml) {
        Write-Host "Verified frontend HTML at http://localhost:$TargetPort/"
        return
    }

    if ($looksLikeServiceIndex) {
        Write-Warning "Origin on port $TargetPort is returning the API service index JSON on GET /, not the frontend HTML."
        return
    }

    Write-Warning "Origin on port $TargetPort responded to GET / with unexpected content-type '$contentType'."
}

$existingConnection = Get-ListeningConnection -TargetPort $Port

if (-not $existingConnection) {
    switch ($Backend) {
        "Docker" {
            if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
                throw "Docker not found in PATH. Please start Docker Desktop or use -Backend SystemPython."
            }

            Write-Host "Starting backend via docker compose"
            Push-Location $repoRoot
            try {
                & docker compose up --build -d
            } finally {
                Pop-Location
            }
        }
        "SystemPython" {
            $pythonCommand = Get-Command $PythonPath -ErrorAction SilentlyContinue
            if (-not $pythonCommand) {
                throw "Python not found as '$PythonPath'. Please provide the full path or use -Backend Docker."
            }

            $apiProcess = Start-Process -FilePath $pythonCommand.Source `
                -ArgumentList @("-m", "api", "--host", "0.0.0.0", "--port", "$Port") `
                -WorkingDirectory $backendRoot `
                -WindowStyle Hidden `
                -PassThru

            Write-Host "Starting API PID $($apiProcess.Id) via system Python"
        }
        "Existing" {
            throw "No service running on port $Port. Please start the backend manually or use -Backend Docker / -Backend SystemPython."
        }
    }

    $existingConnection = Wait-ForPort -TargetPort $Port
    if (-not $existingConnection) {
        throw "API not listening on port $Port within the expected timeframe."
    }
}

Test-OriginFrontend -TargetPort $Port

powershell -ExecutionPolicy Bypass -File (Join-Path $PSScriptRoot "start_cloudflare_tunnel.ps1") `
    -Port $Port `
    -CloudflaredPath $CloudflaredPath
