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

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$backendRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path

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

$existingConnection = Get-ListeningConnection -TargetPort $Port

if (-not $existingConnection) {
    switch ($Backend) {
        "Docker" {
            if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
                throw "Docker not found in PATH. Please start Docker Desktop or use -Backend SystemPython."
            }

            Write-Host "Starting backend via docker compose"
            & docker compose up --build -d
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

powershell -ExecutionPolicy Bypass -File (Join-Path $PSScriptRoot "start_cloudflare_tunnel.ps1") `
    -Port $Port `
    -CloudflaredPath $CloudflaredPath
