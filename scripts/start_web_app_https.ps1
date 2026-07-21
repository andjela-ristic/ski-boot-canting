param(
    [int]$Port = 8000,
    [string]$CloudflaredPath = "$HOME\Downloads\cloudflared-windows-amd64.exe",
    [ValidateSet("Docker", "SystemPython", "Existing")]
    [string]$Backend = "Docker",
    [string]$PythonPath = "python"
)

$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path

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
                throw "Docker nije dostupan u PATH. Pokreni Docker Desktop ili koristi -Backend SystemPython."
            }

            Write-Host "Pokrecem backend preko docker compose"
            & docker compose up --build -d
        }
        "SystemPython" {
            $pythonCommand = Get-Command $PythonPath -ErrorAction SilentlyContinue
            if (-not $pythonCommand) {
                throw "Python nije dostupan kao '$PythonPath'. Prosledi punu putanju ili koristi -Backend Docker."
            }

            $apiProcess = Start-Process -FilePath $pythonCommand.Source `
                -ArgumentList @("-m", "api", "--host", "0.0.0.0", "--port", "$Port") `
                -WorkingDirectory $repoRoot `
                -WindowStyle Hidden `
                -PassThru

            Write-Host "Pokrenut API PID $($apiProcess.Id) preko system Python-a"
        }
        "Existing" {
            throw "Na portu $Port trenutno nema servisa. Pokreni backend rucno ili koristi -Backend Docker / -Backend SystemPython."
        }
    }

    $existingConnection = Wait-ForPort -TargetPort $Port
    if (-not $existingConnection) {
        throw "API nije poceo da slusa na portu $Port u predvidjenom roku."
    }
}

powershell -ExecutionPolicy Bypass -File (Join-Path $repoRoot "scripts\start_cloudflare_tunnel.ps1") `
    -Port $Port `
    -CloudflaredPath $CloudflaredPath
