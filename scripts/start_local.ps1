# Local one-click startup: API + Streamlit Dashboard
#
# Usage:
#   .\scripts\start_local.ps1
#   .\scripts\start_local.ps1 -SkipBootstrap
#   .\scripts\start_local.bat

param(
    [switch]$SkipBootstrap,
    [string]$ApiHost = "127.0.0.1",
    [int]$ApiPort = 8000,
    [int]$HealthTimeoutSec = 30
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $ProjectRoot

function Write-Step {
    param([string]$Message)
    Write-Host "[WorldCup] $Message" -ForegroundColor Cyan
}

function Test-CommandExists {
    param([string]$Name)
    if (-not (Get-Command $Name -ErrorAction SilentlyContinue)) {
        throw "Command not found: $Name. Install Python 3.11+ and run: pip install -e `".[dev]`""
    }
}

function Wait-ForApi {
    param(
        [string]$BaseUrl,
        [int]$TimeoutSec
    )
    $deadline = (Get-Date).AddSeconds($TimeoutSec)
    while ((Get-Date) -lt $deadline) {
        try {
            $resp = Invoke-RestMethod -Uri "$BaseUrl/health" -TimeoutSec 2
            if ($resp.status -eq "ok") {
                return
            }
        } catch {
            Start-Sleep -Milliseconds 500
        }
    }
    throw "API not ready after ${TimeoutSec}s: $BaseUrl/health"
}

function Ensure-Bootstrap {
    param([switch]$Skip)

    if ($Skip) {
        Write-Step "Skip bootstrap (-SkipBootstrap)"
        return
    }

    $checkpointDir = Join-Path $ProjectRoot "artifacts\checkpoints"
    $latestCheckpoint = Get-ChildItem -Path $checkpointDir -Filter "baseline_dixon_coles_*.json" -ErrorAction SilentlyContinue |
        Sort-Object Name |
        Select-Object -Last 1

    $featureMart = Join-Path $ProjectRoot "data\feature_mart\match_features.parquet"

    if (-not (Test-Path $featureMart)) {
        Write-Step "Feature mart missing. Running export + ingest + build_features ..."
        $sampleMatches = Join-Path $ProjectRoot "data\samples\matches.csv"
        if (-not (Test-Path $sampleMatches)) {
            python -m scripts.export_sample_data
        }
        python -m scripts.ingest
        python -m scripts.build_features
    }

    if (-not $latestCheckpoint) {
        Write-Step "Checkpoint missing. Running train ..."
        python -m scripts.train
        $latestCheckpoint = Get-ChildItem -Path $checkpointDir -Filter "baseline_dixon_coles_*.json" -ErrorAction SilentlyContinue |
            Sort-Object Name |
            Select-Object -Last 1
    }

    if (Test-Path $featureMart) {
        Write-Step "Syncing wc2026 model-implied odds ..."
        python -m scripts.export_model_odds
        python -m scripts.ingest
        python -m scripts.build_features
    }
}

Test-CommandExists python
Ensure-Bootstrap -Skip:$SkipBootstrap

$apiUrl = "http://${ApiHost}:$ApiPort"
$env:WORLDCUP_API_URL = $apiUrl

Write-Step "Starting API at $apiUrl"
$apiProcess = Start-Process python `
    -ArgumentList @("-m", "uvicorn", "worldcup.api.main:app", "--host", $ApiHost, "--port", "$ApiPort") `
    -PassThru `
    -WindowStyle Minimized

try {
    Wait-ForApi -BaseUrl $apiUrl -TimeoutSec $HealthTimeoutSec
    Write-Step "API is ready"
    Write-Step "Starting Streamlit dashboard (Ctrl+C to stop; API will be closed)"
    Write-Step "Dashboard API URL: $apiUrl"

    $dashboardPath = Join-Path $ProjectRoot "src\worldcup\dashboard\app.py"
    python -m streamlit run $dashboardPath
} finally {
    if ($null -ne $apiProcess -and -not $apiProcess.HasExited) {
        Write-Step "Stopping API process PID=$($apiProcess.Id)"
        Stop-Process -Id $apiProcess.Id -Force -ErrorAction SilentlyContinue
    }
}
