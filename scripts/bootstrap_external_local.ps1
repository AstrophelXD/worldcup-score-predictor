# External data ingest + ScoreGen local training (Windows)
#
# Usage:
#   .\scripts\bootstrap_external_local.ps1
#   .\scripts\bootstrap_external_local.ps1 -SkipPrepare

param(
    [switch]$SkipPrepare
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $ProjectRoot

function Write-Step {
    param([string]$Message)
    Write-Host "[WorldCup] $Message" -ForegroundColor Cyan
}

if (-not $SkipPrepare) {
    Write-Step "Exporting WC seeds (if needed) ..."
    python -m scripts.export_external_seeds

    Write-Step "Downloading public datasets into data/external/downloads/ ..."
    python -m scripts.bootstrap_external_downloads

    Write-Step "Preparing external canonical CSVs ..."
    python -m scripts.prepare_data --config-name=config data=external
}

Write-Step "Ingesting raw -> curated ..."
python -m scripts.ingest --config-name=config data=external

Write-Step "Building PIT feature mart ..."
python -m scripts.build_features --config-name=config data=external

Write-Step "Training ScoreGen on local GPU (models=scoregen_local training=local) ..."
python -m scripts.train --config-name=config data=external models=scoregen_local training=local

Write-Step "Calibrating ScoreGen checkpoint ..."
python -m scripts.calibrate --config-name=config data=external models=scoregen_local training=local

Write-Step "Done. Checkpoint under artifacts/checkpoints/"
