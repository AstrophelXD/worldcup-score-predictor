@echo off
setlocal
cd /d "%~dp0.."

where powershell >nul 2>&1
if errorlevel 1 (
    echo 未找到 PowerShell，请直接运行: python -m scripts.serve
    exit /b 1
)

powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0start_local.ps1" %*
exit /b %ERRORLEVEL%
