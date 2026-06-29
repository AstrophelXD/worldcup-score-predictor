@echo off
cd /d "%~dp0.."
set "WORLDCUP_MODEL=%~1"
if "%WORLDCUP_MODEL%"=="" set "WORLDCUP_MODEL=baseline_dixon_coles"
set "WORLDCUP_API_URL=%~2"
if "%WORLDCUP_API_URL%"=="" set "WORLDCUP_API_URL=http://127.0.0.1:8000"
set "API_LOG=%~dp0..\_logs\api.log"

where python >nul 2>&1
if errorlevel 1 (
    echo [ERROR] python not found in PATH >> "%API_LOG%"
    echo [ERROR] python not found in PATH
    pause
    exit /b 1
)

echo ===== %DATE% %TIME% model=%WORLDCUP_MODEL% ===== >> "%API_LOG%"
echo Starting API model=%WORLDCUP_MODEL% url=%WORLDCUP_API_URL%
python -m scripts.serve >> "%API_LOG%" 2>&1
set "RC=%ERRORLEVEL%"
echo ===== API exit code %RC% %DATE% %TIME% ===== >> "%API_LOG%"
if not "%RC%"=="0" (
    echo.
    echo [ERROR] API exited with code %RC%  (see %API_LOG%^)
    pause
    exit /b %RC%
)
