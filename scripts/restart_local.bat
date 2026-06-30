@echo off
setlocal EnableExtensions
chcp 65001 >nul 2>&1

rem ============================================================
rem  WorldCup restart: kill old processes -> sync data -> start
rem  Usage:
rem    scripts\restart_local.bat          full restart
rem    scripts\restart_local.bat fast     skip data sync
rem ============================================================

cd /d "%~dp0.."
set "ROOT=%CD%"
set "API_HOST=127.0.0.1"
set "API_PORT=8000"
set "DASH_PORT=8501"
set "API_URL=http://%API_HOST%:%API_PORT%"
set "LOG_DIR=%ROOT%\_logs"
set "API_LOG=%LOG_DIR%\api.log"
set "SKIP_SYNC=0"
if /I "%~1"=="fast" set "SKIP_SYNC=1"
if /I "%~1"=="--fast" set "SKIP_SYNC=1"

echo.
echo ================================================
echo   WorldCup restart
echo   ROOT=%ROOT%
echo ================================================
echo.

where python >nul 2>&1
if errorlevel 1 (
    echo [ERROR] python not found. Install Python 3.11+ and run: pip install -e ".[dev]"
    pause
    exit /b 1
)

if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"

echo [1/6] Stop old API / Dashboard ...
call :kill_port %API_PORT%
call :kill_port %DASH_PORT%
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0_restart_kill.ps1"
call :sleep 2
echo       ports %API_PORT% / %DASH_PORT% released
echo.

if "%SKIP_SYNC%"=="1" (
    echo [2/6] Skip data sync (fast mode^)
    goto SET_MODEL
)

echo [2/6] Check sample data and feature mart ...
if not exist "data\samples\matches.csv" (
    echo       export_sample_data ...
    python -m scripts.export_sample_data
    if errorlevel 1 goto FAIL
)
if not exist "data\feature_mart\match_features.parquet" (
    echo       first-time ingest + build_features ...
    python -m scripts.ingest
    if errorlevel 1 goto FAIL
    python -m scripts.build_features
    if errorlevel 1 goto FAIL
)

echo [3/6] Check model checkpoint ...
set "WORLDCUP_MODEL=baseline_dixon_coles"
if exist "artifacts\checkpoints\scoregen_football_v2_4060.json" set "WORLDCUP_MODEL=scoregen_football"
if exist "artifacts\checkpoints\scoregen_football_v2.json" set "WORLDCUP_MODEL=scoregen_football"
if exist "artifacts\checkpoints\scoregen_football_v1.json" set "WORLDCUP_MODEL=scoregen_football"
if not exist "artifacts\checkpoints\baseline_dixon_coles_v1.json" (
    if not exist "artifacts\checkpoints\scoregen_football_v2_4060.json" (
        echo       no checkpoint found, training baseline (first run may take a while^)...
        python -m scripts.train
        if errorlevel 1 goto FAIL
    )
)

echo [4/6] Sync 2026 results, export samples, model odds, feature mart ...
python -m scripts.seed_wc2026_results --cutoff 2026-06-30
if errorlevel 1 goto FAIL
python -m scripts.export_sample_data
if errorlevel 1 goto FAIL
python -m scripts.export_model_odds
if errorlevel 1 goto FAIL
python -m scripts.ingest
if errorlevel 1 goto FAIL
python -m scripts.build_features
if errorlevel 1 goto FAIL
goto START_API

:SET_MODEL
set "WORLDCUP_MODEL=baseline_dixon_coles"
if exist "artifacts\checkpoints\scoregen_football_v2_4060.json" set "WORLDCUP_MODEL=scoregen_football"
if exist "artifacts\checkpoints\scoregen_football_v2.json" set "WORLDCUP_MODEL=scoregen_football"
if exist "artifacts\checkpoints\scoregen_football_v1.json" set "WORLDCUP_MODEL=scoregen_football"

:START_API
echo.
echo [5/6] Start API (%API_URL%^)  model=%WORLDCUP_MODEL%
set "WORLDCUP_API_URL=%API_URL%"
start "WorldCup API" /min "%~dp0_start_api.bat" %WORLDCUP_MODEL% %API_URL%

echo       waiting for API (up to 120s^) ...
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0_wait_api.ps1" -Url "%API_URL%/health" -MaxWaitSec 120
if errorlevel 1 goto API_FAIL

powershell -NoProfile -Command "try { $p=(Invoke-RestMethod '%API_URL%/openapi.json').paths; if($p.'/predict/batch'){ Write-Host '       OpenAPI: /predict/batch OK' } else { Write-Host '       [WARN] old API, missing /predict/batch' -ForegroundColor Yellow } } catch { Write-Host '       [WARN] cannot read OpenAPI' -ForegroundColor Yellow }"
echo.

echo [6/6] Start Dashboard (Ctrl+C here does not stop API^)
echo       browser: http://localhost:%DASH_PORT%
echo.
set "WORLDCUP_API_URL=%API_URL%"
set "STREAMLIT_SERVER_FILE_WATCHER_TYPE=none"
python -m streamlit run "src\worldcup\dashboard\app.py" --server.port %DASH_PORT% --server.fileWatcherType none
goto END

:API_FAIL
echo.
echo [ERROR] API not ready after 120s.
echo.
echo --- port %API_PORT% listeners ---
netstat -ano | findstr "LISTENING" | findstr /C:"0.0.0.0:%API_PORT% " /C:"[::]:%API_PORT% "
echo.
echo --- last lines of %API_LOG% ---
powershell -NoProfile -Command "if(Test-Path '%API_LOG%'){ Get-Content '%API_LOG%' -Tail 30 } else { Write-Host '(log empty)' }"
echo.
echo Check taskbar window "WorldCup API" or log: %API_LOG%
pause
exit /b 1

:FAIL
echo.
echo [ERROR] previous step failed. Scroll up for details.
pause
exit /b 1

:END
echo.
echo Dashboard closed. Stopping API ...
call :kill_port %API_PORT%
echo Done.
pause
endlocal
exit /b 0

:kill_port
set "_PORT=%~1"
for /f "tokens=5" %%p in ('netstat -ano ^| findstr "LISTENING" ^| findstr /C:"0.0.0.0:%_PORT% " /C:"[::]:%_PORT% "') do (
    echo       kill PID %%p (port %_PORT%^)
    taskkill /F /PID %%p >nul 2>&1
)
exit /b 0

:sleep
set "_SEC=%~1"
if "%_SEC%"=="" set "_SEC=1"
set /a "_PING=%_SEC%+1"
ping -n %_PING% 127.0.0.1 >nul
exit /b 0
