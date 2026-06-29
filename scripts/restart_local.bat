@echo off
setlocal EnableExtensions
chcp 65001 >nul 2>&1

rem ============================================================
rem  WorldCup 完整重启：关闭旧 API/Dashboard -> 同步数据 -> 启动
rem  用法:
rem    scripts\restart_local.bat          完整重启（推荐）
rem    scripts\restart_local.bat fast     跳过数据同步，只重启进程
rem ============================================================

cd /d "%~dp0.."
set "ROOT=%CD%"
set "API_HOST=127.0.0.1"
set "API_PORT=8000"
set "DASH_PORT=8501"
set "API_URL=http://%API_HOST%:%API_PORT%"
set "SKIP_SYNC=0"
if /I "%~1"=="fast" set "SKIP_SYNC=1"
if /I "%~1"=="--fast" set "SKIP_SYNC=1"

echo.
echo ================================================
echo   WorldCup 一键重启
echo   项目: %ROOT%
echo ================================================
echo.

where python >nul 2>&1
if errorlevel 1 (
    echo [错误] 未找到 python，请先安装 Python 3.11+ 并 pip install -e ".[dev]"
    pause
    exit /b 1
)

echo [1/6] 关闭旧的 API / Dashboard 进程 ...
for /f "tokens=5" %%p in ('netstat -ano ^| findstr ":%API_PORT%" ^| findstr "LISTENING"') do (
    echo       结束 PID %%p ^(API 端口 %API_PORT%^)
    taskkill /F /PID %%p >nul 2>&1
)
for /f "tokens=5" %%p in ('netstat -ano ^| findstr ":%DASH_PORT%" ^| findstr "LISTENING"') do (
    echo       结束 PID %%p ^(Dashboard 端口 %DASH_PORT%^)
    taskkill /F /PID %%p >nul 2>&1
)
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0_restart_kill.ps1"
timeout /t 2 /nobreak >nul
echo       端口 %API_PORT% / %DASH_PORT% 已释放
echo.

if "%SKIP_SYNC%"=="1" (
    echo [2/6] 跳过数据同步 (fast 模式^)
    goto SET_MODEL
)

echo [2/6] 检查样例数据与 feature mart ...
if not exist "data\samples\matches.csv" (
    echo       生成样例数据 ...
    python -m scripts.export_sample_data
    if errorlevel 1 goto FAIL
)
if not exist "data\feature_mart\match_features.parquet" (
    echo       首次 ingest + build_features ...
    python -m scripts.ingest
    if errorlevel 1 goto FAIL
    python -m scripts.build_features
    if errorlevel 1 goto FAIL
)

echo [3/6] 检查模型 checkpoint ...
set "WORLDCUP_MODEL=baseline_dixon_coles"
if exist "artifacts\checkpoints\scoregen_football_v2_4060.json" set "WORLDCUP_MODEL=scoregen_football"
if exist "artifacts\checkpoints\scoregen_football_v2.json" set "WORLDCUP_MODEL=scoregen_football"
if exist "artifacts\checkpoints\scoregen_football_v1.json" set "WORLDCUP_MODEL=scoregen_football"
if not exist "artifacts\checkpoints\baseline_dixon_coles_v1.json" (
    if not exist "artifacts\checkpoints\scoregen_football_v2_4060.json" (
        echo       未找到 checkpoint，运行 baseline 训练（首次较慢）...
        python -m scripts.train
        if errorlevel 1 goto FAIL
    )
)

echo [4/6] 用模型生成 2026 隐含赔率并刷新 feature mart ...
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
echo [5/6] 启动 API (%API_URL%^)  模型: %WORLDCUP_MODEL%
set "WORLDCUP_API_URL=%API_URL%"
start "WorldCup API" /min cmd /c "cd /d \"%ROOT%\" && set WORLDCUP_MODEL=%WORLDCUP_MODEL% && set WORLDCUP_API_URL=%API_URL% && python -m scripts.serve"

echo       等待 API 就绪 ...
set "READY=0"
for /L %%i in (1,1,60) do (
    powershell -NoProfile -Command "try { $r=Invoke-RestMethod -Uri '%API_URL%/health' -TimeoutSec 2; if($r.status -eq 'ok'){ exit 0 } else { exit 1 } } catch { exit 1 }" >nul 2>&1
    if not errorlevel 1 set "READY=1" & goto API_OK
    timeout /t 1 /nobreak >nul
)
echo [错误] API 在 60 秒内未就绪，请查看最小化窗口 "WorldCup API"
pause
exit /b 1

:API_OK
powershell -NoProfile -Command "try { $p=(Invoke-RestMethod '%API_URL%/openapi.json').paths; $has=$p.'/predict/batch' -ne $null; if($has){ Write-Host '       OpenAPI: /predict/batch 已加载' } else { Write-Host '       [警告] 仍是旧 API，缺少 /predict/batch' -ForegroundColor Yellow } } catch { Write-Host '       [警告] 无法读取 OpenAPI' -ForegroundColor Yellow }"
echo.

echo [6/6] 启动 Dashboard (关闭本窗口或 Ctrl+C 不会自动关 API^)
echo       浏览器: http://localhost:%DASH_PORT%
echo.
set "WORLDCUP_API_URL=%API_URL%"
python -m streamlit run "src\worldcup\dashboard\app.py" --server.port %DASH_PORT%
goto END

:FAIL
echo.
echo [错误] 上一步失败，请向上滚动查看日志。
pause
exit /b 1

:END
echo.
echo Dashboard 已退出。正在关闭 API ...
for /f "tokens=5" %%p in ('netstat -ano ^| findstr ":%API_PORT%" ^| findstr "LISTENING"') do taskkill /F /PID %%p >nul 2>&1
echo 完成。
pause
endlocal
exit /b 0
