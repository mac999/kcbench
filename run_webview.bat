@echo off
rem Start the benchmark's browser view from the repository root.
rem
rem     run_webview.bat
rem     run_webview.bat --port 8800 --no-browser
rem     run_webview.bat -c benchmark\config_kr.json
rem
rem Without -c/--config it opens the English dataset (benchmark\config_us.json);
rem run_webview_kr.bat opens the Korean one.
rem
rem Arguments pass through to benchmark\webview.bat, which resolves the
rem interpreter (KCBENCH_PY, default venv_lmm) and checks Flask is installed.
setlocal

set "HASCFG="
for %%a in (%*) do (
    if /i "%%~a"=="-c" set "HASCFG=1"
    if /i "%%~a"=="--config" set "HASCFG=1"
)

if defined HASCFG (
    call "%~dp0benchmark\webview.bat" %*
) else (
    call "%~dp0benchmark\webview.bat" -c "%~dp0benchmark\config_us.json" %*
)
exit /b %errorlevel%
