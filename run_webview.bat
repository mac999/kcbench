@echo off
rem Start the benchmark's browser view from the repository root.
rem
rem     run_webview.bat
rem     run_webview.bat --port 8800 --no-browser
rem
rem Arguments pass through to benchmark\webview.bat, which resolves the
rem interpreter (KCBENCH_PY, default venv_lmm) and checks Flask is installed.
call "%~dp0benchmark\webview.bat" %*
exit /b %errorlevel%
