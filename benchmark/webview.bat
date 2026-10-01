@echo off
rem Start the benchmark's browser view.
rem
rem     webview.bat
rem     webview.bat --port 8800 --no-browser
rem
rem Every argument is passed straight to `cb.py webview`. The interpreter is
rem KCBENCH_PY if set, else a venv\ or .venv\ in the repository root, else the
rem python on PATH. Flask is checked before starting.
setlocal

set "HERE=%~dp0"
if not defined KCBENCH_PY if exist "%HERE%..\venv\Scripts\python.exe"  set "KCBENCH_PY=%HERE%..\venv\Scripts\python.exe"
if not defined KCBENCH_PY if exist "%HERE%..\.venv\Scripts\python.exe" set "KCBENCH_PY=%HERE%..\.venv\Scripts\python.exe"
if not defined KCBENCH_PY set "KCBENCH_PY=python"

"%KCBENCH_PY%" --version >nul 2>&1
if errorlevel 1 (
    echo webview.bat: no python found. Make a venv in the repository root
    echo   ^(python -m venv venv^) or set KCBENCH_PY.
    exit /b 1
)

"%KCBENCH_PY%" -c "import flask" >nul 2>&1
if errorlevel 1 (
    echo webview.bat: Flask is not installed in that environment. Nothing else needs it:
    echo   "%KCBENCH_PY%" -m pip install flask
    exit /b 1
)

"%KCBENCH_PY%" "%HERE%cb.py" webview %*
exit /b %errorlevel%
