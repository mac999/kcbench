@echo off
rem Start the benchmark's browser view under the venv_lmm environment.
rem
rem     webview.bat
rem     webview.bat --port 8800 --no-browser
rem
rem Every argument is passed straight to `cb.py webview`. Set KCBENCH_PY to
rem point at a different interpreter; the default is the venv_lmm one.
setlocal

set "HERE=%~dp0"
if not defined KCBENCH_PY set "KCBENCH_PY=D:\projects\adv\venv_lmm\Scripts\python.exe"

if not exist "%KCBENCH_PY%" (
    echo webview.bat: no interpreter at %KCBENCH_PY%
    echo   set KCBENCH_PY to the python you want, e.g.
    echo   set KCBENCH_PY=C:\path\to\venv\Scripts\python.exe
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
