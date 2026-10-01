@echo off
rem Run any cb.py command from the repository root.
rem
rem     run_cli.bat                        lists the commands
rem     run_cli.bat eval -m qwen3:8b --tag base --closed-book
rem     run_cli.bat compare --base base --after ft-v1
rem
rem Arguments pass straight through. The interpreter is KCBENCH_PY if set,
rem else a venv\ or .venv\ inside the repository, else the python on PATH.
setlocal

set "HERE=%~dp0"
if not defined KCBENCH_PY if exist "%HERE%venv\Scripts\python.exe"  set "KCBENCH_PY=%HERE%venv\Scripts\python.exe"
if not defined KCBENCH_PY if exist "%HERE%.venv\Scripts\python.exe" set "KCBENCH_PY=%HERE%.venv\Scripts\python.exe"
if not defined KCBENCH_PY set "KCBENCH_PY=python"

"%KCBENCH_PY%" --version >nul 2>&1
if errorlevel 1 (
    echo run_cli.bat: no python found. Make a venv in the repository root
    echo   ^(python -m venv venv^) or set KCBENCH_PY.
    exit /b 1
)

"%KCBENCH_PY%" "%HERE%benchmark\cb.py" %*
exit /b %errorlevel%
