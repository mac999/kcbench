@echo off
rem Run any cb.py command from the repository root under the venv_lmm environment.
rem
rem     run_cli.bat                        lists the commands
rem     run_cli.bat eval -m qwen3:8b --tag base --closed-book
rem     run_cli.bat compare --base base --after ft-v1
rem
rem Arguments pass straight through. Set KCBENCH_PY to use another interpreter.
setlocal

set "HERE=%~dp0"
if not defined KCBENCH_PY set "KCBENCH_PY=D:\projects\adv\venv_lmm\Scripts\python.exe"

if not exist "%KCBENCH_PY%" (
    echo run_cli.bat: no interpreter at %KCBENCH_PY%
    echo   set KCBENCH_PY to the python you want, e.g.
    echo   set KCBENCH_PY=C:\path\to\venv\Scripts\python.exe
    exit /b 1
)

"%KCBENCH_PY%" "%HERE%benchmark\cb.py" %*
exit /b %errorlevel%
