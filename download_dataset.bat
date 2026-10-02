@echo off
rem Download the training dataset from the shared Drive into train_data_v052\,
rem the folder the config defaults already point at.
rem
rem     download_dataset.bat               fetch (or resume) the whole dataset
rem
rem Safe to re-run: finished files are skipped, partial ones resumed. The
rem interpreter is picked the way run_cli.bat picks it, and gdown is installed
rem into it on first use. Afterwards run_cli.bat works with no path arguments.
setlocal

set "HERE=%~dp0"
set "FOLDER_URL=https://drive.google.com/drive/folders/1Cz7S-QhXRwQgsajDDyjBAC8vK30jQQTN"
set "DEST=%HERE%train_data_v052"

if not defined KCBENCH_PY if exist "%HERE%venv\Scripts\python.exe"  set "KCBENCH_PY=%HERE%venv\Scripts\python.exe"
if not defined KCBENCH_PY if exist "%HERE%.venv\Scripts\python.exe" set "KCBENCH_PY=%HERE%.venv\Scripts\python.exe"
if not defined KCBENCH_PY set "KCBENCH_PY=python"

"%KCBENCH_PY%" --version >nul 2>&1
if errorlevel 1 (
    echo download_dataset.bat: no python found. Make a venv in the repository
    echo   root ^(python -m venv venv^) or set KCBENCH_PY.
    exit /b 1
)

"%KCBENCH_PY%" -c "import gdown" >nul 2>&1
if errorlevel 1 (
    echo installing gdown ...
    "%KCBENCH_PY%" -m pip install --quiet gdown
    if errorlevel 1 (
        echo download_dataset.bat: pip install gdown failed
        exit /b 1
    )
)

if not exist "%DEST%" mkdir "%DEST%"
echo downloading into %DEST% (re-runs resume where they stopped)
"%KCBENCH_PY%" -m gdown --folder "%FOLDER_URL%" -O "%DEST%" --continue
if errorlevel 1 exit /b 1

for %%d in (data train_data metadata) do (
    if exist "%DEST%\%%d" (
        echo   %%d\ ok
    ) else (
        echo   %%d\ MISSING - the Drive layout may have changed
    )
)
echo done. run_cli.bat works with no path arguments now.
exit /b 0
