@echo off
rem Download the training dataset from the shared Drive.
rem
rem     download_dataset.bat               v052 into train_data_v052\ (the
rem                                        default config_kr.json)
rem     download_dataset.bat v053          also the v0.5.3 regeneration into
rem                                        train_data_v053\train_data, for
rem                                        -c config_kr_v053.json
rem
rem Both releases share one corpus and catalogue, so data\ and metadata\ always
rem go to train_data_v052\, where config_kr_v053.json inherits them from.
rem
rem Safe to re-run: finished files are skipped, partial ones resumed. The
rem interpreter is picked the way run_cli.bat picks it, and gdown is installed
rem into it on first use. Afterwards run_cli.bat works with no path arguments.
setlocal

set "HERE=%~dp0"
rem Subfolders of the shared Drive folder 1Cz7S-QhXRwQgsajDDyjBAC8vK30jQQTN,
rem fetched one by one so the release not asked for is not downloaded with it
set "DRIVE=https://drive.google.com/drive/folders"
set "DATA_ID=1byKtnR64tYzP8p0AviZQRwFvFwairrjH"
set "META_ID=1hSZqIjddwi8PmW_Q0KebmD0bb1ctK5qD"
set "V052_ID=1JqkTvGBbVFHsaK4juMqiJ_9GyUv6C_ft"
set "V053_ID=1kZAL3ZstGPA5y-SDeFvbnB9CzcnMv2rX"
set "DEST=%HERE%train_data_v052"

set "RELEASE=%~1"
if "%RELEASE%"=="" set "RELEASE=v052"
if /i "%RELEASE%"=="v052" (
    set "GEN_ID=%V052_ID%"
    set "GEN_DEST=%DEST%\train_data"
) else if /i "%RELEASE%"=="v053" (
    set "GEN_ID=%V053_ID%"
    set "GEN_DEST=%HERE%train_data_v053\train_data"
) else (
    echo usage: download_dataset.bat [v052^|v053]
    exit /b 1
)

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

echo downloading %RELEASE% (re-runs resume where they stopped)
call :fetch "%DATA_ID%" "%DEST%\data"     || exit /b 1
call :fetch "%META_ID%" "%DEST%\metadata" || exit /b 1
call :fetch "%GEN_ID%"  "%GEN_DEST%"      || exit /b 1

for %%d in ("%DEST%\data" "%DEST%\metadata" "%GEN_DEST%") do (
    dir /s /b /a-d "%%~d" >nul 2>&1 && (echo   %%~d ok) || (echo   %%~d EMPTY - the Drive layout may have changed)
)
if /i "%RELEASE%"=="v053" (
    echo done. pass -c config_kr_v053.json to cb.py to use the v053 answer key.
) else (
    echo done. run_cli.bat works with no path arguments now.
)
exit /b 0

rem Drive folder id, local folder. No trailing backslash on -O: gdown then
rem writes the folder's contents there.
:fetch
if not exist "%~2" mkdir "%~2"
"%KCBENCH_PY%" -m gdown --folder "%DRIVE%/%~1" -O "%~2" --continue
if errorlevel 1 exit /b 1
exit /b 0
