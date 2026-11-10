@echo off
rem Start the benchmark's browser view on the Korean dataset.
rem
rem     run_webview_kr.bat
rem     run_webview_kr.bat --port 8800 --no-browser
rem
rem Arguments pass through to run_webview.bat.
call "%~dp0run_webview.bat" -c "%~dp0benchmark\config_kr.json" %*
exit /b %errorlevel%
