@echo off
setlocal
cd /d "%~dp0"
set HF_HOME=%CD%\.cache\huggingface
set HF_HUB_DISABLE_SYMLINKS_WARNING=1
if exist .venv\Scripts\python.exe goto install
where py >nul 2>nul
if not errorlevel 1 (
  py -3.12 -m venv .venv
) else (
  python -c "import sys; assert sys.version_info[:2] == (3,12), 'Install Python 3.12 and enable Add to PATH'" >nul 2>nul
  if errorlevel 1 goto missing_python
  python -m venv .venv
)
if errorlevel 1 goto failed
:install
if exist .venv\.groundeddesk-installed goto configured
.venv\Scripts\python.exe -m ensurepip --upgrade
if errorlevel 1 goto failed
.venv\Scripts\python.exe -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
if errorlevel 1 goto failed
.venv\Scripts\python.exe -m pip install -r requirements.txt
if errorlevel 1 goto failed
echo installed>.venv\.groundeddesk-installed
:configured
if not exist .env copy .env.example .env >nul
.venv\Scripts\python.exe -c "from app.config import Settings; raise SystemExit(0 if Settings().groq_api_key.get_secret_value() else 1)"
if errorlevel 1 (
  echo Add your GROQ_API_KEY to .env, save it, then close Notepad to continue.
  start /wait notepad.exe .env
)
echo Starting GroundedDesk. Close the UI or press Ctrl+C here to stop both servers.
.venv\Scripts\python.exe scripts\launch.py
if errorlevel 1 goto failed
exit /b 0
:missing_python
echo Python 3.12 was not found. Install it from python.org with Add to PATH enabled.
pause
exit /b 1
:failed
echo GroundedDesk could not start. Read the error above and the README Windows notes.
pause
exit /b 1
