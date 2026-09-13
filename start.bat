@echo off
setlocal EnableExtensions
title Personal Knowledge Assistant
cd /d "%~dp0"

echo.
echo  =====================================================
echo   Personal Knowledge Assistant (RAG)
echo  =====================================================
echo   Folder: %CD%
echo.

REM Prefer project venv Python (most reliable on Windows)
set "PY=.venv\Scripts\python.exe"
if not exist "%PY%" (
  echo  [INFO] Creating virtual environment .venv ...
  where python >nul 2>&1
  if errorlevel 1 (
    echo  [ERROR] Python not found on PATH.
    echo  Install Python 3.10+ from https://www.python.org/downloads/
    echo  During install, check "Add Python to PATH".
    goto :fail
  )
  python -m venv .venv
  if errorlevel 1 (
    echo  [ERROR] Failed to create .venv
    goto :fail
  )
)

if not exist "%PY%" (
  echo  [ERROR] Still missing: %PY%
  goto :fail
)

echo  Using: %PY%
"%PY%" --version

if not exist ".env" (
  if exist ".env.example" (
    echo  [WARN] No .env — copying .env.example to .env
    copy /Y ".env.example" ".env" >nul
    echo  Edit .env and set XAI_API_KEY, then run this again.
  )
)

echo.
echo  Checking dependencies (skip if already installed)...
"%PY%" -c "import streamlit,langchain,langchain_openai,chromadb,sentence_transformers,pypdf,dotenv,openai" 2>nul
if errorlevel 1 (
  echo  Installing requirements — first time can take several minutes...
  "%PY%" -m pip install --upgrade pip
  "%PY%" -m pip install -r requirements.txt
  if errorlevel 1 (
    echo  [ERROR] pip install failed.
    goto :fail
  )
) else (
  echo  Dependencies OK.
)

REM Free port 8501 if a dead process is holding it
echo.
echo  Ensuring port 8501 is free...
for /f "tokens=5" %%P in ('netstat -ano ^| findstr ":8501" ^| findstr "LISTENING"') do (
  echo  Stopping old process on 8501 (PID %%P^)...
  taskkill /F /PID %%P >nul 2>&1
)

echo.
echo  =====================================================
echo   Starting Streamlit
echo   Open:  http://localhost:8501
echo   Keep THIS window open while you use the app.
echo   Press Ctrl+C to stop the server.
echo  =====================================================
echo.

set PYTHONUNBUFFERED=1
set PYTHONASYNCIODEBUG=0
set STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

"%PY%" -m streamlit run app/main.py --server.port 8501 --browser.gatherUsageStats false
set "RC=%ERRORLEVEL%"

echo.
if not "%RC%"=="0" (
  echo  [ERROR] Streamlit exited with code %RC%
  echo  Common fixes:
  echo    - Set XAI_API_KEY in .env
  echo    - Close other apps using port 8501
  echo    - Delete .venv and run start.bat again
  goto :fail
)

echo  Streamlit stopped normally.
echo.
pause
endlocal
exit /b 0

:fail
echo.
echo  -----------------------------------------------------
echo   Something went wrong. Read the messages above.
echo   Window will stay open so you can copy the error.
echo  -----------------------------------------------------
echo.
pause
endlocal
exit /b 1
