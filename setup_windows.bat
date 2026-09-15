@echo off
title PDF RAG Chatbot - Automatic Setup and Launch
setlocal enabledelayedexpansion

REM ================================================================
REM  ONE-CLICK setup for the PDF RAG Chatbot on Windows 10/11.
REM  Installs Git + Python 3.12 (if missing), downloads the app,
REM  installs all dependencies, asks for your key, and launches it.
REM  Just DOUBLE-CLICK this file. Re-running it updates + relaunches.
REM ================================================================

set "PROJECT_DIR=%USERPROFILE%\PDF_RAG_Chatbot"
set "REPO_URL=https://github.com/yaanx-in/PDF_RAG_Chatbot.git"
set "REPO_ZIP=https://github.com/yaanx-in/PDF_RAG_Chatbot/archive/refs/heads/main.zip"
set "PY312=%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
set "GITEXE=%ProgramFiles%\Git\cmd\git.exe"

echo(
echo ==================================================
echo    PDF RAG Chatbot  -  automatic setup + launch
echo ==================================================
echo    This will set everything up for you. It can take
echo    5-15 minutes the first time. Please keep this
echo    window open and just wait when it says so.
echo ==================================================
echo(

REM ---------- 0. Make sure winget exists ----------
where winget >nul 2>&1
if errorlevel 1 (
  echo(
  echo Your Windows is missing "winget", which is needed to auto-install things.
  echo Please open the Microsoft Store, update "App Installer", then run this again.
  echo (Or install Git and Python 3.12 manually, then re-run this file.)
  pause & exit /b 1
)

REM ---------- 1. Ensure Git ----------
where git >nul 2>&1 && goto havegit
if exist "%GITEXE%" goto havegit
echo [1/6] Installing Git (one time)...
winget install -e --id Git.Git --accept-source-agreements --accept-package-agreements
:havegit
where git >nul 2>&1 && (set "GITCMD=git") || (set GITCMD="%GITEXE%")
echo [1/6] Git ready.

REM ---------- 2. Ensure Python 3.12 ----------
if exist "%PY312%" goto havepy
py -3.12 --version >nul 2>&1 && goto havepy
echo [2/6] Installing Python 3.12 (one time, a few minutes)...
winget install -e --id Python.Python.3.12 --accept-source-agreements --accept-package-agreements
:havepy
if exist "%PY312%" (set PYCMD="%PY312%") else (set PYCMD=py -3.12)
echo [2/6] Checking Python...
%PYCMD% --version
if errorlevel 1 (
  echo(
  echo Python 3.12 is not ready yet. Please close this window and
  echo double-click the file again (Windows just needs a fresh start).
  pause & exit /b 1
)

REM ---------- 3. Download or update the app ----------
if exist "%PROJECT_DIR%\.git" (
  echo [3/6] Updating the app...
  cd /d "%PROJECT_DIR%"
  %GITCMD% pull
) else (
  echo [3/6] Downloading the app...
  %GITCMD% clone "%REPO_URL%" "%PROJECT_DIR%"
  if errorlevel 1 (
    echo Git download failed - trying direct download instead...
    pushd "%USERPROFILE%"
    curl -L -o pdf_app.zip "%REPO_ZIP%"
    if errorlevel 1 (echo ERROR: could not download the app - check internet. & pause & exit /b 1)
    tar -xf pdf_app.zip
    if exist "%PROJECT_DIR%" rmdir /s /q "%PROJECT_DIR%"
    move "PDF_RAG_Chatbot-main" "%PROJECT_DIR%" >nul
    del pdf_app.zip
    popd
  )
)
cd /d "%PROJECT_DIR%"
echo [3/6] App files ready.

REM ---------- 4. Virtual environment ----------
if exist "venv\Scripts\python.exe" goto haveenv
echo [4/6] Creating a private Python environment...
%PYCMD% -m venv venv
if errorlevel 1 (echo ERROR: could not create the environment. & pause & exit /b 1)
:haveenv
call "venv\Scripts\activate.bat"
echo [4/6] Environment ready.

REM ---------- 5. Dependencies ----------
echo [5/6] Installing dependencies. This is the LONG step:
echo       a ~1-2 GB download. Please wait - do not close the window.
python -m pip install --upgrade pip >nul
pip install -r requirements.txt
if errorlevel 1 (echo ERROR: dependency install failed. & pause & exit /b 1)
echo [5/6] Dependencies installed.

REM ---------- 5b. Groq key (asked once) ----------
if not exist ".streamlit\secrets.toml" (
  echo(
  echo Paste your Groq API key, then press Enter.
  echo Get one free at: https://console.groq.com/keys
  echo (Or just press Enter to skip - the app still works with basic answers.)
  set /p "GROQKEY=Groq API key: "
  if not exist ".streamlit" mkdir ".streamlit"
  if not "!GROQKEY!"=="" (
    > ".streamlit\secrets.toml" echo GROQ_API_KEY = "!GROQKEY!"
    echo Key saved.
  )
)

REM ---------- 6. Launch ----------
echo(
echo ==================================================
echo   [6/6] Starting the app!
echo   Your web browser will open at:  http://localhost:8501
echo   Keep THIS black window open while using the app.
echo   Close this window to stop the app.
echo ==================================================
echo(
streamlit run app.py

pause
endlocal
