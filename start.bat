@echo off
setlocal
cd /d "%~dp0"
title AutoKolpak AI
echo ==========================================
echo          AutoKolpak AI v2
echo ==========================================
echo Folder: %CD%
echo.

echo [1/5] Checking port 8000...
netstat -ano | findstr ":8000" | findstr "LISTENING" >nul
if not errorlevel 1 (
  echo [ERROR] Port 8000 is already in use. An old copy of the project is probably running.
  echo         Close its console window or kill python.exe in Task Manager, then run again.
  goto :end
)

echo [2/5] Looking for Python...
where py >nul 2>nul
if errorlevel 1 (
  echo [ERROR] Python Launcher "py" not found. Install Python 3.12 and tick "Add to PATH".
  goto :end
)

echo [3/5] Preparing virtual environment...
if not exist ".venv_gpu\Scripts\python.exe" if not exist ".venv\Scripts\python.exe" (
  py -3.12 -m venv .venv
  if errorlevel 1 (
    echo [ERROR] Could not create .venv. Is Python 3.12 installed?
    goto :end
  )
)
if exist ".venv_gpu\Scripts\activate.bat" (
  call ".venv_gpu\Scripts\activate.bat"
) else (
  call ".venv\Scripts\activate.bat"
)

echo [4/5] Checking dependencies...
python -c "import fastapi, uvicorn, PIL, multipart" >nul 2>nul
if errorlevel 1 (
  echo Installing dependencies - internet needed, first run only...
  python -m pip install -r requirements-web.txt
  if errorlevel 1 (
    echo [ERROR] Dependency installation failed.
    goto :end
  )
)

if not exist ".env" echo [WARN] .env not found - create it from .env.example with your API key.

echo [5/5] Starting server at http://127.0.0.1:8000  
start "" cmd /c "timeout /t 4 >nul & start http://127.0.0.1:8000"
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
echo.
echo [INFO] Server stopped. See messages above.

:end
echo.
pause
