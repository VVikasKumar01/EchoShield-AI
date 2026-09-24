@echo off
setlocal enabledelayedexpansion

title EchoShield AI - Dependency Installer and Setup
color 0B

echo =====================================================================
echo                ECHO-SHIELD AI : INSTALLATION AND SETUP
echo       Real-Time Full-Band (48kHz) Speech Enhancement Engine
echo =====================================================================
echo.

:: Ensure current working directory is the script folder
cd /d "%~dp0"

:: 1. Check for Python
echo [*] Checking for Python installation...
set "PY_CMD="

python --version >nul 2>&1
if %ERRORLEVEL% equ 0 (
    set "PY_CMD=python"
) else (
    py -3 --version >nul 2>&1
    if %ERRORLEVEL% equ 0 (
        set "PY_CMD=py -3"
    )
)

if "%PY_CMD%"=="" (
    color 0C
    echo [ERROR] Python was not found on your system!
    echo.
    echo Please install Python 3.9, 3.10, or 3.11 from:
    echo https://www.python.org/downloads/
    echo.
    echo IMPORTANT: Make sure to check "Add Python to PATH" during installation.
    echo.
    pause
    exit /b 1
)

for /f "tokens=*" %%v in ('%PY_CMD% --version 2^>^&1') do echo [+] Found %%v

:: 2. Check / Create Virtual Environment (.venv)
echo.
echo [*] Setting up Python Virtual Environment (.venv)...
if not exist ".venv\Scripts\python.exe" (
    echo [*] Creating fresh .venv environment...
    %PY_CMD% -m venv .venv
    if %ERRORLEVEL% neq 0 (
        color 0C
        echo [ERROR] Failed to create virtual environment.
        pause
        exit /b 1
    )
    echo [+] Virtual environment created successfully.
) else (
    echo [+] Existing virtual environment found.
)

:: 3. Upgrade pip
echo.
echo [*] Upgrading pip package manager...
.\.venv\Scripts\python.exe -m pip install --upgrade pip --quiet

:: 4. Install PyTorch & Core Libraries
echo.
echo [*] Installing PyTorch, Audio DSP, and Deep Learning modules...
echo [*] This may take a few minutes depending on your internet speed...
echo.

.\.venv\Scripts\python.exe -m pip install torch torchaudio sounddevice soundfile numpy scipy loguru requests packaging sympy icecream cffi

if %ERRORLEVEL% neq 0 (
    color 0C
    echo.
    echo [ERROR] Dependency installation encountered an issue.
    pause
    exit /b 1
)

:: 5. Install from requirements.txt if present
if exist "requirements.txt" (
    echo.
    echo [*] Verifying requirements.txt...
    .\.venv\Scripts\python.exe -m pip install -r requirements.txt --quiet
)

:: 6. Verification Test
echo.
echo =====================================================================
echo [*] Running system and model verification test...
echo =====================================================================
.\.venv\Scripts\python.exe -c "import sys; from pathlib import Path; sys.path.insert(0, str(Path('EchoShield').resolve())); from df.enhance import init_df; model, df_state, *rest = init_df(); print('\n[SUCCESS] EchoShield DeepFilterNet3 Model and DSP Engine are ready!')"

if %ERRORLEVEL% equ 0 (
    color 0A
    echo.
    echo =====================================================================
    echo    SUCCESS: All EchoShield AI dependencies are installed!
    echo =====================================================================
    echo.
    echo Quick Run Options:
    echo   1. Web Dashboard     :  .\.venv\Scripts\python.exe web_app.py
    echo   2. Record Live Voice :  .\.venv\Scripts\python.exe record_and_enhance.py --duration 5
    echo   3. Live Headphone Mic:  .\.venv\Scripts\python.exe live_deepfilter.py --hear-myself
    echo.
    set /p LAUNCH="Would you like to launch the Web App now? (Y/N): "
    if /i "!LAUNCH!"=="Y" (
        echo [*] Starting Web App on http://localhost:7860 ...
        start http://localhost:7860
        .\.venv\Scripts\python.exe web_app.py
    )
) else (
    color 0E
    echo.
    echo [!] Installation completed, but verification test gave a warning.
    echo You can still try running: .\.venv\Scripts\python.exe web_app.py
)

echo.
echo Press any key to exit.
pause >nul
