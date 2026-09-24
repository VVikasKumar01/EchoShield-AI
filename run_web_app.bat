@echo off
title EchoShield AI - Web Dashboard
color 0A

cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo [!] Virtual environment not found. Running installer first...
    call install_dependencies.bat
)

echo [*] Starting EchoShield Web App on http://localhost:7860 ...
start http://localhost:7860
.\.venv\Scripts\python.exe web_app.py 7860
pause
