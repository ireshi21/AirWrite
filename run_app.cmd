@echo off
setlocal
cd /d "%~dp0"

if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" app.py
) else (
    echo Virtual environment not found.
    echo Create it with: py -3.8 -m venv .venv
    echo Then install requirements with: .venv\Scripts\python.exe -m pip install -r requirements.txt
    exit /b 1
)
