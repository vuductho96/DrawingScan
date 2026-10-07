@echo off
chcp 65001 >nul
cd /d "%~dp0"

set PY_EXE="%~dp0.venv\Scripts\python.exe"
if not exist %PY_EXE% (
    set PY_EXE="%~dp0..\.venv\Scripts\python.exe"
)
if not exist %PY_EXE% (
    set PY_EXE=python
)

%PY_EXE% -c "import fitz, onnxruntime, PIL, numpy" >nul 2>&1
if errorlevel 1 (
    echo [MiniTool] Dang cai dat thu vien can thiet tu requirements.txt...
    %PY_EXE% -m pip install -r requirements.txt
)

start "" "http://127.0.0.1:8090"
%PY_EXE% server.py --port 8090
pause
