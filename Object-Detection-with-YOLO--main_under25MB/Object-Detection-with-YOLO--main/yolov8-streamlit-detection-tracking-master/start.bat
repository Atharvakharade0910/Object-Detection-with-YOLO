@echo off
cd /d "%~dp0"
if exist "runtime-python.txt" (
  set /p YOLO_PYTHON=<runtime-python.txt
  goto configured
)
if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" app.py
) else (
  python app.py
)
pause
exit /b
:configured
"%YOLO_PYTHON%" app.py
pause
