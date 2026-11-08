@echo off
REM run.bat - Mosaic Mask. Sets up .\venv the first time (and again when requirements.txt changes),
REM then starts app.py. Options pass through: run.bat --camera 1, run.bat --source clip.mp4 ...
setlocal
cd /d "%~dp0"
set "PY=%~dp0venv\Scripts\python.exe"
if not exist "%PY%" (
    echo Creating venv ...
    py -3.11 -m venv venv || goto :fail
)
fc /b requirements.txt venv\requirements.installed >nul 2>&1
if errorlevel 1 (
    echo Installing requirements ...
    "%PY%" -m pip install -r requirements.txt || goto :fail
    copy /y requirements.txt venv\requirements.installed >nul
)
if not exist "models\xseg_1.onnx" (
    echo Downloading the occlusion model, about 70 MB, once ...
    if not exist models mkdir models
    curl -L --fail -o "models\xseg_1.onnx.part" "https://huggingface.co/facefusion/models-3.1.0/resolve/main/xseg_1.onnx" || goto :fail
    move /y "models\xseg_1.onnx.part" "models\xseg_1.onnx" >nul
)
"%PY%" app.py %*
if errorlevel 1 pause
exit /b
:fail
echo Setup failed: Mosaic Mask needs Python 3.11 (MediaPipe doesn't support newer ones yet).
pause
exit /b 1
