@echo off
cd /d "%~dp0"
title PolySign AI Launcher

if not exist venv\Scripts\activate.bat (
    echo Creating virtual environment...
    python -m venv venv
    call venv\Scripts\activate.bat
    echo Installing requirements...
    pip install -r requirements.txt
) else (
    call venv\Scripts\activate.bat
)

:menu
cls
echo ====================================================
echo             PolySign AI - Sign Assistant
echo ====================================================
echo.
echo   1. Launch Native Desktop App (Recommended)
echo   2. Launch Web Application (Streamlit)
echo   3. Record / Collect new sign gestures
echo   4. Retrain AI Model
echo   5. Re-generate baseline dataset
echo   6. Create Desktop Shortcut
echo   7. Exit
echo.
echo ====================================================
set /p choice="Enter choice (1-7): "

if "%choice%"=="1" goto run_desktop
if "%choice%"=="2" goto run_web
if "%choice%"=="3" goto collect
if "%choice%"=="4" goto train
if "%choice%"=="5" goto generate
if "%choice%"=="6" goto shortcut
if "%choice%"=="7" exit /b
goto menu

:run_desktop
echo Starting Native Desktop Application...
python desktop_app.py
pause
goto menu

:run_web
echo Starting Streamlit Web App...
streamlit run app.py
pause
goto menu

:collect
set /p id="Enter Class ID (0-11): "
set /p s="Number of samples (e.g. 40): "
python collect_landmarks.py --class_id %id% --samples %s%
pause
goto menu

:train
echo Training GRU Model...
python train_model.py
pause
goto menu

:generate
echo Generating synthetic baseline dataset...
python generate_dataset.py
python train_model.py
pause
goto menu

:shortcut
powershell -NoProfile -ExecutionPolicy Bypass -Command "$ws = New-Object -ComObject WScript.Shell; $desktop = [Environment]::GetFolderPath('Desktop'); $s = $ws.CreateShortcut(\"$desktop\PolySign AI.lnk\"); $s.TargetPath = '%~dp0Launch_Desktop_App.bat'; $s.WorkingDirectory = '%~dp0'; $s.IconLocation = 'shell32.dll,14'; $s.Save(); Write-Host 'Shortcut created on Desktop successfully!' -ForegroundColor Green"
pause
goto menu
