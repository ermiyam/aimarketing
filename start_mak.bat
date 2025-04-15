@echo off
echo Starting Mak AI System...

:: Activate virtual environment if it exists
if exist "tensorflow_env\Scripts\activate.bat" (
    call tensorflow_env\Scripts\activate.bat
)

:: Run the main launcher
python src\run.py

:: Keep the window open if there's an error
if errorlevel 1 (
    echo Error occurred. Press any key to exit...
    pause
) 