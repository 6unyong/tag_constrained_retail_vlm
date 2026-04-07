@echo off
echo ========================================================
echo Installing Ollama via Winget for Windows...
echo ========================================================

winget install -e --id Ollama.Ollama --accept-source-agreements --accept-package-agreements
if %errorlevel% neq 0 (
    echo If Winget fails, please download Ollama manually from https://ollama.com/download
    pause
    exit /b %errorlevel%
)

echo.
echo ========================================================
echo Starting Ollama Background Service...
echo ========================================================
start "" ollama serve
timeout /t 5

echo.
echo ========================================================
echo Pulling LLaVA 1.5 7B model (4.5GB)...
echo This will take some time depending on your internet connection.
echo ========================================================
ollama pull llava

echo.
echo ========================================================
echo Ollama and LLaVA setup complete!
echo ========================================================
pause
