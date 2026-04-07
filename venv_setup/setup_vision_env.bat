@echo off
echo ========================================================
echo Setting up Vision Virtual Environment (DINO, CLIP, OCR)
echo ========================================================

:: 1. Create and activate a separate virtual environment inside venv_setup to prevent conflict with Gemini/FastAPI core dependencies
cd %~dp0
python -m venv venv_vision
call venv_vision\Scripts\activate.bat

echo [1/4] Upgrading pip...
python -m pip install --upgrade pip

echo [2/4] Installing PyTorch with CUDA 12.1 support...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

echo [3/4] Installing OpenAI CLIP...
pip install git+https://github.com/openai/CLIP.git

echo [4/4] Installing Grounding DINO from Source...
:: Note: Grounding DINO requires Microsoft C++ Build Tools installed on Windows.
pip install git+https://github.com/IDEA-Research/GroundingDINO.git

echo [Optional] Installing PaddleOCR
pip install paddlepaddle paddleocr

echo ========================================================
echo Vision Environment Setup Complete!
echo To use this environment, run: venv_setup\venv_vision\Scripts\activate
echo ========================================================

