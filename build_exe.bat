@echo off
REM Build ToonCrafterAnimator onedir on Windows 10/11 x64.
REM Requires Python 3.10 (64-bit) with pip.
setlocal EnableExtensions
cd /d "%~dp0"

if not exist ".venv-build\Scripts\python.exe" (
  py -3.10 -m venv .venv-build
)
call .venv-build\Scripts\activate.bat

python -m pip install --upgrade pip wheel
python -m pip install -r requirements.txt -r requirements-dev.txt
python -m pip install pyinstaller
REM CUDA 12.1 torch (GPU EXE). For CPU-only: use requirements-torch.txt --index-url .../cpu
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121

python packaging\generate_icon.py
python packaging\fetch_ffmpeg.py
python -m PyInstaller --noconfirm --clean packaging\ToonCrafterAnimator.spec
python packaging\make_release.py --dist dist\ToonCrafterAnimator --ffmpeg-dir packaging\ffmpeg_cache

echo.
echo Release folder: release\ToonCrafterAnimator\
echo Double-click ToonCrafterAnimator.exe after you have placed model weights locally.
endlocal
