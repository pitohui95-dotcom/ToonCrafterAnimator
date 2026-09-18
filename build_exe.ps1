# Build ToonCrafterAnimator onedir on Windows 10/11 x64.
# Requires Python 3.10 (64-bit).
$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

$py = Get-Command py -ErrorAction SilentlyContinue
if (-not $py) { throw "Python launcher 'py' not found. Install Python 3.10 x64." }

if (-not (Test-Path ".venv-build\Scripts\python.exe")) {
    py -3.10 -m venv .venv-build
}
& .\.venv-build\Scripts\Activate.ps1

python -m pip install --upgrade pip wheel
python -m pip install -r requirements.txt -r requirements-dev.txt
python -m pip install pyinstaller
# CUDA 12.1 torch (GPU EXE). For CPU-only:
#   pip install -r requirements-torch.txt --index-url https://download.pytorch.org/whl/cpu
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121

python packaging\generate_icon.py
python packaging\fetch_ffmpeg.py
python -m PyInstaller --noconfirm --clean packaging\ToonCrafterAnimator.spec
python packaging\make_release.py --dist dist\ToonCrafterAnimator --ffmpeg-dir packaging\ffmpeg_cache

Write-Host ""
Write-Host "Release folder: release\ToonCrafterAnimator\"
Write-Host "Double-click ToonCrafterAnimator.exe after you have placed model weights locally."
