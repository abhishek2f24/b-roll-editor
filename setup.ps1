# setup.ps1 — one-shot environment setup on Windows
# Run: .\setup.ps1

$ErrorActionPreference = "Stop"

Write-Host "`n=== AI B-roll Editor Setup ===" -ForegroundColor Cyan

# 1. Check Python
$py = (Get-Command python -ErrorAction SilentlyContinue)?.Source
if (-not $py) {
    Write-Host "Python not found. Install Python 3.11+ from https://python.org" -ForegroundColor Red
    exit 1
}
Write-Host "Python: $py" -ForegroundColor Green

# 2. Create venv
if (-not (Test-Path "venv")) {
    Write-Host "Creating virtual environment..." -ForegroundColor Yellow
    python -m venv venv
}

# 3. Activate + install
Write-Host "Installing dependencies..." -ForegroundColor Yellow
& ".\venv\Scripts\pip.exe" install --upgrade pip -q
& ".\venv\Scripts\pip.exe" install -r requirements.txt

# 4. Check FFmpeg
$ff = (Get-Command ffmpeg -ErrorAction SilentlyContinue)?.Source
if (-not $ff) {
    Write-Host "`nFFmpeg not found!" -ForegroundColor Red
    Write-Host "Install via: winget install ffmpeg" -ForegroundColor Yellow
    Write-Host "Or download from: https://ffmpeg.org/download.html" -ForegroundColor Yellow
} else {
    Write-Host "FFmpeg: $ff" -ForegroundColor Green
}

# 5. Create folders
New-Item -ItemType Directory -Force -Path "videos"  | Out-Null
New-Item -ItemType Directory -Force -Path "output"  | Out-Null
New-Item -ItemType Directory -Force -Path ".index"  | Out-Null

Write-Host "`nSetup complete!" -ForegroundColor Green
Write-Host "`nNext steps:"
Write-Host "  1. Drop your .mp4 / .mov files into the videos/ folder"
Write-Host "  2. Write your script in a .txt file (one sentence per paragraph)"
Write-Host "  3. Run:  .\venv\Scripts\python.exe agent.py run --script your_script.txt"
Write-Host ""
Write-Host "Or start the web API:"
Write-Host "  .\venv\Scripts\python.exe api.py"
