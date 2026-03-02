# start.ps1 - One-click launcher for Stock Prediction Platform
# Run: powershell -ExecutionPolicy Bypass -File .\start.ps1

Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force

$ROOT   = Split-Path -Parent $MyInvocation.MyCommand.Path
$PYTHON = Join-Path $ROOT ".venv\Scripts\python.exe"
$FRONT  = Join-Path $ROOT "frontend"

Set-Location $ROOT

# Sanity checks
if (-not (Test-Path $PYTHON)) {
    Write-Host "ERROR: .venv not found. Run: python -m venv .venv && .venv\Scripts\pip install -r backend\requirements.txt" -ForegroundColor Red
    Read-Host "Press Enter to exit"; exit 1
}
if (-not (Get-Command npm -ErrorAction SilentlyContinue)) {
    Write-Host "ERROR: npm not found. Install Node.js from https://nodejs.org" -ForegroundColor Red
    Read-Host "Press Enter to exit"; exit 1
}

# Install frontend deps on first run
if (-not (Test-Path (Join-Path $FRONT "node_modules"))) {
    Write-Host "Installing frontend dependencies (first run)..." -ForegroundColor Yellow
    Push-Location $FRONT; npm install; Pop-Location
}

# Kill anything already on these ports
Write-Host "Clearing old processes..." -ForegroundColor DarkGray
Get-Process -Name "node","python" -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
Start-Sleep -Seconds 1

# Open backend in its own window with auto-restart loop
Write-Host "Starting Backend  (port 8000)..." -ForegroundColor Cyan
$backendScript = Join-Path $ROOT "run_backend.ps1"
Start-Process powershell.exe -ArgumentList "-NoExit -ExecutionPolicy Bypass -File `"$backendScript`""

# Open frontend in its own window with auto-restart loop
Write-Host "Starting Frontend (port 3000)..." -ForegroundColor Cyan
$frontendScript = Join-Path $ROOT "run_frontend.ps1"
Start-Process powershell.exe -ArgumentList "-NoExit -ExecutionPolicy Bypass -File `"$frontendScript`""

Write-Host ""
Write-Host "  Both servers are starting in separate windows." -ForegroundColor Green
Write-Host "  Backend  -> http://localhost:8000" -ForegroundColor Green
Write-Host "  Frontend -> http://localhost:3000" -ForegroundColor Green
Write-Host "  To stop: run .\stop.ps1" -ForegroundColor DarkGray
Write-Host ""
