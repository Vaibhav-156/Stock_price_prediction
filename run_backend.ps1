# run_backend.ps1 - Runs the FastAPI backend with auto-restart on crash
# Launched by start.ps1 in its own persistent PowerShell window.

Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force

$ROOT   = Split-Path -Parent $MyInvocation.MyCommand.Path
$PYTHON = Join-Path $ROOT ".venv\Scripts\python.exe"
$BACK   = Join-Path $ROOT "backend"

Set-Location $BACK
try { $Host.UI.RawUI.WindowTitle = "Backend | Stock Prediction" } catch {}

$restarts = 0
while ($true) {
    if ($restarts -gt 0) {
        Write-Host ""
        Write-Host "[Backend] Restarting (attempt #$restarts)..." -ForegroundColor Yellow
        Start-Sleep -Seconds 3
    }

    Write-Host "[Backend] Starting on http://localhost:8000" -ForegroundColor Cyan

    & $PYTHON -m uvicorn app.main:app `
        --host 0.0.0.0 `
        --port 8000 `
        --loop asyncio `
        --timeout-keep-alive 75 `
        --limit-concurrency 200 `
        --backlog 2048 `
        --log-level info

    $restarts++
    Write-Host ""
    Write-Host "[Backend] Process exited (code=$LASTEXITCODE). Restarting in 3s... Ctrl+C to stop." -ForegroundColor Red
}
