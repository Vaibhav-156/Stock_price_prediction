# run_frontend.ps1 - Runs the Next.js frontend with auto-restart on crash
# Launched by start.ps1 in its own persistent PowerShell window.

Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force

$ROOT  = Split-Path -Parent $MyInvocation.MyCommand.Path
$FRONT = Join-Path $ROOT "frontend"

Set-Location $FRONT
try { $Host.UI.RawUI.WindowTitle = "Frontend | Stock Prediction" } catch {}

$env:NEXT_TELEMETRY_DISABLED = "1"
$env:NODE_ENV = "development"

$restarts = 0
while ($true) {
    if ($restarts -gt 0) {
        Write-Host ""
        Write-Host "[Frontend] Restarting (attempt #$restarts)..." -ForegroundColor Yellow
        Start-Sleep -Seconds 3
    }

    Write-Host "[Frontend] Starting on http://localhost:3000" -ForegroundColor Cyan

    npm run dev

    $restarts++
    Write-Host ""
    Write-Host "[Frontend] Process exited (code=$LASTEXITCODE). Restarting in 3s... Ctrl+C to stop." -ForegroundColor Red
}
