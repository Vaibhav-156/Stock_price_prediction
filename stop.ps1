# stop.ps1 - Clean shutdown of backend and frontend servers
# Usage: powershell -ExecutionPolicy Bypass -File .\stop.ps1

Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force

function Kill-Port {
    param([int]$Port, [string]$Label)
    $found = $false
    $rows = (netstat -ano 2>$null) | Select-String (":$Port\s")
    foreach ($row in $rows) {
        $p = ($row.ToString() -split '\s+')[-1]
        if ($p -match '^\d+$' -and [int]$p -gt 0) {
            try {
                Stop-Process -Id ([int]$p) -Force -ErrorAction Stop
                Write-Host "  [$Label] Killed PID $p on port $Port" -ForegroundColor Green
                $found = $true
            } catch {
                Write-Host "  [$Label] Could not kill PID $p" -ForegroundColor Yellow
            }
        }
    }
    if (-not $found) {
        Write-Host "  [$Label] Nothing running on port $Port" -ForegroundColor DarkGray
    }
}

Write-Host ""
Write-Host "  Stopping Stock Prediction Platform..." -ForegroundColor Cyan
Write-Host ""

Kill-Port 8000 "Backend"
Kill-Port 3000 "Frontend"

# Belt-and-suspenders: also scan by process name
foreach ($name in "uvicorn", "python") {
    Get-Process -Name $name -ErrorAction SilentlyContinue | ForEach-Object {
        try {
            $_.Kill()
            Write-Host "  [cleanup] Killed $name PID $($_.Id)" -ForegroundColor Green
        } catch {}
    }
}

Get-Process -Name "node" -ErrorAction SilentlyContinue | ForEach-Object {
    try {
        $_.Kill()
        Write-Host "  [cleanup] Killed node PID $($_.Id)" -ForegroundColor Green
    } catch {}
}

# Clean up any watchdog jobs
Get-Job -Name "*Watchdog*" -ErrorAction SilentlyContinue | ForEach-Object {
    Stop-Job  $_ -ErrorAction SilentlyContinue
    Remove-Job $_ -Force -ErrorAction SilentlyContinue
    Write-Host "  [watchdog] Stopped job: $($_.Name)" -ForegroundColor Green
}

Write-Host ""
Write-Host "  All servers stopped." -ForegroundColor Cyan
Start-Sleep -Seconds 1
