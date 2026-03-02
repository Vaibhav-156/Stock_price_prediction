# watchdog.ps1 - Auto-restart watchdog for backend + frontend servers
# Run via start.ps1 or: powershell -ExecutionPolicy Bypass -File .\watchdog.ps1

$ROOT   = Split-Path -Parent $MyInvocation.MyCommand.Path
$LOGS   = Join-Path $ROOT "logs"
$PYTHON = Join-Path $ROOT ".venv\Scripts\python.exe"

# Create logs dir if missing
$null = New-Item -ItemType Directory -Path $LOGS -Force

function Write-Log {
    param([string]$File, [string]$Msg)
    $ts = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Add-Content -Path (Join-Path $LOGS $File) -Value "[$ts] $Msg" -Encoding UTF8
}

function Rotate-Log {
    param([string]$File)
    $path = Join-Path $LOGS $File
    if ((Test-Path $path) -and (Get-Item $path).Length -gt 10MB) {
        $archive = $path -replace '\.log$', ("_" + (Get-Date -Format "yyyyMMdd_HHmmss") + ".log")
        Move-Item $path $archive
    }
}

function Port-Alive {
    param([int]$Port)
    try {
        $tcp = New-Object System.Net.Sockets.TcpClient
        $r = $tcp.ConnectAsync("127.0.0.1", $Port).Wait(600)
        $alive = $tcp.Connected
        $tcp.Close()
        return $alive
    } catch { return $false }
}

function Wait-Port {
    param([int]$Port, [int]$TimeoutSec = 45)
    $deadline = (Get-Date).AddSeconds($TimeoutSec)
    while ((Get-Date) -lt $deadline) {
        if (Port-Alive $Port) { return $true }
        Start-Sleep -Milliseconds 500
    }
    return $false
}

function Http-OK {
    param([string]$Url)
    try {
        $r = Invoke-WebRequest $Url -UseBasicParsing -TimeoutSec 4 -ErrorAction Stop
        return ($r.StatusCode -lt 400)
    } catch { return $false }
}

# Shared state (synchronized hashtable is thread-safe for reads/writes from jobs)
$State = [hashtable]::Synchronized(@{
    BackendPID       = -1
    FrontendPID      = -1
    BackendUp        = $false
    FrontendUp       = $false
    BackendRestarts  = 0
    FrontendRestarts = 0
    BackendStatus    = "starting"
    FrontendStatus   = "starting"
    LastBackendErr   = ""
    LastFrontendErr  = ""
    Started          = Get-Date
})

# ---- Backend watchdog job ---------------------------------------------------
$BackendJob = Start-Job -Name "BackendWatchdog" -ScriptBlock {
    param($ROOT, $PYTHON, $LOGS, $State)

    function Write-Log {
        param([string]$File, [string]$Msg)
        $ts = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
        Add-Content -Path (Join-Path $LOGS $File) -Value "[$ts] $Msg" -Encoding UTF8
    }

    function Rotate-Log {
        param([string]$File)
        $path = Join-Path $LOGS $File
        if ((Test-Path $path) -and (Get-Item $path).Length -gt 10MB) {
            $archive = $path -replace '\.log$', ("_" + (Get-Date -Format "yyyyMMdd_HHmmss") + ".log")
            Move-Item $path $archive
        }
    }

    $BACK    = Join-Path $ROOT "backend"
    $OUT     = "backend_out.log"
    $ERR     = "backend_err.log"
    $BASE_DELAY = 3

    $State.BackendStatus = "starting"
    Write-Log $OUT "Backend watchdog started."

    while ($true) {
        Rotate-Log $OUT
        Rotate-Log $ERR

        $uvArgs = "-m uvicorn app.main:app --host 0.0.0.0 --port 8000 " +
                  "--loop asyncio --timeout-keep-alive 75 " +
                  "--timeout-graceful-shutdown 10 --limit-concurrency 200 " +
                  "--backlog 2048 --log-level info"

        $psi = New-Object System.Diagnostics.ProcessStartInfo
        $psi.FileName               = $PYTHON
        $psi.Arguments              = $uvArgs
        $psi.WorkingDirectory       = $BACK
        $psi.UseShellExecute        = $false
        $psi.RedirectStandardOutput = $true
        $psi.RedirectStandardError  = $true
        $psi.CreateNoWindow         = $true

        $proc = New-Object System.Diagnostics.Process
        $proc.StartInfo = $psi

        $outPath = Join-Path $LOGS $OUT
        $errPath = Join-Path $LOGS $ERR

        $oHandler = [System.Diagnostics.DataReceivedEventHandler]{
            param($s, $e)
            if ($e.Data) { Add-Content -Path $outPath -Value $e.Data -Encoding UTF8 }
        }
        $eHandler = [System.Diagnostics.DataReceivedEventHandler]{
            param($s, $e)
            if ($e.Data) { Add-Content -Path $errPath -Value $e.Data -Encoding UTF8 }
        }
        $proc.add_OutputDataReceived($oHandler)
        $proc.add_ErrorDataReceived($eHandler)

        $launched = $proc.Start()
        if (-not $launched) {
            $State.BackendStatus = "launch-failed"
            Write-Log $ERR "Failed to launch uvicorn. Retrying in ${BASE_DELAY}s..."
            Start-Sleep -Seconds $BASE_DELAY
            continue
        }

        $proc.BeginOutputReadLine()
        $proc.BeginErrorReadLine()

        $State.BackendPID    = $proc.Id
        $State.BackendStatus = "running"
        Write-Log $OUT "Backend started (PID $($proc.Id))"

        $proc.WaitForExit()

        $State.BackendPID    = -1
        $State.BackendUp     = $false
        $State.BackendStatus = "restarting"
        $State.BackendRestarts++

        $msg = "Backend exited (code=$($proc.ExitCode), restart #$($State.BackendRestarts))"
        $State.LastBackendErr = $msg
        Write-Log $ERR $msg

        $delay = [Math]::Min(30, $BASE_DELAY * [Math]::Pow(1.5, [Math]::Min($State.BackendRestarts - 1, 6)))
        Start-Sleep -Seconds $delay
    }
} -ArgumentList $ROOT, $PYTHON, $LOGS, $State


# ---- Frontend watchdog job --------------------------------------------------
$FrontendJob = Start-Job -Name "FrontendWatchdog" -ScriptBlock {
    param($ROOT, $LOGS, $State)

    function Write-Log {
        param([string]$File, [string]$Msg)
        $ts = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
        Add-Content -Path (Join-Path $LOGS $File) -Value "[$ts] $Msg" -Encoding UTF8
    }

    function Rotate-Log {
        param([string]$File)
        $path = Join-Path $LOGS $File
        if ((Test-Path $path) -and (Get-Item $path).Length -gt 10MB) {
            $archive = $path -replace '\.log$', ("_" + (Get-Date -Format "yyyyMMdd_HHmmss") + ".log")
            Move-Item $path $archive
        }
    }

    $FRONT      = Join-Path $ROOT "frontend"
    $OUT        = "frontend_out.log"
    $ERR        = "frontend_err.log"
    $BASE_DELAY = 3

    $nodeCmd = Get-Command node -ErrorAction SilentlyContinue
    $node = if ($nodeCmd) { $nodeCmd.Source } else { "node" }

    $State.FrontendStatus = "starting"
    Write-Log $OUT "Frontend watchdog started."

    while ($true) {
        Rotate-Log $OUT
        Rotate-Log $ERR

        $psi = New-Object System.Diagnostics.ProcessStartInfo
        $psi.FileName               = $node
        $psi.Arguments              = "node_modules\.bin\next dev"
        $psi.WorkingDirectory       = $FRONT
        $psi.UseShellExecute        = $false
        $psi.RedirectStandardOutput = $true
        $psi.RedirectStandardError  = $true
        $psi.CreateNoWindow         = $true
        $psi.EnvironmentVariables["NODE_ENV"]                = "development"
        $psi.EnvironmentVariables["NEXT_TELEMETRY_DISABLED"] = "1"

        $proc = New-Object System.Diagnostics.Process
        $proc.StartInfo = $psi

        $outPath = Join-Path $LOGS $OUT
        $errPath = Join-Path $LOGS $ERR

        $oHandler = [System.Diagnostics.DataReceivedEventHandler]{
            param($s, $e)
            if ($e.Data) { Add-Content -Path $outPath -Value $e.Data -Encoding UTF8 }
        }
        $eHandler = [System.Diagnostics.DataReceivedEventHandler]{
            param($s, $e)
            if ($e.Data) { Add-Content -Path $errPath -Value $e.Data -Encoding UTF8 }
        }
        $proc.add_OutputDataReceived($oHandler)
        $proc.add_ErrorDataReceived($eHandler)

        $launched = $proc.Start()
        if (-not $launched) {
            $State.FrontendStatus = "launch-failed"
            Write-Log $ERR "Failed to launch Next.js. Retrying in ${BASE_DELAY}s..."
            Start-Sleep -Seconds $BASE_DELAY
            continue
        }

        $proc.BeginOutputReadLine()
        $proc.BeginErrorReadLine()

        $State.FrontendPID    = $proc.Id
        $State.FrontendStatus = "running"
        Write-Log $OUT "Frontend started (PID $($proc.Id))"

        $proc.WaitForExit()

        $State.FrontendPID    = -1
        $State.FrontendUp     = $false
        $State.FrontendStatus = "restarting"
        $State.FrontendRestarts++

        $msg = "Frontend exited (code=$($proc.ExitCode), restart #$($State.FrontendRestarts))"
        $State.LastFrontendErr = $msg
        Write-Log $ERR $msg

        $delay = [Math]::Min(30, $BASE_DELAY * [Math]::Pow(1.5, [Math]::Min($State.FrontendRestarts - 1, 6)))
        Start-Sleep -Seconds $delay
    }
} -ArgumentList $ROOT, $LOGS, $State


# ---- Foreground status monitor ----------------------------------------------
Write-Host ""
Write-Host "  Stock Prediction Platform  -  Watchdog Active" -ForegroundColor Cyan
Write-Host ""
Write-Host "  Backend  : http://localhost:8000   (API docs: /docs)"
Write-Host "  Frontend : http://localhost:3000"
Write-Host "  Logs     : $LOGS"
Write-Host "  Press Ctrl+C to stop all servers."
Write-Host ""

Write-Host "  Waiting for backend  (port 8000, up to 45s)..." -ForegroundColor DarkGray
if (Wait-Port 8000 45) {
    Write-Host "  Backend  port 8000 open." -ForegroundColor Green
} else {
    Write-Host "  Backend  NOT open yet  -  check logs\backend_err.log" -ForegroundColor Red
}

Write-Host "  Waiting for frontend (port 3000, up to 60s)..." -ForegroundColor DarkGray
if (Wait-Port 3000 60) {
    Write-Host "  Frontend port 3000 open." -ForegroundColor Green
} else {
    Write-Host "  Frontend NOT open yet  -  check logs\frontend_err.log" -ForegroundColor Red
}

Write-Host ""
Write-Host ("  {0,-10} {1,-12} {2,-10} {3,8}    {4,-10} {5,-12} {6,-10} {7,8}    {8}" -f `
    "SERVICE","STATUS","PID","RESTARTS","SERVICE","STATUS","PID","RESTARTS","UPTIME") `
    -ForegroundColor DarkGray
Write-Host ("  " + ("-" * 108)) -ForegroundColor DarkGray

$tick = 0

try {
    while ($true) {
        Start-Sleep -Seconds 8

        # Drain job output buffers
        Receive-Job $BackendJob  -ErrorAction SilentlyContinue | Out-Null
        Receive-Job $FrontendJob -ErrorAction SilentlyContinue | Out-Null

        # Update health
        $bUp = Http-OK "http://localhost:8000/health"
        $fUp = Port-Alive 3000
        $State.BackendUp  = $bUp
        $State.FrontendUp = $fUp

        if ($bUp)  { $State.BackendStatus  = "healthy"   }
        elseif ($State.BackendPID  -gt 0) { $State.BackendStatus  = "unhealthy" }

        if ($fUp)  { $State.FrontendStatus = "healthy"   }
        elseif ($State.FrontendPID -gt 0) { $State.FrontendStatus = "unhealthy" }

        $uptime = (Get-Date) - $State.Started

        $bPidStr = if ($State.BackendPID  -gt 0) { "PID $($State.BackendPID)"  } else { "no-proc" }
        $fPidStr = if ($State.FrontendPID -gt 0) { "PID $($State.FrontendPID)" } else { "no-proc" }

        $line = "  {0,-10} {1,-12} {2,-10} {3,8}    {4,-10} {5,-12} {6,-10} {7,8}    {8:hh\:mm\:ss}" -f `
            "BACKEND",  $State.BackendStatus.ToUpper(),  $bPidStr, $State.BackendRestarts, `
            "FRONTEND", $State.FrontendStatus.ToUpper(), $fPidStr, $State.FrontendRestarts, `
            $uptime

        $color = if ($bUp -and $fUp) { "Green" } elseif ($bUp -or $fUp) { "Yellow" } else { "Red" }
        Write-Host $line -ForegroundColor $color

        foreach ($pair in @( @($BackendJob, "Backend"), @($FrontendJob, "Frontend") )) {
            $j = $pair[0]; $n = $pair[1]
            if ($j.State -in @("Failed","Completed")) {
                Write-Host "  WARNING: $n watchdog job has stopped! Check logs." -ForegroundColor Red
            }
        }

        $tick++
        if ($tick % 40 -eq 0) {
            Write-Host ("  " + ("-" * 108)) -ForegroundColor DarkGray
        }
    }
} finally {
    Write-Host ""
    Write-Host "  Shutting down..." -ForegroundColor Yellow

    foreach ($j in @($BackendJob, $FrontendJob)) {
        if ($j) {
            Stop-Job  $j -ErrorAction SilentlyContinue
            Remove-Job $j -Force -ErrorAction SilentlyContinue
        }
    }

    foreach ($port in 8000, 3000) {
        $pids = (netstat -ano 2>$null) | Select-String (":$port\s") |
                ForEach-Object { ($_.ToString() -split '\s+')[-1] } |
                Select-Object -Unique
        foreach ($p in $pids) {
            if ($p -match '^\d+$') {
                Stop-Process -Id ([int]$p) -Force -ErrorAction SilentlyContinue
            }
        }
    }

    Write-Host "  All servers stopped." -ForegroundColor Yellow
}
