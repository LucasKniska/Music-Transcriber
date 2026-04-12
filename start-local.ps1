#Requires -Version 5.1
<#
.SYNOPSIS
    Start Music Transcriber locally on Windows (no Supabase required).
.DESCRIPTION
    Sets up the Python venv, installs dependencies, then launches all three
    servers (WebSocket :8000, REST API :5000, Vite :5173) in separate windows.
    Press Ctrl+C in any window, or close this one, to stop everything.
#>

$ErrorActionPreference = "Stop"
$ProjectDir  = $PSScriptRoot
$BackendDir  = Join-Path $ProjectDir "backend"
$VenvDir     = Join-Path $BackendDir ".venv"
$PythonExe   = Join-Path $VenvDir "Scripts\python.exe"
$PipExe      = Join-Path $VenvDir "Scripts\pip.exe"

Write-Host "=== Music Transcriber - Local Test Mode (Windows) ===" -ForegroundColor Cyan
Write-Host "Project: $ProjectDir"

# --- Pre-flight: kill anything already on our ports ---
foreach ($port in @(8000, 5000, 5173)) {
    $pids = (netstat -ano | Select-String ":$port\s" | ForEach-Object {
        ($_ -split '\s+')[-1]
    } | Sort-Object -Unique)
    foreach ($p in $pids) {
        if ($p -match '^\d+$' -and $p -ne '0') {
            try {
                Stop-Process -Id ([int]$p) -Force -ErrorAction Stop
                Write-Host "[ports] Killed PID $p on port $port" -ForegroundColor Yellow
            } catch { }
        }
    }
}

# --- 0. Require Python 3.11 or 3.10 ---
$Python3x     = $null
$Python3xArgs = @()

$candidates = @(
    @{ Exe = "py";         Args = @("-3.11") },
    @{ Exe = "python3.11"; Args = @() },
    @{ Exe = "py";         Args = @("-3.10") },
    @{ Exe = "python3.10"; Args = @() },
    @{ Exe = "python3";    Args = @() },
    @{ Exe = "python";     Args = @() }
)

foreach ($c in $candidates) {
    try {
        $ver = (& $c.Exe @($c.Args) --version 2>&1).ToString()
        if ($ver -match "Python 3\.(11|10)\.") {
            $Python3x     = $c.Exe
            $Python3xArgs = $c.Args
            break
        }
    } catch { }
}

if (-not $Python3x) {
    Write-Host "[error] Python 3.11 or 3.10 not found on PATH." -ForegroundColor Red
    Write-Host "        Download from https://www.python.org/downloads/" -ForegroundColor Red
    Write-Host "        Make sure to check 'Add Python to PATH' during install." -ForegroundColor Red
    exit 1
}

$verString = (& $Python3x @Python3xArgs --version 2>&1).ToString()
Write-Host "[setup] Using $verString" -ForegroundColor Green

# --- 1. Python venv ---
# Recreate the venv if it was built with a different Python version
$needNewVenv = $true
if (Test-Path $PythonExe) {
    $venvVer = (& $PythonExe --version 2>&1).ToString()
    if ($venvVer -match "Python 3\.(11|10)\.") { $needNewVenv = $false }
}

if ($needNewVenv) {
    if (Test-Path $VenvDir) {
        Write-Host "[setup] Existing venv is not Python 3.10/3.11 - recreating..." -ForegroundColor Yellow
        Remove-Item $VenvDir -Recurse -Force
    } else {
        Write-Host "[setup] Creating Python venv..." -ForegroundColor Yellow
    }
    & $Python3x @Python3xArgs -m venv $VenvDir
} else {
    Write-Host "[setup] Venv OK ($venvVer)" -ForegroundColor Green
}

# --- 2. Python deps ---
Write-Host "[setup] Installing Python dependencies..." -ForegroundColor Yellow
& $PipExe install --quiet fastapi uvicorn websockets numpy basic-pitch

# --- 3. Frontend deps ---
Write-Host "[setup] Syncing npm packages..." -ForegroundColor Yellow
Push-Location $ProjectDir
npm install
npm prune
Pop-Location

# --- 4. Launch servers in separate windows ---
Write-Host ""
Write-Host "[start] Launching servers..." -ForegroundColor Cyan

# WebSocket server - port 8000
$wsJob = Start-Process powershell -ArgumentList @(
    "-NoExit", "-Command",
    "Write-Host 'WebSocket server (port 8000)' -ForegroundColor Cyan; " +
    "Set-Location '$BackendDir'; " +
    "& '$PythonExe' audio.py"
) -PassThru

# REST API - port 5000
$apiJob = Start-Process powershell -ArgumentList @(
    "-NoExit", "-Command",
    "Write-Host 'REST API (port 5000)' -ForegroundColor Cyan; " +
    "Set-Location '$BackendDir'; " +
    "& '$PythonExe' -m uvicorn api:app --host 0.0.0.0 --port 5000"
) -PassThru

# Vite frontend - port 5173
$viteJob = Start-Process powershell -ArgumentList @(
    "-NoExit", "-Command",
    "Write-Host 'Vite dev server (port 5173)' -ForegroundColor Cyan; " +
    "Set-Location '$ProjectDir'; " +
    "npm run dev"
) -PassThru

Write-Host ""
Write-Host "=== All services started ===" -ForegroundColor Green
Write-Host "  Frontend:  http://localhost:5173/record" -ForegroundColor White
Write-Host "  REST API:  http://localhost:5000"        -ForegroundColor White
Write-Host "  WebSocket: ws://localhost:8000"          -ForegroundColor White
Write-Host ""
Write-Host "Note: audio.py takes ~15s to load the AI model before recording works." -ForegroundColor Yellow
Write-Host "Close this window (or press Ctrl+C) to stop all servers." -ForegroundColor Yellow
Write-Host ""

# Wait here; kill child windows on exit
try {
    while ($true) { Start-Sleep -Seconds 5 }
} finally {
    Write-Host "[stop] Shutting down..." -ForegroundColor Red
    foreach ($p in @($wsJob, $apiJob, $viteJob)) {
        if ($p -and -not $p.HasExited) {
            Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue
        }
    }
    Write-Host "[stop] Done." -ForegroundColor Red
}
