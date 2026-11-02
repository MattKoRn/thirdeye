# From the source checkout: .\scripts\install.ps1 -Game hack -Source C:\Games\HACK
param(
  [ValidateSet("eob3","hack","skip")][string]$Game,
  [string]$Source,
  [switch]$Archive,
  [switch]$NoDeps,
  [switch]$CleanBuild,
  [switch]$RefreshDeps
)
$ErrorActionPreference = "Stop"
$python = Get-Command py -ErrorAction SilentlyContinue
if ($python) {
    $exe = $python.Source
    $arguments = @("-3", "$PSScriptRoot\install.py")
} else {
    $python = Get-Command python -ErrorAction SilentlyContinue
    if (-not $python) {
        Write-Host "Installing Python 3 with winget..."
        winget install --id Python.Python.3.13 -e --accept-package-agreements --accept-source-agreements
        if ($LASTEXITCODE -ne 0) { throw "Python installation failed. Install Python 3 manually." }
        $python = Get-Command python -ErrorAction SilentlyContinue
        if (-not $python) {
            $candidate = Join-Path $env:LOCALAPPDATA "Programs\Python\Python313\python.exe"
            if (Test-Path $candidate) { $python = @{ Source = $candidate } }
        }
        if (-not $python) {
            throw "Python installed but not yet on PATH. Reopen PowerShell and run the installer again."
        }
    }
    $exe = $python.Source
    $arguments = @("$PSScriptRoot\install.py")
}
if ($Game) { $arguments += @("--game", $Game) }
if ($Source) { $arguments += @("--source", $Source) }
if ($Archive) { $arguments += "--archive" }
if ($NoDeps) { $arguments += "--no-deps" }
if ($CleanBuild) { $arguments += "--clean-build" }
if ($RefreshDeps) { $arguments += "--refresh-deps" }
& $exe @arguments
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
