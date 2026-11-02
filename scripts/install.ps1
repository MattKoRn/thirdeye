# From the source checkout: .\scripts\install.ps1 -Game hack -Source C:\Games\HACK
param(
  [ValidateSet("eob3","hack","skip")][string]$Game,
  [string]$Source,
  [switch]$Archive,
  [switch]$NoDeps
)
$ErrorActionPreference = "Stop"
$python = Get-Command py -ErrorAction SilentlyContinue
if ($python) {
    $exe = $python.Source
    $arguments = @("-3", "$PSScriptRoot\install.py")
} else {
    $python = Get-Command python -ErrorAction SilentlyContinue
    if (-not $python) {
        throw "Install Python 3 with winget install Python.Python.3.13, then rerun."
    }
    $exe = $python.Source
    $arguments = @("$PSScriptRoot\install.py")
}
if ($Game) { $arguments += @("--game", $Game) }
if ($Source) { $arguments += @("--source", $Source) }
if ($Archive) { $arguments += "--archive" }
if ($NoDeps) { $arguments += "--no-deps" }
& $exe @arguments
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
