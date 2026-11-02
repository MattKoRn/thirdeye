@echo off
setlocal
rem Launch the PowerShell installer with a process-local execution policy.
rem This avoids changing the user's system-wide PowerShell settings.
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0install.ps1" %*
exit /b %ERRORLEVEL%
