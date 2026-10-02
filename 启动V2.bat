@echo off
chcp 65001 >nul
cd /d "%~dp0V2"
python app.py %*
if errorlevel 1 pause
