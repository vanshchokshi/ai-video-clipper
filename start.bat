@echo off
title AI Video Clipper Studio
cd /d "%~dp0"
echo Starting AI Video Clipper Studio...
if exist .venv\Scripts\python.exe (
    .venv\Scripts\python.exe run.py
) else (
    python run.py
)
pause

