@echo off
cd /d "%~dp0PolySignAI"
title PolySign AI
call venv\Scripts\activate.bat
python desktop_app.py
