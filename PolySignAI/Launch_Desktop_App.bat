@echo off
cd /d "%~dp0"
title PolySign AI - Desktop Application
call venv\Scripts\activate.bat
python desktop_app.py
