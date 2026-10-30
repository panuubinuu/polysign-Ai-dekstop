@echo off
cd /d "%~dp0PolySignAI"
title PolySign AI - Web App
call venv\Scripts\activate.bat
streamlit run app.py
