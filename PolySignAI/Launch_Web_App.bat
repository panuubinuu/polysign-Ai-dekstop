@echo off
cd /d "%~dp0"
title PolySign AI - Web Application
call venv\Scripts\activate.bat
streamlit run app.py
