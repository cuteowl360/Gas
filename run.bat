@echo off
title GasWatch
cd /d "%~dp0"
echo Starting GasWatch...
python -m streamlit run app.py --server.port 8506 --server.headless false
pause
