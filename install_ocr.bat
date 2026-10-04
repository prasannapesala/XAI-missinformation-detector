@echo off
echo Installing EasyOCR into project venv (required for image text extraction)...
cd /d "%~dp0"
venv\Scripts\pip install easyocr
echo.
echo Done. Restart Streamlit: venv\Scripts\streamlit run src\dashboard\app.py
pause
