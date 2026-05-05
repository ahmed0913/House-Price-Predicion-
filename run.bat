@echo off
REM ===========================================
REM House Price Prediction — Run Script (Windows)
REM ===========================================

cd /d "%~dp0"

echo 🏠 House Price Prediction — California Housing
echo ================================================

REM Check if venv exists
if not exist "venv" (
    echo 📦 Creating virtual environment...
    python -m venv venv
    echo 📥 Installing dependencies...
    venv\Scripts\pip install -r requirements.txt
    echo ✅ Dependencies installed!
)

REM Check if model files exist
if not exist "model.pkl" (
    echo ⚠️  Model files not found. Please run the notebook first:
    echo     jupyter notebook notebook.ipynb
    echo     Run all cells to generate model.pkl, scaler.pkl, metrics.pkl
    pause
    exit /b 1
)

REM Kill any existing Streamlit process on port 8501
for /f "tokens=5" %%a in ('netstat -aon ^| find ":8501" ^| find "LISTENING"') do (
    echo ⚠️  Port 8501 is busy. Stopping the old server...
    taskkill /F /PID %%a >nul 2>&1
    echo ✅ Old server stopped.
)

echo.
echo 🚀 Starting Streamlit app...
echo    Open http://localhost:8501 in your browser
echo    Press Ctrl+C to stop
echo.
venv\Scripts\streamlit run app.py --server.headless true --server.port 8501
pause
