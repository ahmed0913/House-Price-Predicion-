#!/bin/bash
# ===========================================
# House Price Prediction — Run Script (Linux)
# ===========================================

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "🏠 House Price Prediction — California Housing"
echo "================================================"

# Check if venv exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
    echo "📥 Installing dependencies..."
    ./venv/bin/pip install -r requirements.txt
    echo "✅ Dependencies installed!"
fi

# Check if model files exist
if [ ! -f "model.pkl" ]; then
    echo "⚠️  Model files not found. Please run the notebook first:"
    echo "    jupyter notebook notebook.ipynb"
    echo "    (Run all cells to generate model.pkl, scaler.pkl, metrics.pkl)"
    exit 1
fi

# Kill any existing Streamlit process on port 8501
if fuser 8501/tcp >/dev/null 2>&1; then
    echo "⚠️  Port 8501 is busy. Stopping the old server..."
    fuser -k 8501/tcp >/dev/null 2>&1
    sleep 2
    echo "✅ Old server stopped."
fi

# Activate venv and run streamlit
echo ""
echo "🚀 Starting Streamlit app..."
echo "   Open http://localhost:8501 in your browser"
echo "   Press Ctrl+C to stop"
echo ""
./venv/bin/streamlit run app.py --server.headless true --server.port 8501
