@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo Project Python environment was not found.
    echo Open this folder in VS Code and follow the setup steps in README.md.
    pause
    exit /b 1
)

if not exist "mnist_digit_model.keras" (
    echo Trained model was not found.
    echo In VS Code, run: .venv\Scripts\python.exe train_model.py
    pause
    exit /b 1
)

echo Starting Handwritten Number Recognition...
".venv\Scripts\python.exe" -m streamlit run app.py

#. Dobara chalana ho to project folder se run_project.bat kholo; phir browser mein http://localhost:8501 par app khul jayega.