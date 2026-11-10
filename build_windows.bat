@echo off
REM ===========================================================================
REM  Build MLWorkbench.exe on Windows (run this ON WINDOWS, not in Wine/Linux).
REM  Requires Python 3.11 or 3.12 x64 from python.org (with "Add to PATH").
REM ===========================================================================
setlocal
echo [1/4] creating a clean virtual environment...
if not exist .venv py -3 -m venv .venv
call .venv\Scripts\activate

echo [2/4] installing dependencies (this downloads ~250 MB the first time)...
python -m pip install --upgrade pip >nul
python -m pip install numpy pandas scikit-learn matplotlib joblib pyinstaller

echo [3/4] optional: molecule support (RDKit). Comment out to build a smaller exe.
python -m pip install rdkit

echo [4/4] packaging...
pyinstaller --noconfirm --clean --onefile --windowed ^
  --name MLWorkbench ^
  --icon assets\icon.ico ^
  --collect-all rdkit ^
  --exclude-module PyQt5 --exclude-module PyQt6 --exclude-module PySide2 --exclude-module PySide6 ^
  --exclude-module IPython --exclude-module notebook --exclude-module pytest ^
  --exclude-module scipy.io.wavfile --exclude-module tkinter.test ^
  run_mlworkbench.py

echo.
echo Done. Your app is: dist\MLWorkbench.exe
pause
