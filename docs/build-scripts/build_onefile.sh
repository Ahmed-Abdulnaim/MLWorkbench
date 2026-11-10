#!/bin/bash
export WINEPREFIX=/home/user/.winefresh WINEDEBUG=-all WINEDLLOVERRIDES="mscoree,mshtml="
LOG=/home/user/build_logs
cd $WINEPREFIX/drive_c/build/mlworkbench || exit 1
xvfb-run -a wine C:\\Py311\\python.exe -m PyInstaller --noconfirm --clean --windowed --onefile --name MLWorkbench \
 --icon C:\\build\\mlworkbench\\assets\\icon.ico --collect-all rdkit \
 --exclude-module PyQt5 --exclude-module PyQt6 --exclude-module PySide2 --exclude-module PySide6 \
 --exclude-module IPython --exclude-module notebook --exclude-module pytest --exclude-module sphinx \
 --distpath C:\\build\\mlworkbench\\dist_onefile --workpath C:\\build\\mlworkbench\\build_onefile \
 run_mlworkbench.py > $LOG/onefile2.log 2>&1
echo "exit: $?"
ls -la dist_onefile/
echo "=== ONEFILE DONE ==="
