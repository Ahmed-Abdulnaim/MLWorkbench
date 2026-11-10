#!/bin/bash
set -x
export WINEPREFIX=/home/user/.winefresh
export WINEARCH=win64
export WINEDEBUG=-all
export WINEDLLOVERRIDES="mscoree,mshtml="
LOG=/home/user/build_logs
cd /home/user/build_win || exit 1

run() { xvfb-run -a wine C:\\Py311\\python.exe "$@"; }

run -m pip install --upgrade --no-warn-script-location pip setuptools wheel > $LOG/pip0.log 2>&1
tail -2 $LOG/pip0.log

run -m pip install --no-warn-script-location numpy pandas scikit-learn matplotlib joblib pyinstaller > $LOG/pip1.log 2>&1
echo "core pkgs exit: $?"
tail -5 $LOG/pip1.log

run -m pip install --no-warn-script-location rdkit > $LOG/pip2.log 2>&1
echo "rdkit exit: $?"
tail -3 $LOG/pip2.log

run -c "import numpy,pandas,sklearn,matplotlib,joblib,PyInstaller; print('numpy',numpy.__version__,'pandas',pandas.__version__,'sklearn',sklearn.__version__,'mpl',matplotlib.__version__,'pyi',PyInstaller.__version__)" 2>&1 | tail -3
run -c "import rdkit; print('rdkit', rdkit.__version__)" 2>&1 | tail -2
echo "=== PKGS DONE ==="
