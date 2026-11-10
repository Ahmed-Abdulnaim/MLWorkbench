#!/bin/bash
set -x
export WINEPREFIX=/home/user/.winefresh
export WINEARCH=win64
export WINEDLLOVERRIDES="mscoree,mshtml="
export WINEDEBUG=-all
LOG=/home/user/build_logs
cd /home/user/build_win || exit 1
rm -rf $WINEPREFIX

xvfb-run -a wineboot -u > $LOG/wineboot2.log 2>&1
sleep 8
xvfb-run -a wine cmd /c echo hello > $LOG/winecmd.log 2>&1
cat $LOG/winecmd.log

xvfb-run -a wine python-3.11.9-amd64.exe /quiet InstallAllUsers=1 TargetDir=C:\\Py311 \
  Include_launcher=0 Include_test=0 Include_doc=0 Include_pip=1 Include_tcltk=1 SimpleInstall=1 \
  > $LOG/pyinstall2.log 2>&1
echo "installer exit: $?"
sleep 20
ls $WINEPREFIX/drive_c/ 2>/dev/null
ls $WINEPREFIX/drive_c/Py311 2>/dev/null | head -8
xvfb-run -a wine C:\\Py311\\python.exe -V 2>&1 | tail -2
xvfb-run -a wine C:\\Py311\\python.exe -c "import tkinter; print('tk', tkinter.TkVersion)" 2>&1 | tail -2
echo "=== STEP1 DONE ==="
