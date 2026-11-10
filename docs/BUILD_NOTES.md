# How the shipped .exe was produced

This is the honest record of the build, because it is unusual: the binary in `dist/` was
compiled **on Linux** by running a real Windows Python 3.11 under **Wine**, then packaged
with PyInstaller. No Windows machine was involved.

```
Linux → Wine 10.0 (wine64 + wine32) → Python 3.11.9 (Windows build, with tkinter)
      → numpy 1.26.4 · pandas 2.2.3 · scikit-learn 1.5.2 · matplotlib 3.9.2 · rdkit 2026.9.1
      → PyInstaller 6.22 (--windowed --onefile / --onedir)
      → MLWorkbench.exe  (PE32+ x86-64)
```

## Notes and gotchas worth knowing

1. **numpy must be < 2.** numpy 2.x calls `ucrtbase.dll.crealf`, which Wine does not
   implement, and the interpreter aborts with `unimplemented function`. The whole stack is
   therefore pinned to the numpy-1.x generation (see the versions above). This is a Wine
   limitation, not an app limitation — on real Windows numpy 2.x works fine.
2. **Use the official python.org installer, not the embeddable zip.** The embeddable
   distribution has no `tkinter` and no `pip`, and this app's GUI needs tkinter. The
   installer also needs 32-bit support (`wine32:i386`) because its bootstrap is 32-bit.
3. **A fresh `WINEPREFIX` is required** after adding `wine32`; a prefix created before that
   is broken (`could not load kernel32.dll`).
4. `--collect-all rdkit` is mandatory: RDKit ships compiled extension modules and data files
   that PyInstaller's hooks do not discover on their own.
5. The build was verified by running `MLWorkbench.exe --selftest <dir>` **inside Wine**,
   which trains a model, predicts, optimises a grid and generates descriptors, writing
   `selftest.json`. If that file says `"ok": true`, the frozen bundle's whole engine works.

## Reproducing it on Windows (the normal way)

```bat
build_windows.bat
```

or push to GitHub and let `.github/workflows/build-exe.yml` build it on a Windows runner.
Both produce a functionally identical exe; only the library versions will differ slightly
(any numpy 1.x or 2.x is fine on real Windows).
