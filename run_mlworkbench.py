"""ML Workbench launcher.

Double-click behaviour  -> opens the desktop app.
Command line           -> runs the CLI (console attached when possible).

Build a Windows .exe with:
    pyinstaller --noconfirm --clean --onefile --windowed --name MLWorkbench \
                --icon assets/icon.ico --collect-all rdkit run_mlworkbench.py
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

APP = "ML Workbench"


def _attach_console() -> None:
    """When the GUI build is started from cmd/PowerShell, reuse the parent console
    so CLI output is visible instead of vanishing."""
    if sys.platform.startswith("win"):
        try:
            import ctypes

            ctypes.windll.kernel32.AttachConsole(-1)
            sys.stdout = open("CONOUT$", "w", encoding="utf-8", errors="replace")
            sys.stderr = open("CONOUT$", "w", encoding="utf-8", errors="replace")
        except Exception:
            pass


def _fatal(exc: BaseException) -> None:
    """Never die silently: a windowed exe has no console to print to."""
    text = "".join(traceback.format_exception(exc))
    log = Path.home() / "MLWorkbench_crash.log"
    try:
        log.write_text(text, encoding="utf-8")
    except Exception:
        log = None
    try:
        import tkinter as tk
        from tkinter import messagebox

        root = tk.Tk()
        root.withdraw()
        messagebox.showerror(
            f"{APP} — unexpected error",
            f"{type(exc).__name__}: {exc}\n\n"
            + (f"Full details written to:\n{log}" if log else "Could not write a log file."),
        )
        root.destroy()
    except Exception:
        print(text, file=sys.stderr)


def _selftest(outdir: str) -> int:
    r"""Headless proof that the packaged build works end to end.

    Used by the build pipeline (and by you, if an exe ever misbehaves):
        MLWorkbench.exe --selftest C:\temp\mlw_selftest
    Writes selftest.json with the numbers it got.
    """
    import json

    import numpy as np
    import pandas as pd

    from mlw import core

    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    logs: list[str] = []

    rng = np.random.default_rng(3)
    n = 140
    x1 = rng.uniform(60, 110, n)
    x2 = rng.uniform(1, 5, n)
    cat = rng.choice(["A", "B", "C"], n)
    y = 40 + 0.3 * (x1 - 85) - 0.55 * (x1 - 85) ** 2 / 25 + 2.4 * x2 + np.where(cat == "A", 1.5, 0.0) + rng.normal(0, 2, n)
    df = pd.DataFrame({"temperature": x1.round(2), "pressure": x2.round(2), "reactor": cat, "yield": y.round(2)})

    report: dict = {"app": "ML Workbench", "version": core.VERSION, "environment": core.environment()}
    try:
        cfg = core.TrainConfig(target="yield", features=["temperature", "pressure", "reactor"],
                               task="regression", folds=3, seed=7, search_iters=2,
                               models=["Linear", "RandomForest"], holdout_frac=0.2)
        res = core.train_and_evaluate(df, cfg, out_root=out, name="selftest", log=logs.append)
        report["train"] = {"chosen": res["chosen"], "rows": len(df), "run_dir": res["run_dir"],
                           "primary": res["primary_metric"]}

        bundle = core.load_model(res["model_path"])
        pred, val = core.predict_frame(bundle, df.drop(columns=["yield"]).head(5))
        report["predict"] = {"rows": len(pred), "column": str(pred.columns[-1]),
                             "values": [round(float(v), 3) for v in pred.iloc[:, -1]],
                             "warnings": val.get("warnings", [])}

        grid = core.optimise_grid(bundle, {"temperature": (70, 105, 4), "pressure": (1, 5, 4)},
                                  {"reactor": ["A", "B"]})
        report["optimise"] = {"candidates": int(len(grid)), "best": round(float(grid.iloc[0, -3]), 3)}

        spec = core.smiles_features(["CCO", "c1ccccc1", "[Na+].[Cl-]"], mode="descriptors")[0]
        report["molecules"] = {"rows": int(len(spec)), "columns": int(spec.shape[1])}

        report["ok"] = True
    except Exception as exc:  # noqa: BLE001
        report["ok"] = False
        report["error"] = f"{type(exc).__name__}: {exc}"
        report["traceback"] = traceback.format_exc()

    report["log"] = logs[-25:]
    (out / "selftest.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0 if report.get("ok") else 1


def _gui(csv_path: str | None = None) -> None:
    from mlw.gui import main as gui_main

    if csv_path:
        sys.argv = [sys.argv[0], csv_path]  # "Open with -> ML Workbench" support
    gui_main()


def main() -> int:
    # Required for any multiprocessing on Windows; harmless elsewhere.
    try:
        import multiprocessing

        multiprocessing.freeze_support()
    except Exception:
        pass

    args = sys.argv[1:]

    if args and args[0] == "--selftest":
        return _selftest(args[1] if len(args) > 1 else "mlw_selftest")

    # A single .csv argument means "open it in the GUI" (double-click / Open with).
    if len(args) == 1 and args[0].lower().endswith(".csv"):
        try:
            _gui(args[0])
            return 0
        except BaseException as exc:  # noqa: BLE001
            _fatal(exc)
            return 1

    # Any other arguments mean "use the command line interface".
    if args and args[0] not in ("--gui", "gui"):
        _attach_console()
        try:
            from mlw.cli import main as cli_main

            return cli_main(args)
        except BaseException as exc:  # noqa: BLE001
            _fatal(exc)
            return 1

    try:
        from mlw.gui import main as gui_main

        gui_main()
        return 0
    except BaseException as exc:  # noqa: BLE001
        _fatal(exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
