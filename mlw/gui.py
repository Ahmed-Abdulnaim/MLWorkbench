"""mlw.gui — the desktop workbench (tkinter, no extra UI dependency).

Seven tabs, one idea: everything you decide is written down, every number has a
baseline next to it, and the model you save contains its own preprocessing.
"""

from __future__ import annotations

import os
import queue
import subprocess
import sys
import threading
import traceback
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np
import pandas as pd

from . import core

ACCENT = "#0f8a7a"
ACCENT_DARK = "#0a5f54"
BG = "#f4f7f8"
CARD = "#ffffff"
INK = "#1d2429"
MUTED = "#5b6b75"
RED = "#b3261e"
AMBER = "#8a5200"
FONT = ("Segoe UI", 9)
FONT_B = ("Segoe UI", 9, "bold")
FONT_H1 = ("Segoe UI", 15, "bold")
FONT_H2 = ("Segoe UI", 10, "bold")
FONT_MONO = ("Consolas", 9)

DISPLAY_METRIC = {
    "model": "model",
    "complexity": "complexity",
    "r2": "R² (CV)",
    "r2_std": "R² spread",
    "mean_absolute_error": "MAE",
    "mean_absolute_error_std": "MAE spread",
    "root_mean_squared_error": "RMSE",
    "root_mean_squared_error_std": "RMSE spread",
    "accuracy": "accuracy",
    "balanced_accuracy": "balanced acc",
    "f1_macro": "F1 macro",
    "f1_weighted": "F1 weighted",
    "holdout_r2": "test R²",
    "holdout_mae": "test MAE",
    "holdout_rmse": "test RMSE",
    "holdout_mape": "test MAPE %",
    "holdout_accuracy": "test acc",
    "holdout_balanced_accuracy": "test bal. acc",
    "holdout_f1_macro": "test F1",
    "holdout_f1_weighted": "test F1 wtd",
    "holdout_roc_auc": "test AUC",
    "holdout_roc_auc_ovr": "test AUC ovr",
    "holdout_precision_macro": "test precision",
    "holdout_recall_macro": "test recall",
}


# --------------------------------------------------------------------------- #
#  helpers
# --------------------------------------------------------------------------- #
class ScrollFrame(ttk.Frame):
    """A vertically scrollable frame (used for the model list and range editor)."""

    def __init__(self, master, **kw):
        super().__init__(master, **kw)
        self.canvas = tk.Canvas(self, borderwidth=0, background=CARD, highlightthickness=0)
        self.vsb = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.inner = ttk.Frame(self.canvas, style="Card.TFrame")
        self.inner.bind("<Configure>",
                        lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self._win = self.canvas.create_window((0, 0), window=self.inner, anchor="nw")
        self.canvas.bind("<Configure>", lambda e: self.canvas.itemconfigure(self._win, width=e.width))
        self.canvas.configure(yscrollcommand=self.vsb.set)
        self.canvas.pack(side="left", fill="both", expand=True)
        self.vsb.pack(side="right", fill="y")
        self.canvas.bind_all("<MouseWheel>", self._wheel, add="+")

    def _wheel(self, event):
        try:
            self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        except Exception:
            pass

    def clear(self):
        for child in self.inner.winfo_children():
            child.destroy()


def open_in_explorer(path: str | Path) -> None:
    path = str(path)
    try:
        if sys.platform.startswith("win"):
            os.startfile(path)  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            subprocess.Popen(["open", path])
        else:
            subprocess.Popen(["xdg-open", path])
    except Exception:
        pass


def readable(path: str) -> str:
    return Path(path).name if path else ""


# --------------------------------------------------------------------------- #
#  application
# --------------------------------------------------------------------------- #
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(f"ML Workbench {core.VERSION}")
        self.geometry("1220x820")
        self.minsize(1080, 700)
        self.configure(bg=BG)

        # ---- state ------------------------------------------------------
        self.df: pd.DataFrame | None = None
        self.data_path: str = ""
        self.meta: dict = {}
        self.issues: list[core.Issue] = []
        self.audit_target: str = ""
        self.result: dict | None = None
        self.model_path: str = ""
        self.bundle: dict | None = None
        self.predictions: pd.DataFrame | None = None
        self.optimised: pd.DataFrame | None = None
        self.out_dir = str(Path.home() / "MLWorkbench_runs")

        self._style()
        self._menu()
        self._header()
        self.nb = ttk.Notebook(self)
        self.nb.pack(fill="both", expand=True, padx=12, pady=(0, 6))
        self._build_data_tab()
        self._build_train_tab()
        self._build_results_tab()
        self._build_predict_tab()
        self._build_optimise_tab()
        self._build_molecules_tab()
        self._build_statusbar()
        self.log(f"Output folder for runs: {self.out_dir}")
        self.log("Start with ‘Open CSV…’ on the Data tab. Nothing is uploaded anywhere — all local.")

    # ------------------------------------------------------------------ style
    def _style(self):
        st = ttk.Style(self)
        try:
            st.theme_use("clam")
        except tk.TclError:
            pass
        st.configure(".", font=FONT, background=BG, foreground=INK)
        st.configure("TFrame", background=BG)
        st.configure("Card.TFrame", background=CARD)
        st.configure("TLabel", background=BG, foreground=INK)
        st.configure("Card.TLabel", background=CARD, foreground=INK)
        st.configure("Muted.TLabel", background=BG, foreground=MUTED)
        st.configure("MutedCard.TLabel", background=CARD, foreground=MUTED)
        st.configure("H1.TLabel", background=BG, foreground=INK, font=FONT_H1)
        st.configure("H2.TLabel", background=BG, foreground=ACCENT_DARK, font=FONT_H2)
        st.configure("TButton", padding=(10, 5))
        st.configure("Accent.TButton", background=ACCENT, foreground="white", font=FONT_B, padding=(12, 7))
        st.map("Accent.TButton", background=[("active", ACCENT_DARK), ("disabled", "#9dbdb7")])
        st.configure("TNotebook", background=BG, borderwidth=0)
        st.configure("TNotebook.Tab", padding=(16, 8), font=FONT_B, background="#e6edef")
        st.map("TNotebook.Tab", background=[("selected", CARD)], foreground=[("selected", ACCENT_DARK)])
        st.configure("TLabelframe", background=BG, borderwidth=1)
        st.configure("TLabelframe.Label", background=BG, foreground=ACCENT_DARK, font=FONT_B)
        st.configure("Treeview", background=CARD, fieldbackground=CARD, rowheight=22, borderwidth=0, font=FONT)
        st.configure("Treeview.Heading", font=FONT_B, background="#e6edef", foreground=INK)
        st.map("Treeview", background=[("selected", "#cdeae4")], foreground=[("selected", INK)])
        st.configure("TCheckbutton", background=CARD)
        st.configure("Card.TCheckbutton", background=CARD)
        st.configure("TRadiobutton", background=BG)
        st.configure("TEntry", fieldbackground="white")
        st.configure("Horizontal.TProgressbar", background=ACCENT, troughcolor="#e6edef")

    def _menu(self):
        m = tk.Menu(self)
        f = tk.Menu(m, tearoff=0)
        f.add_command(label="Open data CSV…", command=self.open_csv, accelerator="Ctrl+O")
        f.add_command(label="Open run folder", command=lambda: open_in_explorer(self.out_dir))
        f.add_separator()
        f.add_command(label="Exit", command=self.destroy)
        m.add_cascade(label="File", menu=f)
        h = tk.Menu(m, tearoff=0)
        h.add_command(label="Quick guide", command=self.show_help)
        h.add_command(label="About", command=self.show_about)
        m.add_cascade(label="Help", menu=h)
        self.config(menu=m)
        self.bind_all("<Control-o>", lambda e: self.open_csv())

    def _header(self):
        bar = tk.Frame(self, bg="#0d6f63", height=64)
        bar.pack(fill="x")
        inner = tk.Frame(bar, bg="#0d6f63")
        inner.pack(fill="x", padx=16, pady=10)
        tk.Label(inner, text="ML Workbench", bg="#0d6f63", fg="white", font=FONT_H1).pack(side="left")
        tk.Label(inner, text="  honest, reproducible machine learning from a CSV",
                 bg="#0d6f63", fg="#bfe3dc", font=FONT).pack(side="left", padx=(6, 0), pady=(4, 0))
        self.header_status = tk.Label(inner, text="no data loaded", bg="#0d6f63", fg="#d8f0ec", font=FONT)
        self.header_status.pack(side="right", pady=(4, 0))

    def _build_statusbar(self):
        bar = ttk.Frame(self)
        bar.pack(fill="x", side="bottom")
        self.progress = ttk.Progressbar(bar, mode="determinate", length=260)
        self.progress.pack(side="left", padx=12, pady=6)
        self.status = ttk.Label(bar, text="ready", style="Muted.TLabel")
        self.status.pack(side="left")
        ttk.Button(bar, text="Open runs folder", command=lambda: open_in_explorer(self.out_dir)).pack(side="right", padx=8)

    # ------------------------------------------------------------- utilities
    def log(self, message: str, tag: str = "info"):
        ts = pd.Timestamp.now().strftime("%H:%M:%S")
        self.logbox.configure(state="normal")
        self.logbox.insert("end", f"[{ts}] {message}\n", tag)
        self.logbox.see("end")
        self.logbox.configure(state="disabled")

    def set_status(self, message: str):
        self.status.configure(text=message)
        self.update_idletasks()

    def set_progress(self, value: float):
        self.progress["value"] = max(0, min(100, value * 100))
        self.update_idletasks()

    # --------------------------------------------------------------- Data tab
    def _build_data_tab(self):
        tab = ttk.Frame(self.nb)
        self.nb.add(tab, text="1 · Data")

        top = ttk.Frame(tab)
        top.pack(fill="x", pady=(10, 6))
        ttk.Button(top, text="Open CSV…", style="Accent.TButton", command=self.open_csv).pack(side="left")
        self.lbl_file = ttk.Label(top, text="No file loaded", style="Muted.TLabel")
        self.lbl_file.pack(side="left", padx=12)
        self.lbl_task = ttk.Label(top, text="", style="H2.TLabel")
        self.lbl_task.pack(side="right")

        panes = ttk.PanedWindow(tab, orient="horizontal")
        panes.pack(fill="both", expand=True)

        # --- left: preview
        left = ttk.Labelframe(panes, text=" Preview (first 200 rows) ")
        pgrid = ttk.Frame(left)
        pgrid.pack(fill="both", expand=True, padx=8, pady=8)
        self.preview = ttk.Treeview(pgrid, show="headings", height=18)
        pvsb = ttk.Scrollbar(pgrid, orient="vertical", command=self.preview.yview)
        phsb = ttk.Scrollbar(pgrid, orient="horizontal", command=self.preview.xview)
        self.preview.configure(yscrollcommand=pvsb.set, xscrollcommand=phsb.set)
        self.preview.grid(row=0, column=0, sticky="nsew")
        pvsb.grid(row=0, column=1, sticky="ns")
        phsb.grid(row=1, column=0, sticky="ew")
        pgrid.rowconfigure(0, weight=1)
        pgrid.columnconfigure(0, weight=1)
        self.preview.bind("<MouseWheel>", self._preview_wheel)
        panes.add(left, weight=3)

        # --- right: roles + audit
        right = ttk.Frame(panes)
        panes.add(right, weight=2)

        roles = ttk.Labelframe(right, text=" What are we predicting? ")
        roles.pack(fill="x", pady=(0, 8))
        row = ttk.Frame(roles)
        row.pack(fill="x", padx=8, pady=8)
        ttk.Label(row, text="Target column:", style="Card.TLabel").pack(side="left")
        self.cmb_target = ttk.Combobox(row, state="readonly", width=26)
        self.cmb_target.pack(side="left", padx=8)
        self.cmb_target.bind("<<ComboboxSelected>>", lambda e: self._on_target_change())
        self.lbl_inferred = ttk.Label(roles, text="", style="MutedCard.TLabel")
        self.lbl_inferred.pack(fill="x", padx=10, pady=(0, 6), anchor="w")

        feats = ttk.Labelframe(right, text=" Features (untick to exclude) ")
        feats.pack(fill="both", expand=True, pady=(0, 8))
        bar = ttk.Frame(feats)
        bar.pack(fill="x", padx=8, pady=(8, 0))
        ttk.Button(bar, text="all", width=5, command=lambda: self._set_all_features(True)).pack(side="left")
        ttk.Button(bar, text="none", width=5, command=lambda: self._set_all_features(False)).pack(side="left", padx=4)
        self.lbl_feature_count = ttk.Label(bar, text="", style="MutedCard.TLabel")
        self.lbl_feature_count.pack(side="right")
        self.feat_scroll = ScrollFrame(feats)
        self.feat_scroll.pack(fill="both", expand=True, padx=8, pady=8)

        audit_frame = ttk.Labelframe(right, text=" Data health check ")
        audit_frame.pack(fill="both", expand=True)
        self.audit_tree = ttk.Treeview(audit_frame, columns=("level", "message"), show="headings", height=7)
        self.audit_tree.heading("level", text="")
        self.audit_tree.heading("message", text="finding")
        self.audit_tree.column("level", width=64, stretch=False)
        self.audit_tree.column("message", width=420)
        self.audit_tree.tag_configure("error", foreground=RED)
        self.audit_tree.tag_configure("warning", foreground=AMBER)
        self.audit_tree.tag_configure("info", foreground=MUTED)
        self.audit_tree.pack(fill="both", expand=True, padx=8, pady=8)
        self.audit_tree.bind("<<TreeviewSelect>>", self._show_audit_hint)
        self.lbl_hint = ttk.Label(audit_frame, text="", style="MutedCard.TLabel", wraplength=420, justify="left")
        self.lbl_hint.pack(fill="x", padx=10, pady=(0, 8))

    def _preview_wheel(self, event):
        self.preview.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def _set_all_features(self, value: bool):
        for var in self.feature_vars.values():
            var.set(value)
        self._update_feature_count()

    def _update_feature_count(self):
        n = sum(1 for v in self.feature_vars.values() if v.get())
        self.lbl_feature_count.configure(text=f"{n} selected")

    def open_csv(self):
        path = filedialog.askopenfilename(title="Open a CSV file",
                                          filetypes=[("CSV files", "*.csv"), ("Text files", "*.txt"), ("All files", "*.*")])
        if not path:
            return
        try:
            df, meta = core.load_csv(path)
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Could not read the file", str(exc))
            return
        self.df, self.meta, self.data_path = df, meta, path
        self.header_status.configure(text=f"{Path(path).name} · {len(df)} rows × {df.shape[1]} cols")
        self.lbl_file.configure(text=f"{path}  ({meta['encoding']}, separator {meta['sep']!r})")
        self._populate_preview()
        self._populate_roles()
        self.audit_target = ""
        self._run_audit(None)
        self.log(f"Loaded {Path(path).name}: {len(df)} rows, {df.shape[1]} columns, encoding {meta['encoding']}")
        self.nb.select(0)

    def _populate_preview(self):
        tv = self.preview
        tv.delete(*tv.get_children())
        cols = list(self.df.columns)
        tv["columns"] = cols
        for c in cols:
            tv.heading(c, text=c)
            tv.column(c, width=max(80, min(190, 12 + 8 * len(str(c)))), stretch=True)
        for _, row in self.df.head(200).iterrows():
            tv.insert("", "end", values=[("" if pd.isna(v) else v) for v in row.tolist()])

    def _populate_roles(self):
        cols = list(self.df.columns)
        self.cmb_target["values"] = cols
        if cols:
            guess = self._guess_target(cols)
            self.cmb_target.set(guess)
        self.feat_scroll.clear()
        self.feature_vars: dict[str, tk.BooleanVar] = {}
        for c in cols:
            var = tk.BooleanVar(value=True)
            self.feature_vars[c] = var
            cb = ttk.Checkbutton(self.feat_scroll.inner, text=c, variable=var,
                                 style="Card.TCheckbutton", command=self._update_feature_count)
            cb.pack(anchor="w", padx=4, pady=1)
        self._update_feature_count()
        self._on_target_change()

    @staticmethod
    def _guess_target(cols: list[str]) -> str:
        for c in cols:
            low = c.lower()
            if low in ("target", "y", "label", "class", "grade", "yield_pct") or low.startswith("target"):
                return c
        return cols[-1] if cols else ""

    def _on_target_change(self):
        target = self.cmb_target.get()
        if not target or self.df is None:
            return
        task = core.infer_task(self.df[target])
        self.lbl_inferred.configure(
            text=f"Detected task: {task}   ·   "
                 + (f"{self.df[target].nunique()} distinct values" if task == "classification"
                    else f"range {self.df[target].min():.4g} … {self.df[target].max():.4g}")
        )
        self.lbl_task.configure(text=f"{task} · {target}")
        if hasattr(self, "model_vars"):
            self._refresh_models(task)
        if hasattr(self, "cmb_group"):
            self.cmb_group["values"] = [c for c in self.df.columns if c != target]
        if target in self.feature_vars:
            self.feature_vars[target].set(False)
        self._update_feature_count()
        self._run_audit(target)

    def _run_audit(self, target: str | None):
        self.audit_tree.delete(*self.audit_tree.get_children())
        if self.df is None:
            return
        self.issues = core.audit(self.df, target=target or None)
        order = {"error": 0, "warning": 1, "info": 2}
        for issue in sorted(self.issues, key=lambda i: order[i.level]):
            self.audit_tree.insert("", "end", values=(issue.level, issue.message), tags=(issue.level,))
        self.audit_target = target or ""
        n_err = sum(1 for i in self.issues if i.level == "error")
        n_warn = sum(1 for i in self.issues if i.level == "warning")
        self.set_status(f"audit: {n_err} error(s), {n_warn} warning(s)")

    def _show_audit_hint(self, _event=None):
        sel = self.audit_tree.selection()
        if not sel:
            return
        idx = self.audit_tree.index(sel[0])
        order = {"error": 0, "warning": 1, "info": 2}
        sorted_issues = sorted(self.issues, key=lambda i: order[i.level])
        if idx < len(sorted_issues):
            issue = sorted_issues[idx]
            self.lbl_hint.configure(text=f"→ {issue.hint}" if issue.hint else "")

    # -------------------------------------------------------------- Train tab
    def _build_train_tab(self):
        tab = ttk.Frame(self.nb)
        self.nb.add(tab, text="2 · Train")

        upper = ttk.Frame(tab)
        upper.pack(fill="both", expand=True, pady=(10, 0))
        left = ttk.Labelframe(upper, text=" Models ")
        left.pack(side="left", fill="both", expand=True, padx=(0, 8))
        bar = ttk.Frame(left)
        bar.pack(fill="x", padx=8, pady=(8, 0))
        ttk.Button(bar, text="recommended", command=lambda: self._preset_models("auto")).pack(side="left")
        ttk.Button(bar, text="all", command=lambda: self._preset_models("all")).pack(side="left", padx=4)
        ttk.Button(bar, text="none", command=lambda: self._preset_models("none")).pack(side="left")
        self.model_scroll = ScrollFrame(left)
        self.model_scroll.pack(fill="both", expand=True, padx=8, pady=8)
        self.model_vars: dict[str, tk.BooleanVar] = {}

        right = ttk.Frame(upper)
        right.pack(side="right", fill="both", padx=(0, 0))
        right.configure(width=380)

        logframe = ttk.Labelframe(tab, text=" Log ")
        logframe.pack(fill="both", expand=False, pady=(10, 10))
        self.logbox = tk.Text(logframe, height=8, wrap="word", bg="#12212a", fg="#dbe7ec",
                              insertbackground="#dbe7ec", relief="flat", font=FONT_MONO,
                              highlightthickness=0)
        self.logbox.pack(fill="both", expand=True, padx=8, pady=8)
        self.logbox.tag_configure("error", foreground="#ff9d94")
        self.logbox.tag_configure("ok", foreground="#7fd1c0")
        self.logbox.configure(state="disabled")
        self._refresh_models("regression")

        settings = ttk.Labelframe(right, text=" Validation ")
        settings.pack(fill="x")
        grid = ttk.Frame(settings)
        grid.pack(fill="x", padx=10, pady=8)
        grid.columnconfigure(1, weight=1)

        r = 0
        ttk.Label(grid, text="Task", style="Card.TLabel").grid(row=r, column=0, sticky="w", pady=3)
        self.cmb_task = ttk.Combobox(grid, state="readonly", values=["auto", "regression", "classification"], width=16)
        self.cmb_task.set("auto")
        self.cmb_task.grid(row=r, column=1, sticky="w", pady=3)
        r += 1
        ttk.Label(grid, text="Folds", style="Card.TLabel").grid(row=r, column=0, sticky="w", pady=3)
        self.spn_folds = ttk.Spinbox(grid, from_=2, to=20, width=6)
        self.spn_folds.set(5)
        self.spn_folds.grid(row=r, column=1, sticky="w", pady=3)
        r += 1
        ttk.Label(grid, text="CV scheme", style="Card.TLabel").grid(row=r, column=0, sticky="w", pady=3)
        self.cmb_cv = ttk.Combobox(grid, state="readonly", values=["kfold", "group", "timeseries"], width=16)
        self.cmb_cv.set("kfold")
        self.cmb_cv.grid(row=r, column=1, sticky="w", pady=3)
        self.cmb_cv.bind("<<ComboboxSelected>>", lambda e: self._cv_changed())
        r += 1
        ttk.Label(grid, text="Group column", style="Card.TLabel").grid(row=r, column=0, sticky="w", pady=3)
        self.cmb_group = ttk.Combobox(grid, state="readonly", width=16, values=[])
        self.cmb_group.grid(row=r, column=1, sticky="w", pady=3)
        self.cmb_group.configure(state="disabled")
        r += 1
        ttk.Label(grid, text="Holdout fraction", style="Card.TLabel").grid(row=r, column=0, sticky="w", pady=3)
        hold = ttk.Frame(grid)
        hold.grid(row=r, column=1, sticky="w", pady=3)
        self.var_holdout = tk.BooleanVar(value=True)
        ttk.Checkbutton(hold, text="", variable=self.var_holdout).pack(side="left")
        self.spn_holdout = ttk.Spinbox(hold, from_=0.05, to=0.5, increment=0.05, width=6)
        self.spn_holdout.set(0.2)
        self.spn_holdout.pack(side="left", padx=4)
        r += 1
        ttk.Label(grid, text="Random seed", style="Card.TLabel").grid(row=r, column=0, sticky="w", pady=3)
        self.spn_seed = ttk.Spinbox(grid, from_=0, to=99999, width=6)
        self.spn_seed.set(77)
        self.spn_seed.grid(row=r, column=1, sticky="w", pady=3)
        r += 1
        ttk.Label(grid, text="Search budget", style="Card.TLabel").grid(row=r, column=0, sticky="w", pady=3)
        self.spn_search = ttk.Spinbox(grid, from_=0, to=200, increment=5, width=6)
        self.spn_search.set(12)
        self.spn_search.grid(row=r, column=1, sticky="w", pady=3)
        ttk.Label(grid, text="(0 = library defaults)", style="MutedCard.TLabel").grid(row=r, column=2, sticky="w", padx=4)
        r += 1
        self.var_balance = tk.BooleanVar(value=False)
        ttk.Checkbutton(grid, text="Balance classes (class_weight)", variable=self.var_balance,
                        style="Card.TCheckbutton").grid(row=r, column=0, columnspan=3, sticky="w", pady=(6, 0))
        r += 1
        self.var_saveall = tk.BooleanVar(value=False)
        ttk.Checkbutton(grid, text="Save every model, not just the chosen one",
                        variable=self.var_saveall, style="Card.TCheckbutton").grid(
            row=r, column=0, columnspan=3, sticky="w", pady=(2, 0))

        lagbox = ttk.Labelframe(right, text=" Time-series features (optional) ")
        lagbox.pack(fill="x", pady=(10, 0))
        lg = ttk.Frame(lagbox)
        lg.pack(fill="x", padx=10, pady=8)
        ttk.Label(lg, text="Lags", style="Card.TLabel").pack(side="left")
        self.ent_lags = ttk.Entry(lg, width=12)
        self.ent_lags.insert(0, "1,2,3")
        self.ent_lags.pack(side="left", padx=6)
        ttk.Label(lg, text="e.g. 1,2,7", style="MutedCard.TLabel").pack(side="left")
        ttk.Label(lagbox, text="Adds lag + rolling-mean columns per group, then drops rows\nwithout history.",
                  style="MutedCard.TLabel", justify="left").pack(fill="x", padx=10, pady=(0, 8))

        runbox = ttk.Frame(right)
        runbox.pack(fill="x", pady=(12, 0))
        ttk.Button(runbox, text="Run training", style="Accent.TButton", command=self.run_training).pack(fill="x")
        ttk.Label(runbox, text="Writes run.json, report.md, metrics.csv, plots/ and models/ to the runs folder.",
                  style="Muted.TLabel", wraplength=360, justify="left").pack(fill="x", pady=6)



    def _cv_changed(self):
        state = "readonly" if self.cmb_cv.get() == "group" else "disabled"
        self.cmb_group.configure(state=state)

    def _preset_models(self, which: str):
        if which == "none":
            for v in self.model_vars.values():
                v.set(False)
            return
        if which == "all":
            for v in self.model_vars.values():
                v.set(True)
            return
        task = self._effective_task()
        auto = set(core.AUTO_MODELS.get(task, []))
        for name, var in self.model_vars.items():
            var.set(name in auto)

    def _effective_task(self) -> str:
        task = self.cmb_task.get()
        if task != "auto":
            return task
        target = self.cmb_target.get()
        if self.df is not None and target:
            return core.infer_task(self.df[target])
        return "regression"

    def _refresh_models(self, task: str):
        self.model_scroll.clear()
        self.model_vars = {}
        auto = set(core.AUTO_MODELS.get(task, []))
        for name in core.available_models(task):
            var = tk.BooleanVar(value=name in auto)
            self.model_vars[name] = var
            label = name + ("  ★" if name in auto else "")
            ttk.Checkbutton(self.model_scroll.inner, text=label, variable=var,
                            style="Card.TCheckbutton").pack(anchor="w", padx=6, pady=1)

    def _selected_features(self) -> list[str]:
        target = self.cmb_target.get()
        return [c for c, v in getattr(self, "feature_vars", {}).items() if v.get() and c != target]

    # ------------------------------------------------------------ training run
    def run_training(self):
        if self.df is None:
            messagebox.showinfo("No data", "Open a CSV file on the Data tab first.")
            return
        target = self.cmb_target.get()
        features = self._selected_features()
        if not features:
            messagebox.showinfo("No features", "Select at least one feature column.")
            return
        task = self._effective_task()
        models = [m for m, v in self.model_vars.items() if v.get()]
        if not models:
            messagebox.showinfo("No models", "Pick at least one model (or press ‘recommended’).")
            return

        df = self.df
        lags_text = self.ent_lags.get().strip()
        if lags_text and self.cmb_cv.get() != "kfold":
            pass
        df_use = df
        if lags_text and target:
            try:
                lags = [int(x) for x in lags_text.split(",") if x.strip()]
                group = self.cmb_group.get() if self.cmb_cv.get() == "group" else None
                df_use = core.add_lag_features(df, target, group_col=group, lags=lags, rolls=(3,))
                self.log(f"Lag features added: {len(df)} → {len(df_use)} rows; "
                         f"{len([c for c in df_use.columns if str(c).startswith(target + '_')])} new columns")
                features = [f for f in features if f in df_use.columns] + \
                           [c for c in df_use.columns if str(c).startswith(target + "_")]
            except Exception as exc:  # noqa: BLE001
                messagebox.showerror("Lag features failed", str(exc))
                return

        cfg = core.TrainConfig(
            target=target,
            features=features,
            task=task,
            folds=int(self.spn_folds.get() or 5),
            cv_strategy=self.cmb_cv.get(),
            group_col=(self.cmb_group.get() or None) if self.cmb_cv.get() == "group" else None,
            holdout_frac=float(self.spn_holdout.get() or 0) if self.var_holdout.get() else 0.0,
            seed=int(self.spn_seed.get() or 77),
            search_iters=int(self.spn_search.get() or 0),
            models=models,
            balance=self.var_balance.get(),
            save_all_models=self.var_saveall.get(),
        )

        self.set_progress(0)
        self.set_status("training…")
        self.log("=" * 60)
        self.log(f"Training {len(models)} model(s) on {len(df_use)} rows, target '{target}' [{task}]")

        def work(log, progress):
            return core.train_and_evaluate(df_use, cfg, out_root=self.out_dir, name=Path(self.data_path).stem or "run",
                                           log=log, progress=progress, source_path=self.data_path or None)

        self._run_async(work, on_done=self._training_done)

    def _training_done(self, result: dict):
        self.result = result
        self.model_path = result["model_path"]
        self.set_progress(1)
        self.set_status(f"done · recommended {result['chosen']}")
        self._fill_results(result)
        self.nb.select(2)
        self.ent_pred_model.delete(0, "end")
        self.ent_pred_model.insert(0, result["model_path"])
        self.ent_opt_model.delete(0, "end")
        self.ent_opt_model.insert(0, result["model_path"])

    # ------------------------------------------------------------ Results tab
    def _build_results_tab(self):
        tab = ttk.Frame(self.nb)
        self.nb.add(tab, text="3 · Results")

        self.lbl_reco = ttk.Label(tab, text="Run a training job to see results here.",
                                  style="H2.TLabel", wraplength=1100, justify="left")
        self.lbl_reco.pack(fill="x", pady=(10, 6), anchor="w")

        table_frame = ttk.Labelframe(tab, text=" Model comparison (click a column header to sort) ")
        table_frame.pack(fill="both", expand=True)
        tgrid = ttk.Frame(table_frame)
        tgrid.pack(fill="both", expand=True, padx=8, pady=8)
        self.results_tree = ttk.Treeview(tgrid, show="headings", height=9)
        vsb = ttk.Scrollbar(tgrid, orient="vertical", command=self.results_tree.yview)
        hsb = ttk.Scrollbar(tgrid, orient="horizontal", command=self.results_tree.xview)
        self.results_tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        self.results_tree.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        hsb.grid(row=1, column=0, sticky="ew")
        tgrid.rowconfigure(0, weight=1)
        tgrid.columnconfigure(0, weight=1)
        self.results_tree.tag_configure("baseline", foreground=MUTED)
        self.results_tree.tag_configure("chosen", background="#e6f4f2")
        self.results_tree.bind("<Double-1>", self._show_model_detail)

        bottom = ttk.Frame(tab)
        bottom.pack(fill="both", expand=True, pady=(8, 10))

        imp_box = ttk.Labelframe(bottom, text=" What drives the prediction ")
        imp_box.pack(side="left", fill="both", expand=True, padx=(0, 8))
        self.imp_tree = ttk.Treeview(imp_box, columns=("feature", "importance"), show="headings", height=7)
        self.imp_tree.heading("feature", text="feature")
        self.imp_tree.heading("importance", text="importance")
        self.imp_tree.column("feature", width=200)
        self.imp_tree.column("importance", width=90, anchor="e")
        self.imp_tree.pack(fill="both", expand=True, padx=8, pady=8)

        right = ttk.Frame(bottom)
        right.pack(side="right", fill="both", padx=(0, 0))
        files = ttk.Labelframe(right, text=" Artefacts ")
        files.pack(fill="x")
        self.lbl_artefacts = ttk.Label(files, text="—", style="MutedCard.TLabel", justify="left")
        self.lbl_artefacts.pack(fill="x", padx=10, pady=8)
        ttk.Button(files, text="Open run folder", command=lambda: open_in_explorer(self.result["run_dir"]) if self.result else None).pack(fill="x", padx=10)
        ttk.Button(files, text="Open report.md", command=lambda: open_in_explorer(self.result["report_path"]) if self.result else None).pack(fill="x", padx=10, pady=4)
        ttk.Button(files, text="View a plot…", command=self._pick_plot).pack(fill="x", padx=10, pady=(0, 10))
        ttk.Button(files, text="Send model to Predict tab",
                   command=self._send_model_to_predict).pack(fill="x", padx=10, pady=(0, 10))

    def _send_model_to_predict(self):
        if not self.model_path:
            return
        self.ent_pred_model.delete(0, "end")
        self.ent_pred_model.insert(0, self.model_path)
        self.ent_opt_model.delete(0, "end")
        self.ent_opt_model.insert(0, self.model_path)
        self.nb.select(3)
        self.log(f"model handed to the Predict tab: {Path(self.model_path).name}")

    def _fill_results(self, result: dict):
        self.lbl_reco.configure(text=result["recommendation"])
        mt = result["metrics_table"]
        tv = self.results_tree
        tv.delete(*tv.get_children())
        cols = list(mt.columns)
        tv["columns"] = cols
        for c in cols:
            tv.heading(c, text=DISPLAY_METRIC.get(c, c), command=lambda cc=c: self._sort_results(cc))
            width = 190 if c == "model" else (96 if "holdout" in c else 84)
            tv.column(c, width=width, anchor="w" if c == "model" else "e", stretch=(c == "model"))
        for _, row in mt.iterrows():
            values = ["" if (isinstance(v, float) and pd.isna(v)) else
                      (f"{v:.4f}" if isinstance(v, float) else v) for v in row.tolist()]
            tags = ()
            if "BASELINE" in str(row.get("model", "")):
                tags = ("baseline",)
            elif str(row.get("model")) == result["chosen"]:
                tags = ("chosen",)
            tv.insert("", "end", values=values, tags=tags)

        self.imp_tree.delete(*self.imp_tree.get_children())
        for row in result["importance"][:20]:
            self.imp_tree.insert("", "end", values=(row["feature"], f"{row['importance']:+.4f}"))

        plots = result.get("plots", {})
        self.lbl_artefacts.configure(
            text=f"run.json · report.md · metrics.csv\n{len(plots)} plot(s) · models/{Path(result['model_path']).name}")
        self.log(f"Results ready. Recommended: {result['chosen']}")

    def _sort_results(self, column: str):
        if not self.result:
            return
        mt = self.result["metrics_table"]
        by_num = column != "model"
        key = (lambda v: pd.to_numeric(v, errors="coerce")) if by_num else (lambda v: str(v))
        mt = mt.assign(_k=mt[column].map(key)).sort_values("_k", ascending=not by_num, na_position="last").drop(columns="_k")
        self.result["metrics_table"] = mt
        self._fill_results(self.result)

    def _show_model_detail(self, _event=None):
        sel = self.results_tree.selection()
        if not sel or not self.result:
            return
        name = self.results_tree.item(sel[0], "values")[0]
        row = next((r for r in self.result["results"] if r["model"] == name), None)
        if not row:
            messagebox.showinfo(name, "Baseline model — it predicts the mean / majority class.")
            return
        lines = [f"{name}", "", f"hyperparameters: {row['params']}", f"complexity rank: {row['complexity']}",
                 f"fitted parameters: {row['n_params']}", f"fit time: {row['fit_seconds']}s", ""]
        for metric, stats in row["cv"].items():
            folds = ", ".join(f"{f:.3f}" for f in stats["folds"])
            lines.append(f"{metric:24s} {stats['mean']:.4f} ± {stats['std']:.4f}   folds: {folds}")
        if row.get("holdout"):
            lines += ["", "holdout (never used for tuning):"]
            lines += [f"  {k:22s} {v:.4f}" for k, v in row["holdout"].items()]
        messagebox.showinfo(f"Model detail — {name}", "\n".join(lines))

    def _pick_plot(self):
        if not self.result or not self.result.get("plots"):
            return
        import tkinter.simpledialog as sd
        names = list(self.result["plots"])
        choice = sd.askstring("Open plot", "Available plots:\n\n  " + "\n  ".join(names) + "\n\nType one name:")
        if choice and choice.strip() in self.result["plots"]:
            open_in_explorer(self.result["plots"][choice.strip()])

    # ------------------------------------------------------------ Predict tab
    def _build_predict_tab(self):
        tab = ttk.Frame(self.nb)
        self.nb.add(tab, text="4 · Predict")

        top = ttk.Labelframe(tab, text=" Model and data ")
        top.pack(fill="x", pady=10)
        g = ttk.Frame(top)
        g.pack(fill="x", padx=10, pady=8)
        g.columnconfigure(1, weight=1)

        ttk.Label(g, text="Model (.pkl)", style="Card.TLabel").grid(row=0, column=0, sticky="w", pady=3)
        self.ent_pred_model = ttk.Entry(g)
        self.ent_pred_model.grid(row=0, column=1, sticky="ew", padx=8, pady=3)
        ttk.Button(g, text="Browse…", command=self._pick_model).grid(row=0, column=2, pady=3)

        ttk.Label(g, text="New data (CSV)", style="Card.TLabel").grid(row=1, column=0, sticky="w", pady=3)
        self.ent_pred_data = ttk.Entry(g)
        self.ent_pred_data.grid(row=1, column=1, sticky="ew", padx=8, pady=3)
        ttk.Button(g, text="Browse…", command=self._pick_predict_data).grid(row=1, column=2, pady=3)

        ttk.Label(g, text="Output CSV", style="Card.TLabel").grid(row=2, column=0, sticky="w", pady=3)
        self.ent_pred_out = ttk.Entry(g)
        self.ent_pred_out.grid(row=2, column=1, sticky="ew", padx=8, pady=3)
        ttk.Button(g, text="Save as…", command=self._pick_pred_out).grid(row=2, column=2, pady=3)

        ttk.Button(tab, text="Validate & predict", style="Accent.TButton",
                   command=self.run_predict).pack(anchor="w", pady=(0, 8))

        self.pred_report = tk.Text(tab, height=8, wrap="word", bg=CARD, relief="flat", font=FONT,
                                   highlightthickness=1, highlightbackground="#dfe6e9")
        self.pred_report.pack(fill="x")
        self.pred_report.tag_configure("bad", foreground=RED)
        self.pred_report.tag_configure("warn", foreground=AMBER)
        self.pred_report.tag_configure("ok", foreground=ACCENT_DARK)

        prev = ttk.Labelframe(tab, text=" Output preview ")
        prev.pack(fill="both", expand=True, pady=(8, 10))
        self.pred_tree = ttk.Treeview(prev, show="headings", height=8)
        self.pred_tree.pack(fill="both", expand=True, padx=8, pady=8)

    def _pick_model(self):
        path = filedialog.askopenfilename(title="Open a saved model",
                                          filetypes=[("ML Workbench model", "*.pkl"), ("All files", "*.*")])
        if path:
            self.ent_pred_model.delete(0, "end")
            self.ent_pred_model.insert(0, path)

    def _pick_predict_data(self):
        path = filedialog.askopenfilename(title="Open the CSV to predict",
                                          filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
        if path:
            self.ent_pred_data.delete(0, "end")
            self.ent_pred_data.insert(0, path)
            if not self.ent_pred_out.get():
                out = Path(path).with_name(Path(path).stem + "_predictions.csv")
                self.ent_pred_out.insert(0, str(out))

    def _pick_pred_out(self):
        path = filedialog.asksaveasfilename(title="Save predictions as", defaultextension=".csv",
                                            filetypes=[("CSV files", "*.csv")])
        if path:
            self.ent_pred_out.delete(0, "end")
            self.ent_pred_out.insert(0, path)

    def run_predict(self):
        model = self.ent_pred_model.get().strip()
        data = self.ent_pred_data.get().strip()
        out = self.ent_pred_out.get().strip()
        if not model or not data:
            messagebox.showinfo("Missing input", "Choose a model .pkl and a CSV to predict.")
            return
        try:
            bundle = core.load_model(model)
            df, _ = core.load_csv(data)
            pred, report = core.predict_frame(bundle, df)
        except Exception as exc:  # noqa: BLE001
            self.pred_report.configure(state="normal")
            self.pred_report.delete("1.0", "end")
            self.pred_report.insert("end", f"FAILED\n\n{exc}", "bad")
            self.pred_report.configure(state="disabled")
            self.log(f"prediction failed: {exc}")
            return
        if not out:
            out = str(Path(data).with_name(Path(data).stem + "_predictions.csv"))
            self.ent_pred_out.delete(0, "end")
            self.ent_pred_out.insert(0, out)
        pred.to_csv(out, index=False)
        self.predictions = pred

        self.pred_report.configure(state="normal")
        self.pred_report.delete("1.0", "end")
        self.pred_report.insert("end", f"OK — {len(pred)} row(s) predicted with ‘{Path(model).name}’\n", "ok")
        self.pred_report.insert("end", f"target: {bundle['target']} ({bundle['task']})  ·  written to {out}\n")
        if report.get("coerced_to_numeric"):
            self.pred_report.insert("end", f"converted text→number in: {', '.join(report['coerced_to_numeric'])}\n", "warn")
        for w in report.get("warnings", []):
            self.pred_report.insert("end", f"WARNING: {w}\n", "warn")
        if report.get("out_of_range"):
            for col, info in report["out_of_range"].items():
                self.pred_report.insert(
                    "end", f"EXTRAPOLATION: {info['count']} row(s) of ‘{col}’ outside training range "
                           f"{info['training_range'][0]:.4g}…{info['training_range'][1]:.4g}\n", "bad")
        if not report.get("warnings"):
            self.pred_report.insert("end", "No schema problems, no unseen categories, no extrapolation.\n", "ok")
        self.pred_report.configure(state="disabled")

        tv = self.pred_tree
        tv.delete(*tv.get_children())
        cols = list(pred.columns)[:12]
        tv["columns"] = cols
        for c in cols:
            tv.heading(c, text=c)
            tv.column(c, width=120, anchor="w")
        for _, row in pred.head(100).iterrows():
            tv.insert("", "end", values=[("" if pd.isna(v) else (f"{v:.4f}" if isinstance(v, float) else v))
                                         for v in row[cols].tolist()])
        self.log(f"Predicted {len(pred)} rows → {out}")
        self.set_status(f"prediction written to {out}")

    # ----------------------------------------------------------- Optimise tab
    def _build_optimise_tab(self):
        tab = ttk.Frame(self.nb)
        self.nb.add(tab, text="5 · Optimise")

        top = ttk.Labelframe(tab, text=" Model ")
        top.pack(fill="x", pady=(10, 6))
        g = ttk.Frame(top)
        g.pack(fill="x", padx=10, pady=8)
        g.columnconfigure(1, weight=1)
        ttk.Label(g, text="Model (.pkl)", style="Card.TLabel").grid(row=0, column=0, sticky="w")
        self.ent_opt_model = ttk.Entry(g)
        self.ent_opt_model.grid(row=0, column=1, sticky="ew", padx=8)
        ttk.Button(g, text="Browse…", command=lambda: self._pick_model_into(self.ent_opt_model)).grid(row=0, column=2)
        ttk.Button(g, text="Load ranges", command=self.load_ranges).grid(row=0, column=3, padx=(8, 0))

        mid = ttk.Frame(tab)
        mid.pack(fill="both", expand=True)

        rng = ttk.Labelframe(mid, text=" Search space (edit min / max / steps) ")
        rng.pack(side="left", fill="both", expand=True, padx=(0, 8))
        self.range_holder = ScrollFrame(rng)
        self.range_holder.pack(fill="both", expand=True, padx=8, pady=8)
        self.range_entries: dict[str, tuple] = {}

        side = ttk.Frame(mid)
        side.pack(side="right", fill="y")
        opt = ttk.Labelframe(side, text=" Objective ")
        opt.pack(fill="x")
        self.var_maximise = tk.BooleanVar(value=True)
        ttk.Radiobutton(opt, text="maximise the prediction", variable=self.var_maximise, value=True).pack(anchor="w", padx=10, pady=2)
        ttk.Radiobutton(opt, text="minimise the prediction", variable=self.var_maximise, value=False).pack(anchor="w", padx=10, pady=2)
        rowk = ttk.Frame(opt)
        rowk.pack(fill="x", padx=10, pady=6)
        ttk.Label(rowk, text="show top", style="Card.TLabel").pack(side="left")
        self.spn_top = ttk.Spinbox(rowk, from_=5, to=200, width=6)
        self.spn_top.set(25)
        self.spn_top.pack(side="left", padx=6)
        ttk.Button(opt, text="Run optimisation", style="Accent.TButton", command=self.run_optimise).pack(fill="x", padx=10, pady=(6, 10))
        ttk.Button(opt, text="Save results as CSV…", command=self._save_optimised).pack(fill="x", padx=10, pady=(0, 10))

        out = ttk.Labelframe(tab, text=" Ranked candidates ")
        out.pack(fill="both", expand=True, pady=(8, 10))
        self.opt_tree = ttk.Treeview(out, show="headings", height=9)
        self.opt_tree.pack(fill="both", expand=True, padx=8, pady=8)
        self.opt_tree.tag_configure("edge", foreground=AMBER)
        self.opt_tree.tag_configure("outside", foreground=RED)
        ttk.Label(out, text="Amber = the best value sits on the edge of the grid (the true optimum may lie outside it). "
                            "Red = outside the range the model was trained on, i.e. extrapolation.",
                  style="Muted.TLabel", wraplength=1100, justify="left").pack(fill="x", padx=10, pady=(0, 8))

    def _pick_model_into(self, entry: ttk.Entry):
        path = filedialog.askopenfilename(title="Open a saved model",
                                          filetypes=[("ML Workbench model", "*.pkl"), ("All files", "*.*")])
        if path:
            entry.delete(0, "end")
            entry.insert(0, path)

    def load_ranges(self):
        path = self.ent_opt_model.get().strip()
        if not path:
            messagebox.showinfo("No model", "Choose a model first.")
            return
        try:
            self.bundle = core.load_model(path)
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Could not load model", str(exc))
            return
        self.range_holder.clear()
        self.range_entries = {}
        header = ttk.Frame(self.range_holder.inner)
        header.pack(fill="x")
        for text, width in (("feature", 26), ("min", 10), ("max", 10), ("steps", 8)):
            ttk.Label(header, text=text, style="H2.TLabel", width=width).pack(side="left", padx=2)
        for col in self.bundle["numeric"]:
            lo, hi = self.bundle["ranges"][col]
            row = ttk.Frame(self.range_holder.inner)
            row.pack(fill="x", pady=1)
            ttk.Label(row, text=col, style="Card.TLabel", width=26).pack(side="left", padx=2)
            e_lo = ttk.Entry(row, width=10)
            e_lo.insert(0, f"{lo:.6g}")
            e_lo.pack(side="left", padx=2)
            e_hi = ttk.Entry(row, width=10)
            e_hi.insert(0, f"{hi:.6g}")
            e_hi.pack(side="left", padx=2)
            e_st = ttk.Entry(row, width=8)
            e_st.insert(0, "6")
            e_st.pack(side="left", padx=2)
            self.range_entries[col] = (e_lo, e_hi, e_st)
        for col in self.bundle["categorical"]:
            row = ttk.Frame(self.range_holder.inner)
            row.pack(fill="x", pady=1)
            ttk.Label(row, text=col, style="Card.TLabel", width=26).pack(side="left", padx=2)
            e = ttk.Entry(row, width=32)
            levels = "|".join(self.bundle["categories"].get(col, [])[:8])
            e.insert(0, levels)
            e.pack(side="left", padx=2)
            self.range_entries[col] = (e, None, None)
        self.log(f"Ranges loaded for {len(self.range_entries)} feature(s); "
                 f"target is '{self.bundle['target']}' ({self.bundle['task']})")

    def run_optimise(self):
        if self.bundle is None:
            messagebox.showinfo("No model", "Press ‘Load ranges’ first.")
            return
        ranges, levels = {}, {}
        try:
            for col, widgets in self.range_entries.items():
                if col in self.bundle["numeric"]:
                    lo, hi, steps = widgets
                    ranges[col] = (float(lo.get()), float(hi.get()), int(steps.get()))
                else:
                    entry, _, _ = widgets
                    vals = [v.strip() for v in entry.get().split("|") if v.strip()]
                    if vals:
                        levels[col] = vals
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Bad range", f"Could not parse the ranges: {exc}")
            return
        try:
            grid = core.optimise_grid(self.bundle, ranges, levels,
                                      maximise=self.var_maximise.get(),
                                      top_n=int(self.spn_top.get() or 25), log=self.log)
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Optimisation failed", str(exc))
            return
        self.optimised = grid
        tv = self.opt_tree
        tv.delete(*tv.get_children())
        cols = list(grid.columns)
        tv["columns"] = cols
        for c in cols:
            tv.heading(c, text=c)
            tv.column(c, width=150 if (c in ranges or c in levels) else 120,
                      anchor="e" if c.startswith(("predicted", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9")) else "w")
        for i, (_, row) in enumerate(grid.head(int(self.spn_top.get() or 25)).iterrows()):
            values = []
            for v in row.tolist():
                if isinstance(v, float):
                    values.append(f"{v:.4g}")
                elif isinstance(v, (np.bool_, bool)):
                    values.append("yes" if v else "no")
                else:
                    values.append(v)
            tags = ()
            if not row["_inside_training_range"]:
                tags = ("outside",)
            elif row["_at_grid_boundary"]:
                tags = ("edge",)
            tv.insert("", "end", values=values, tags=tags)
        pred_col = f"predicted_{self.bundle['target']}"
        best = grid.iloc[0]
        self.set_status(f"best {pred_col} = {best[pred_col]:.4f}")
        self.log(f"Optimisation done: {len(grid):,} candidates; best {pred_col} = {best[pred_col]:.4f}"
                 f" (boundary: {best['_at_grid_boundary'] or 'no'},"
                 f" inside training range: {'yes' if best['_inside_training_range'] else 'NO'})")

    def _save_optimised(self):
        if self.optimised is None:
            return
        path = filedialog.asksaveasfilename(title="Save ranked candidates", defaultextension=".csv",
                                            filetypes=[("CSV files", "*.csv")])
        if path:
            self.optimised.to_csv(path, index=False)
            self.log(f"saved {len(self.optimised)} candidates to {path}")

    # ---------------------------------------------------------- Molecules tab
    def _build_molecules_tab(self):
        tab = ttk.Frame(self.nb)
        self.nb.add(tab, text="6 · Molecules")

        import importlib.util

        has_rdkit = importlib.util.find_spec("rdkit") is not None

        info = ttk.Label(
            tab,
            text="SMILES → descriptors or fingerprints. Salts and mixtures are standardised to the largest "
                 "fragment, unparsable rows are reported instead of silently producing blanks." if has_rdkit
            else "RDKit is not installed in this build, so the molecule features are unavailable. "
                 "Everything else in the app works normally.",
            style="Muted.TLabel", wraplength=1100, justify="left")
        info.pack(fill="x", pady=(10, 8), anchor="w")

        box = ttk.Labelframe(tab, text=" Input ")
        box.pack(fill="x")
        g = ttk.Frame(box)
        g.pack(fill="x", padx=10, pady=8)
        g.columnconfigure(1, weight=1)
        ttk.Label(g, text="CSV with SMILES", style="Card.TLabel").grid(row=0, column=0, sticky="w")
        self.ent_mol_csv = ttk.Entry(g)
        self.ent_mol_csv.grid(row=0, column=1, sticky="ew", padx=8)
        ttk.Button(g, text="Browse…", command=self._pick_mol_csv).grid(row=0, column=2)

        ttk.Label(g, text="SMILES column", style="Card.TLabel").grid(row=1, column=0, sticky="w", pady=6)
        self.cmb_mol_col = ttk.Combobox(g, state="readonly", width=24, values=["SMILES"])
        self.cmb_mol_col.set("SMILES")
        self.cmb_mol_col.grid(row=1, column=1, sticky="w", padx=8, pady=6)

        opts = ttk.Frame(box)
        opts.pack(fill="x", padx=10, pady=(0, 8))
        self.var_mol_mode = tk.StringVar(value="descriptors")
        ttk.Radiobutton(opts, text="descriptors", variable=self.var_mol_mode, value="descriptors").pack(side="left")
        ttk.Radiobutton(opts, text="Morgan fingerprint", variable=self.var_mol_mode, value="morgan").pack(side="left", padx=10)
        ttk.Radiobutton(opts, text="RDKit fingerprint", variable=self.var_mol_mode, value="rdkit").pack(side="left")
        ttk.Label(opts, text="   bits", style="Card.TLabel").pack(side="left", padx=(14, 2))
        self.spn_bits = ttk.Spinbox(opts, from_=64, to=4096, increment=128, width=7)
        self.spn_bits.set(1024)
        self.spn_bits.pack(side="left")
        ttk.Label(opts, text="radius", style="Card.TLabel").pack(side="left", padx=(10, 2))
        self.spn_radius = ttk.Spinbox(opts, from_=1, to=5, width=4)
        self.spn_radius.set(2)
        self.spn_radius.pack(side="left")
        self.var_standardise = tk.BooleanVar(value=True)
        ttk.Checkbutton(opts, text="standardise (largest fragment)", variable=self.var_standardise).pack(side="left", padx=14)

        btns = ttk.Frame(tab)
        btns.pack(fill="x", pady=10)
        ttk.Button(btns, text="Generate features", style="Accent.TButton",
                   command=self.run_molecules, state=("normal" if has_rdkit else "disabled")).pack(side="left")
        ttk.Button(btns, text="Check applicability domain vs training CSV…",
                   command=self.run_ad, state=("normal" if has_rdkit else "disabled")).pack(side="left", padx=8)

        prev = ttk.Labelframe(tab, text=" Output ")
        prev.pack(fill="both", expand=True, pady=(0, 10))
        self.mol_text = tk.Text(prev, height=14, wrap="word", bg=CARD, relief="flat", font=FONT_MONO,
                                highlightthickness=1, highlightbackground="#dfe6e9")
        self.mol_text.pack(fill="both", expand=True, padx=8, pady=8)

    def _pick_mol_csv(self):
        path = filedialog.askopenfilename(title="Open a CSV of molecules",
                                          filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
        if not path:
            return
        self.ent_mol_csv.delete(0, "end")
        self.ent_mol_csv.insert(0, path)
        try:
            df, _ = core.load_csv(path)
            self.cmb_mol_col["values"] = list(df.columns)
            for c in df.columns:
                if "smiles" in str(c).lower():
                    self.cmb_mol_col.set(c)
                    break
            self.log(f"molecule file: {len(df)} rows, columns {list(df.columns)}")
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Could not read file", str(exc))

    def run_molecules(self):
        path = self.ent_mol_csv.get().strip()
        if not path:
            messagebox.showinfo("No file", "Choose a CSV that contains a SMILES column.")
            return
        try:
            df, _ = core.load_csv(path)
            col = self.cmb_mol_col.get()
            if col not in df.columns:
                raise ValueError(f"Column '{col}' not found. Available: {list(df.columns)}")
            feats, notes = core.smiles_features(
                df[col], mode=self.var_mol_mode.get(), bits=int(self.spn_bits.get()),
                radius=int(self.spn_radius.get()), standardize=self.var_standardise.get())
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Molecule features failed", str(exc))
            return
        out = Path(path).with_name(Path(path).stem + "_features.csv")
        pd.concat([df.reset_index(drop=True), feats.reset_index(drop=True)], axis=1).to_csv(out, index=False)
        failed = [n for n in notes if n["status"] != "ok"]
        changed = [n for n in notes if n["note"]]
        self.mol_text.delete("1.0", "end")
        self.mol_text.insert("end", f"{len(feats)} molecules → {feats.shape[1]} feature columns\n")
        self.mol_text.insert("end", f"written to {out}\n\n")
        if failed:
            self.mol_text.insert("end", f"{len(failed)} row(s) could not be parsed: {[f['row'] for f in failed][:10]}\n")
        if changed:
            self.mol_text.insert("end", f"{len(changed)} molecule(s) standardised (salt/mixture → largest fragment)\n")
        self.mol_text.insert("end", "\n" + feats.head(12).to_string())
        self.log(f"molecule features: {feats.shape} → {out}")

    def run_ad(self):
        query = self.ent_mol_csv.get().strip()
        if not query:
            messagebox.showinfo("No file", "Choose the query CSV first.")
            return
        ref = filedialog.askopenfilename(title="Training set with SMILES (the model's own molecules)",
                                        filetypes=[("CSV files", "*.csv")])
        if not ref:
            return
        try:
            q, _ = core.load_csv(query)
            r, _ = core.load_csv(ref)
            col = self.cmb_mol_col.get()
            ad = core.applicability_domain(r[col], q[col])
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Applicability domain failed", str(exc))
            return
        out = Path(query).with_name(Path(query).stem + "_applicability_domain.csv")
        ad.to_csv(out, index=False)
        in_dom = int(ad["in_domain"].sum())
        self.mol_text.delete("1.0", "end")
        self.mol_text.insert("end", f"{in_dom}/{len(ad)} query molecules are inside the applicability domain "
                                    f"(Tanimoto ≥ 0.4 to the nearest training molecule)\n\n")
        self.mol_text.insert("end", ad.to_string())
        self.log(f"applicability domain → {out}")

    # ------------------------------------------------------------- threading
    def _run_async(self, work, on_done):
        q: queue.Queue = queue.Queue()

        def worker():
            try:
                result = work(lambda m: q.put(("log", m)), lambda p, m: q.put(("progress", p, m)))
                q.put(("done", result))
            except Exception as exc:  # noqa: BLE001
                q.put(("error", f"{exc}\n\n{traceback.format_exc()}"))

        threading.Thread(target=worker, daemon=True).start()
        self._poll(q, on_done)

    def _poll(self, q: queue.Queue, on_done):
        try:
            while True:
                item = q.get_nowait()
                if item[0] == "log":
                    self.log(item[1])
                elif item[0] == "progress":
                    self.set_progress(item[1])
                    self.set_status(item[2])
                elif item[0] == "done":
                    self.set_status("finished")
                    on_done(item[1])
                    return
                elif item[0] == "error":
                    self.set_status("failed")
                    self.log(item[1], "error")
                    messagebox.showerror("Something went wrong", item[1].split("\n\n")[0])
                    return
        except queue.Empty:
            pass
        self.after(120, lambda: self._poll(q, on_done))

    # ------------------------------------------------------------------ misc
    def show_help(self):
        messagebox.showinfo(
            "Quick guide",
            "1. Data — open a CSV. Pick the target column. Read the health-check list; "
            "red items must be fixed, amber items are judgement calls.\n\n"
            "2. Train — choose models (★ = sensible default), folds, seed and search budget. "
            "Keep 'holdout' ticked for an unbiased final number.\n\n"
            "3. Results — the table shows the baseline first: a model only counts if it beats it. "
            "The recommendation prefers the simplest model within one standard error of the best.\n\n"
            "4. Predict — the model contains its own preprocessing, so any CSV with the same columns works. "
            "Rows outside the training range are flagged as extrapolation.\n\n"
            "5. Optimise — sweep feature ranges, rank the predictions, watch the boundary warnings.\n\n"
            "6. Molecules — SMILES → descriptors or fingerprints, with standardisation and an "
            "applicability-domain check.\n\n"
            "Every run writes run.json + report.md so the work is reproducible and shareable.",
        )

    def show_about(self):
        messagebox.showinfo("About",
                            f"ML Workbench {core.VERSION}\n\n"
                            "A no-code workbench built to close the gaps in GUI ML tools:\n"
                            "• preprocessing saved inside the model (no train/predict drift)\n"
                            "• a baseline always on the table\n"
                            "• an honest holdout, group and time-series CV\n"
                            "• explicit hyperparameter search, written to the report\n"
                            "• permutation importance and linear coefficients\n"
                            "• schema, extrapolation and applicability-domain checks\n"
                            "• run.json + report.md for every experiment\n\n"
                            f"Python {sys.version.split()[0]} · scikit-learn {core.environment().get('scikit-learn')}\n"
                            "Everything runs offline; no data leaves this machine.")

    # ------------------------------------------------------------------ run
    def after_load_hook(self):  # pragma: no cover - used by the headless smoke test
        pass


def main():  # pragma: no cover
    app = App()
    if len(sys.argv) > 1 and Path(sys.argv[1]).suffix.lower() == ".csv":
        try:
            df, meta = core.load_csv(sys.argv[1])
            app.df, app.meta, app.data_path = df, meta, sys.argv[1]
            app.lbl_file.configure(text=sys.argv[1])
            app._populate_preview()
            app._populate_roles()
            app.header_status.configure(text=f"{Path(sys.argv[1]).name} · {len(df)} rows")
        except Exception:
            pass
    app.mainloop()


if __name__ == "__main__":  # pragma: no cover
    main()
