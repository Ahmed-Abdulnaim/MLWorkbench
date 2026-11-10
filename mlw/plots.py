"""mlw.plots — every figure the app writes, using a headless matplotlib backend."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

TEAL = "#0f8a7a"
GREY = "#9aa7ae"
RED = "#b3261e"

plt.rcParams.update(
    {
        "figure.dpi": 130,
        "savefig.dpi": 130,
        "font.size": 9,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titleweight": "bold",
        "axes.titlesize": 10,
        "figure.facecolor": "white",
    }
)


def _save(fig, out_dir: Path, name: str) -> str:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.png"
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def pred_vs_actual_plot(pipe, X: pd.DataFrame, y: pd.Series, out_dir: Path, task: str) -> str:
    """Scatter (regression) or confusion heatmap (classification) of the fitted model."""
    pred = np.ravel(pipe.predict(X))
    fig, ax = plt.subplots(figsize=(4.4, 4.0))
    if task == "classification":
        labels = sorted(pd.Series(y).unique().tolist())
        cm = pd.crosstab(pd.Series(y, name="actual"), pd.Series(pred, name="predicted"))
        cm = cm.reindex(index=labels, columns=labels, fill_value=0)
        ax.imshow(cm.to_numpy(), cmap="Blues")
        ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
        ax.set_yticks(range(len(labels)), labels)
        ax.set_xlabel("predicted")
        ax.set_ylabel("actual")
        for i in range(len(labels)):
            for j in range(len(labels)):
                v = int(cm.iloc[i, j])
                if v:
                    ax.text(j, i, v, ha="center", va="center",
                            color="white" if v > cm.to_numpy().max() / 2 else "#1d2429")
        ax.set_title("Predicted vs actual (all data)")
        ax.grid(False)
        return _save(fig, out_dir, "predicted_vs_actual")

    ax.scatter(y, pred, s=16, alpha=0.65, color=TEAL, edgecolor="none")
    lo, hi = float(min(y.min(), pred.min())), float(max(y.max(), pred.max()))
    ax.plot([lo, hi], [lo, hi], "--", color=GREY, lw=1.2, label="perfect")
    ax.set_xlabel("actual")
    ax.set_ylabel("predicted")
    ax.set_title("Predicted vs actual")
    ax.legend(frameon=False, loc="upper left")
    return _save(fig, out_dir, "predicted_vs_actual")


def residual_plot(pipe, X: pd.DataFrame, y: pd.Series, out_dir: Path) -> str:
    pred = np.ravel(pipe.predict(X))
    resid = np.asarray(y, dtype=float) - pred
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.4))
    axes[0].scatter(pred, resid, s=14, alpha=0.65, color=TEAL, edgecolor="none")
    axes[0].axhline(0, ls="--", color=GREY, lw=1.2)
    axes[0].set_xlabel("predicted")
    axes[0].set_ylabel("residual (actual − predicted)")
    axes[0].set_title("Residuals vs prediction")
    axes[1].hist(resid, bins=min(30, max(8, len(resid) // 6)), color=TEAL, alpha=0.85)
    axes[1].set_xlabel("residual")
    axes[1].set_title("Residual distribution")
    return _save(fig, out_dir, "residuals")


def confusion_matrix_plot(pipe, X: pd.DataFrame, y: pd.Series, out_dir: Path) -> str:
    from sklearn.metrics import confusion_matrix

    pred = pipe.predict(X)
    labels = sorted(pd.Series(y).unique().tolist())
    cm = confusion_matrix(y, pred, labels=labels)
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)), [str(l) for l in labels], rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), [str(l) for l in labels])
    ax.set_xlabel("predicted")
    ax.set_ylabel("actual")
    ax.set_title("Confusion matrix (all data)")
    ax.grid(False)
    cmax = cm.max() if cm.size else 1
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if cm[i, j]:
                ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                        color="white" if cm[i, j] > cmax / 2 else "#1d2429", fontsize=9)
    return _save(fig, out_dir, "confusion_matrix")


def correlation_heatmap(X: pd.DataFrame, out_dir: Path, max_cols: int = 20) -> str:
    num = X.select_dtypes(include=[np.number])
    if num.shape[1] < 2:
        raise ValueError("not enough numeric columns for a correlation heatmap")
    if num.shape[1] > max_cols:
        variances = num.var().sort_values(ascending=False)
        num = num[variances.index[:max_cols]]
    corr = num.corr()
    fig, ax = plt.subplots(figsize=(min(8.6, 0.42 * num.shape[1] + 2.2),
                                    min(8.0, 0.42 * num.shape[1] + 1.8)))
    im = ax.imshow(corr.to_numpy(), cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr)), corr.columns, rotation=90, fontsize=7)
    ax.set_yticks(range(len(corr)), corr.columns, fontsize=7)
    ax.grid(False)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02, label="Pearson r")
    ax.set_title("Feature correlation")
    return _save(fig, out_dir, "correlation")


def importance_plot(importance: pd.DataFrame, out_dir: Path, top: int = 15) -> str:
    imp = importance.head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(5.6, max(2.6, 0.28 * len(imp) + 1.2)))
    colors = [TEAL if v >= 0 else RED for v in imp["importance"]]
    ax.barh(imp["feature"], imp["importance"], color=colors, xerr=imp.get("std"))
    ax.axvline(0, color="#1d2429", lw=0.8)
    ax.set_xlabel("drop in score when shuffled")
    ax.set_title("Permutation importance")
    return _save(fig, out_dir, "importance")


def model_comparison_plot(metrics: pd.DataFrame, primary: str, out_dir: Path) -> str:
    df = metrics.dropna(subset=[primary]).copy()
    df = df.sort_values(primary)
    fig, ax = plt.subplots(figsize=(5.8, max(2.4, 0.32 * len(df) + 1.1)))
    colors = [GREY if "BASELINE" in str(m) else TEAL for m in df["model"]]
    ax.barh(df["model"], df[primary], color=colors,
            xerr=df.get(f"{primary}_std"))
    ax.set_xlabel(primary)
    ax.set_title("Model comparison (grey = baseline)")
    return _save(fig, out_dir, "model_comparison")


def optimisation_plot(grid: pd.DataFrame, target: str, out_dir: Path, top: int = 20) -> str:
    top_df = grid.head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(5.4, max(2.6, 0.3 * len(top_df) + 1.2)))
    ax.barh([f"#{i+1}" for i in range(len(top_df))], top_df[f"predicted_{target}"], color=TEAL)
    ax.set_xlabel(f"predicted {target}")
    ax.set_title(f"Top {len(top_df)} candidate conditions")
    return _save(fig, out_dir, "optimisation_top")
