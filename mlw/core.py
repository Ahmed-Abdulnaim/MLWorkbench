"""
mlw.core — the engine behind ML Workbench.

Pure pandas / scikit-learn: no GUI imports, so it can be driven from a script,
from the command line, or from the desktop app.

It exists to close the specific gaps in GUI ML tools such as EZmachinelearning:

  gap in the old tool                     fix here
  --------------------------------------  ------------------------------------------
  one-hot encoder never saved             every model is a full sklearn Pipeline
                                          (impute + encode + scale + estimator) that
                                          is pickled as one object -> train/predict
                                          can never drift apart
  no missing-value handling               median / most-frequent imputation inside
                                          the pipeline, reported up front
  no baseline to compare against          DummyRegressor / DummyClassifier are always
                                          trained and shown
  no honest holdout                       optional stratified holdout scored
                                          separately from the CV score
  no time-series awareness                TimeSeriesSplit + group-aware CV + lag/
                                          rolling feature builder
  hyperparameters searched invisibly      explicit RandomizedSearchCV with the
                                          chosen parameters written to the report
  no interpretability                     permutation importance + coefficients
  no run records                          every run writes run.json, report.md,
                                          metrics.csv, plots/ and a data hash
  no leakage / schema checks              an audit pass plus strict prediction-time
                                          schema validation
  no applicability domain                 range checks + unseen-category counts and,
                                          for molecules, Tanimoto similarity
  awkward step/array grid file            a range-based optimiser with boundary and
                                          extrapolation warnings
"""

from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import itertools
import json
import math
import platform
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    GroupKFold,
    KFold,
    RandomizedSearchCV,
    StratifiedKFold,
    TimeSeriesSplit,
    cross_validate,
    train_test_split,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC, SVR, LinearSVC, LinearSVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
)

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

VERSION = "1.0.0"

# In a frozen (PyInstaller) build, sys.executable is the exe itself, so joblib's
# process-based workers would re-launch the app with a code string as argv and die.
# Threads need no respawn and are fast enough for tabular work.
FROZEN = bool(getattr(sys, "frozen", False))
PARALLEL_BACKEND = "threading" if FROZEN else None


def _parallel_ctx():
    if PARALLEL_BACKEND:
        return joblib.parallel_backend(PARALLEL_BACKEND)
    return contextlib.nullcontext()

try:  # optional, only used if installed
    from xgboost import XGBClassifier, XGBRegressor

    HAS_XGB = True
except Exception:  # pragma: no cover
    HAS_XGB = False


# =============================================================================
# 1.  Loading & auditing data
# =============================================================================


@dataclass
class Issue:
    level: str  # "error" | "warning" | "info"
    code: str
    message: str
    hint: str = ""

    def as_dict(self) -> dict:
        return dataclasses.asdict(self)


ENCODINGS = ("utf-8-sig", "utf-8", "shift_jis", "cp1252", "latin-1")


def load_csv(path: str | Path) -> tuple[pd.DataFrame, dict]:
    """Read a CSV trying the encodings that actually occur in engineering data."""
    path = Path(path)
    last: Exception | None = None
    for enc in ENCODINGS:
        for sep in (",", ";", "\t"):
            try:
                df = pd.read_csv(path, encoding=enc, sep=sep)
            except Exception as exc:  # noqa: BLE001
                last = exc
                continue
            if df.shape[1] > 1:
                df.columns = [str(c).strip().lstrip("\ufeff") for c in df.columns]
                return df, {"encoding": enc, "sep": sep, "rows": len(df), "cols": df.shape[1]}
    raise ValueError(f"Could not read {path.name} as CSV ({last})")


def dataset_hash(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def infer_task(series: pd.Series) -> str:
    """Decide regression vs classification from the target column itself."""
    s = series.dropna()
    if not pd.api.types.is_numeric_dtype(s):
        return "classification"
    nunique = s.nunique()
    if nunique <= 2:
        return "classification"
    # integers with few distinct values that look like labels / codes
    if pd.api.types.is_integer_dtype(s) and nunique <= max(10, int(0.02 * len(s))) and nunique < 20:
        return "classification"
    return "regression"


def _looks_like_id(series: pd.Series) -> bool:
    """Identifiers are unique-per-row *and* either textual or a monotonic integer key.
    Continuous measurements (a target, a temperature) are not identifiers."""
    n = len(series.dropna())
    if n < 8:
        return False
    uniq = series.nunique(dropna=True)
    ratio = uniq / max(n, 1)
    if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
        return ratio > 0.95
    if pd.api.types.is_integer_dtype(series):
        vals = series.dropna()
        monotonic = bool(vals.is_monotonic_increasing or vals.is_monotonic_decreasing)
        return ratio > 0.98 and monotonic
    return False


def audit(df: pd.DataFrame, target: str | None = None) -> list[Issue]:
    """Everything the old app silently tolerated, surfaced as a list of findings."""
    issues: list[Issue] = []
    n_rows, n_cols = df.shape

    if n_rows < 50:
        issues.append(
            Issue(
                "warning",
                "small-n",
                f"Only {n_rows} rows. Cross-validation scores will have a wide spread.",
                "Treat differences under ~0.05 as ties and prefer the simpler model.",
            )
        )
    if n_cols < 2:
        issues.append(Issue("error", "one-column", "Fewer than two columns — nothing to learn from."))

    # ---- header hygiene -----------------------------------------------------
    non_ascii = [c for c in df.columns if any(ord(ch) > 127 for ch in c)]
    if non_ascii:
        issues.append(
            Issue(
                "info",
                "non-ascii-headers",
                f"{len(non_ascii)} column name(s) contain non-ASCII characters.",
                "Fine inside this app, but export with ASCII names before sharing.",
            )
        )
    dup_cols = [c for c in df.columns if list(df.columns).count(c) > 1]
    if dup_cols:
        issues.append(
            Issue("error", "duplicate-columns", f"Duplicate column names: {sorted(set(dup_cols))}", "Rename them.")
        )

    # ---- missing values -----------------------------------------------------
    na = df.isna().sum()
    na = na[na > 0]
    if len(na):
        worst = na.sort_values(ascending=False).head(5)
        detail = ", ".join(f"{c}={int(v)}" for c, v in worst.items())
        pct = 100 * na.sum() / max(n_rows * n_cols, 1)
        issues.append(
            Issue(
                "warning",
                "missing-values",
                f"Missing cells in {len(na)} column(s) ({pct:.1f}% of all cells): {detail}",
                "Will be imputed automatically (median for numbers, most frequent for text). "
                "Very sparse columns are better dropped.",
            )
        )
        heavy = na[na / n_rows > 0.4]
        if len(heavy):
            issues.append(
                Issue(
                    "error",
                    "sparse-columns",
                    f"More than 40% missing in: {', '.join(heavy.index)}",
                    "Imputing these will manufacture data. Exclude them.",
                )
            )

    # ---- duplicate rows -----------------------------------------------------
    dup = int(df.duplicated().sum())
    if dup:
        issues.append(
            Issue(
                "warning",
                "duplicate-rows",
                f"{dup} duplicate row(s) ({100 * dup / max(n_rows,1):.1f}%).",
                "Duplicates across CV folds inflate scores. Drop or justify them.",
            )
        )

    # ---- constant / near-constant ------------------------------------------
    for c in df.columns:
        s = df[c]
        if s.nunique(dropna=True) <= 1:
            issues.append(Issue("warning", "constant", f"'{c}' has a single distinct value.", "It carries no information."))

    # ---- id-like columns ----------------------------------------------------
    suspects = [c for c in df.columns if _looks_like_id(df[c]) and c != target]
    if suspects:
        issues.append(
            Issue(
                "warning",
                "id-columns",
                f"Column(s) that look like identifiers: {', '.join(suspects[:6])}",
                "Unique-per-row columns are leakage. Do not use them as features.",
            )
        )

    # ---- numeric-looking text ----------------------------------------------
    for c in df.columns:
        if df[c].dtype == object:
            s = df[c].dropna().astype(str)
            if len(s) == 0:
                continue
            converted = pd.to_numeric(s.str.replace(",", "", regex=False), errors="coerce")
            if converted.notna().mean() > 0.95:
                issues.append(
                    Issue(
                        "warning",
                        "numeric-text",
                        f"'{c}' is stored as text but looks numeric.",
                        "Convert it to numbers, or the model will treat each value as a category.",
                    )
                )

    # ---- whitespace / inconsistent labels ---------------------------------
    for c in df.columns:
        if df[c].dtype == object:
            vals = df[c].dropna().astype(str)
            if vals.empty:
                continue
            stripped = vals.str.strip()
            if (vals != stripped).any():
                issues.append(
                    Issue("warning", "whitespace-labels", f"'{c}' has leading/trailing spaces in some values.",
                          "Strip them, otherwise 'A' and 'A ' become two different classes.")
                )
            low = stripped.str.lower()
            if low.nunique() < stripped.nunique():
                issues.append(
                    Issue("warning", "case-labels", f"'{c}' has values that differ only by case.",
                          "Normalise case to avoid phantom classes.")
                )

    # ---- target specific ---------------------------------------------------
    if target and target in df.columns:
        y = df[target]
        n_missing = int(y.isna().sum())
        if n_missing:
            issues.append(
                Issue("error", "target-missing", f"Target '{target}' has {n_missing} missing value(s).",
                      "Rows without a target cannot be trained on. Use a predict file instead.")
            )
        task = infer_task(y)
        if task == "classification":
            counts = y.value_counts()
            if len(counts) <= 1:
                issues.append(Issue("error", "one-class", "Target has only one class."))
            else:
                imbalance = counts.iloc[0] / counts.sum()
                issues.append(
                    Issue(
                        "info",
                        "class-balance",
                        f"Classes: {', '.join(f'{k}={v}' for k, v in counts.items())}. "
                        f"Majority rate = {imbalance:.3f} — beat this, or accuracy means nothing.",
                        "Tick 'balance classes' in the Train tab if the minority class matters.",
                    )
                )
                rare = [str(k) for k, v in counts.items() if v < max(5, 0.02 * len(df))]
                if rare:
                    issues.append(
                        Issue("warning", "rare-class", f"Class(es) with very few rows: {', '.join(rare)}",
                              "Fewer than ~5 rows per class cannot be learned or validated reliably.")
                    )
        else:
            issues.append(
                Issue(
                    "info",
                    "target-range",
                    f"Regression target '{target}': min={y.min():.4g}, max={y.max():.4g}, "
                    f"mean={y.mean():.4g}, std={y.std():.4g}",
                )
            )

        # ---- leakage: a feature that is almost a copy of the target --------
        for c in df.columns:
            if c == target or not pd.api.types.is_numeric_dtype(df[c]):
                continue
            if not pd.api.types.is_numeric_dtype(y):
                continue
            try:
                corr = df[[c, target]].corr().iloc[0, 1]
            except Exception:
                continue
            if pd.notna(corr) and abs(corr) > 0.995:
                issues.append(
                    Issue(
                        "error",
                        "leakage",
                        f"'{c}' correlates {corr:.4f} with the target.",
                        "This is almost certainly target leakage or a duplicate measurement.",
                    )
                )

        # ---- very strong feature-feature redundancy -----------------------
        num = df.select_dtypes(include=[np.number]).drop(columns=[target], errors="ignore")
        if num.shape[1] > 1:
            corr_m = num.corr().abs()
            pairs = [
                (a, b, float(corr_m.loc[a, b]))
                for i, a in enumerate(corr_m.columns)
                for b in corr_m.columns[i + 1:]
                if pd.notna(corr_m.loc[a, b]) and corr_m.loc[a, b] > 0.98
            ]
            if pairs:
                shown = ", ".join(f"{a}/{b} ({v:.2f})" for a, b, v in pairs[:4])
                issues.append(
                    Issue("info", "redundant-features", f"Near-duplicate features: {shown}",
                          "Harmless for trees, destabilising for linear models. Consider keeping one of each pair.")
                )

    # ---- infinities --------------------------------------------------------
    num_all = df.select_dtypes(include=[np.number])
    inf_count = int(np.isinf(num_all.to_numpy()).sum()) if num_all.shape[1] else 0
    if inf_count:
        issues.append(Issue("error", "infinite-values", f"{inf_count} infinite value(s).", "Replace with NaN or fix the source."))

    return issues


# =============================================================================
# 2.  Model zoo
# =============================================================================

# (factory, param_distributions, complexity, needs_scaling, supports_class_weight)
ModelSpec = tuple[Callable[[], Any], dict, int, bool, bool]


def _zoo() -> dict[str, dict[str, ModelSpec]]:
    reg: dict[str, ModelSpec] = {
        "Linear": (LinearRegression, {}, 0, False, False),
        "Ridge": (Ridge, {"alpha": [0.01, 0.1, 1, 3, 10, 30, 100, 300]}, 1, True, False),
        "Lasso": (Lasso, {"alpha": [0.001, 0.01, 0.1, 1, 10]}, 1, True, False),
        "ElasticNet": (ElasticNet, {"alpha": [0.001, 0.01, 0.1, 1], "l1_ratio": [0.2, 0.5, 0.8]}, 2, True, False),
        "PLS": (lambda: PLSRegression(scale=False), {"n_components": [1, 2, 3, 4, 5, 8]}, 2, True, False),
        "LinearSupportVector": (LinearSVR, {"C": [0.01, 0.1, 1, 10], "epsilon": [0, 0.1, 0.5]}, 2, True, False),
        "SupportVector": (SVR, {"C": [0.1, 1, 10, 100], "gamma": ["scale", 0.01, 0.1], "epsilon": [0.05, 0.1, 0.5]}, 3, True, False),
        "KNeighbors": (KNeighborsRegressor, {"n_neighbors": [3, 5, 7, 11, 15], "weights": ["uniform", "distance"]}, 2, True, False),
        "DecisionTree": (DecisionTreeRegressor, {"max_depth": [2, 3, 5, 8, None], "min_samples_leaf": [1, 2, 5, 10]}, 3, False, False),
        "RandomForest": (RandomForestRegressor, {"n_estimators": [200, 400, 800], "max_depth": [None, 3, 5, 8],
                                                 "min_samples_leaf": [1, 2, 4], "max_features": ["sqrt", 1.0]}, 4, False, False),
        "ExtraTrees": (ExtraTreesRegressor, {"n_estimators": [200, 400], "max_depth": [None, 5, 10],
                                             "min_samples_leaf": [1, 2, 4]}, 4, False, False),
        "GradientBoosting": (GradientBoostingRegressor, {"n_estimators": [100, 200, 400], "learning_rate": [0.02, 0.05, 0.1],
                                                        "max_depth": [2, 3, 4], "subsample": [0.7, 1.0]}, 5, False, False),
        "HistGradientBoosting": (HistGradientBoostingRegressor, {"max_iter": [200, 400], "learning_rate": [0.03, 0.06, 0.1],
                                                                "max_leaf_nodes": [15, 31, 63]}, 5, False, False),
        "MLP": (lambda: MLPRegressor(max_iter=1500, early_stopping=True, random_state=0),
                {"hidden_layer_sizes": [(32,), (64,), (64, 32)], "alpha": [1e-4, 1e-2, 0.1]}, 5, True, False),
    }
    clf: dict[str, ModelSpec] = {
        "Logistic": (lambda: LogisticRegression(max_iter=3000), {"C": [0.01, 0.1, 1, 10, 100]}, 1, True, True),
        "LinearDiscriminantAnalysis": (LinearDiscriminantAnalysis, {"solver": ["svd", "lsqr"]}, 1, True, False),
        "GaussianNaiveBayes": (GaussianNB, {"var_smoothing": [1e-9, 1e-8, 1e-7]}, 1, False, False),
        "KNeighbors": (KNeighborsClassifier, {"n_neighbors": [3, 5, 7, 11], "weights": ["uniform", "distance"]}, 2, True, False),
        "SupportVector": (lambda: SVC(probability=True), {"C": [0.1, 1, 10, 100], "gamma": ["scale", 0.01, 0.1], "kernel": ["rbf", "linear"]}, 3, True, True),
        "LinearSupportVector": (lambda: LinearSVC(max_iter=5000), {"C": [0.01, 0.1, 1, 10]}, 2, True, True),
        "DecisionTree": (DecisionTreeClassifier, {"max_depth": [2, 3, 5, 8, None], "criterion": ["gini", "entropy"],
                                                 "min_samples_leaf": [1, 2, 5, 10]}, 3, False, True),
        "RandomForest": (RandomForestClassifier, {"n_estimators": [200, 400, 800], "max_depth": [None, 3, 5, 8],
                                                 "min_samples_leaf": [1, 2, 4], "max_features": ["sqrt", None]}, 4, False, True),
        "ExtraTrees": (ExtraTreesClassifier, {"n_estimators": [200, 400], "max_depth": [None, 5, 10],
                                             "min_samples_leaf": [1, 2, 4]}, 4, False, True),
        "GradientBoosting": (GradientBoostingClassifier, {"n_estimators": [100, 200, 400], "learning_rate": [0.02, 0.05, 0.1],
                                                         "max_depth": [2, 3, 4], "subsample": [0.7, 1.0]}, 5, False, False),
        "HistGradientBoosting": (HistGradientBoostingClassifier, {"max_iter": [200, 400], "learning_rate": [0.03, 0.06, 0.1],
                                                                 "max_leaf_nodes": [15, 31, 63]}, 5, False, False),
        "QuadraticDiscriminantAnalysis": (QuadraticDiscriminantAnalysis, {"reg_param": [0.0, 0.1, 0.3]}, 2, False, False),
        "MultilayerPerceptron": (lambda: MLPClassifier(max_iter=2000, early_stopping=True, random_state=0),
                                 {"hidden_layer_sizes": [(32,), (64,), (64, 32)], "alpha": [1e-4, 1e-2, 0.1]}, 5, True, False),
        "XGBoost": (lambda: XGBClassifier(eval_metric="logloss", verbosity=0, random_state=0),
                    {"n_estimators": [200, 400], "learning_rate": [0.03, 0.1], "max_depth": [3, 4, 6],
                     "subsample": [0.7, 1.0]}, 6, False, False),
    }
    reg["XGBoost"] = (lambda: XGBRegressor(verbosity=0, random_state=0),
                      {"n_estimators": [200, 400], "learning_rate": [0.03, 0.1], "max_depth": [3, 4, 6],
                       "subsample": [0.7, 1.0]}, 6, False, False)
    if not HAS_XGB:
        reg.pop("XGBoost")
        clf.pop("XGBoost")
    return {"regression": reg, "classification": clf}


ZOO = _zoo()

AUTO_MODELS = {
    "regression": ["Linear", "Ridge", "PLS", "RandomForest", "GradientBoosting", "HistGradientBoosting"],
    "classification": ["Logistic", "LinearDiscriminantAnalysis", "RandomForest", "GradientBoosting", "HistGradientBoosting"],
}

PRIMARY_METRIC = {"regression": "r2", "classification": "f1_macro"}

SCORERS = {
    "regression": ["r2", "neg_mean_absolute_error", "neg_root_mean_squared_error"],
    "classification": ["accuracy", "balanced_accuracy", "f1_macro", "f1_weighted"],
}


def available_models(task: str) -> list[str]:
    return list(ZOO[task].keys())


# =============================================================================
# 3.  Training
# =============================================================================


@dataclass
class TrainConfig:
    target: str
    features: list[str]
    task: str = "regression"
    folds: int = 5
    cv_strategy: str = "kfold"  # kfold | group | timeseries
    group_col: str | None = None
    holdout_frac: float = 0.0
    seed: int = 77
    search_iters: int = 0
    models: list[str] | None = None
    balance: bool = False
    n_jobs: int = -1
    save_all_models: bool = False

    def as_dict(self) -> dict:
        d = dataclasses.asdict(self)
        d["models"] = self.models or []
        return d


def _preprocessor(numeric: list[str], categorical: list[str], scale: bool) -> ColumnTransformer:
    num_steps: list[tuple[str, Any]] = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        num_steps.append(("scale", StandardScaler()))
    transformers: list[tuple[str, Any, list[str]]] = []
    if numeric:
        transformers.append(("num", Pipeline(num_steps), numeric))
    if categorical:
        cat_steps: list[tuple[str, Any]] = [
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
        transformers.append(("cat", Pipeline(cat_steps), categorical))
    return ColumnTransformer(transformers, remainder="drop", sparse_threshold=0)


def _build_estimator(task: str, name: str, balance: bool) -> tuple[Any, dict, int]:
    factory, dist, complexity, needs_scale, supports_cw = ZOO[task][name]
    est = factory()
    if balance and supports_cw:
        try:
            est.set_params(class_weight="balanced")
        except Exception:
            pass
    params = {f"model__{k}": v for k, v in dist.items()}
    return est, params, complexity


def build_pipeline(task: str, name: str, numeric: list[str], categorical: list[str], balance: bool = False) -> tuple[Pipeline, dict, int]:
    est, params, complexity = _build_estimator(task, name, balance)
    _, _, _, needs_scale, _ = ZOO[task][name]
    pipe = Pipeline([("prep", _preprocessor(numeric, categorical, needs_scale)), ("model", est)])
    return pipe, params, complexity


def _make_cv(task: str, cfg: TrainConfig, n_rows: int, groups: pd.Series | None):
    folds = max(2, min(cfg.folds, n_rows // 2 if task == "classification" else max(2, n_rows - 1)))
    if cfg.cv_strategy == "group" and groups is not None:
        return GroupKFold(n_splits=min(folds, groups.nunique())), groups
    if cfg.cv_strategy == "timeseries":
        return TimeSeriesSplit(n_splits=folds), None
    if task == "classification":
        return StratifiedKFold(n_splits=folds, shuffle=True, random_state=cfg.seed), None
    return KFold(n_splits=folds, shuffle=True, random_state=cfg.seed), None


def _clean_scores(raw: dict[str, np.ndarray]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for key, vals in raw.items():
        if not key.startswith("test_"):
            continue
        name = key[5:]
        if name.startswith("neg_"):
            name = name[4:]
            vals = -vals
        out[name] = {"mean": float(np.nanmean(vals)), "std": float(np.nanstd(vals)),
                     "folds": [float(v) for v in vals]}
    return out


def _coefficients(pipe: Pipeline, feature_names: list[str]) -> list[dict] | None:
    model = pipe.named_steps["model"]
    coefs = getattr(model, "coef_", None)
    if coefs is None:
        return None
    coefs = np.ravel(coefs)
    names = feature_names
    if len(names) != len(coefs):
        try:
            names = list(pipe.named_steps["prep"].get_feature_names_out())
        except Exception:
            return None
    pairs = sorted(zip(names, coefs.tolist()), key=lambda t: -abs(t[1]))
    return [{"feature": n, "coefficient": float(c)} for n, c in pairs[:40]]


def permutation_importance_manual(
    pipe: Pipeline, X: pd.DataFrame, y: pd.Series, task: str, seed: int = 0, n_repeats: int = 3
) -> pd.DataFrame:
    """Own implementation: shuffles each raw column, so it works with named columns
    inside the ColumnTransformer (and needs no extra dependency)."""
    rng = np.random.default_rng(seed)
    base_score = _score_once(pipe, X, y, task)
    rows = []
    for col in X.columns:
        drops = []
        for _ in range(n_repeats):
            Xp = X.copy()
            Xp[col] = Xp[col].sample(frac=1.0, random_state=int(rng.integers(1 << 30))).to_numpy()
            drops.append(base_score - _score_once(pipe, Xp, y, task))
        rows.append({"feature": col, "importance": float(np.mean(drops)), "std": float(np.std(drops))})
    df = pd.DataFrame(rows).sort_values("importance", ascending=False).reset_index(drop=True)
    return df


def _score_once(pipe: Pipeline, X: pd.DataFrame, y: pd.Series, task: str) -> float:
    from sklearn.metrics import f1_score as _f1

    pred = pipe.predict(X)
    if task == "regression":
        return float(r2_score(y, pred))
    return float(_f1(y, pred, average="macro"))


def _classification_extras(pipe: Pipeline, X: pd.DataFrame, y: pd.Series) -> dict:
    pred = pipe.predict(X)
    labels = sorted(pd.Series(y).unique().tolist())
    metrics = {
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "f1_macro": float(f1_score(y, pred, average="macro")),
        "f1_weighted": float(f1_score(y, pred, average="weighted")),
        "precision_macro": float(precision_score(y, pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y, pred, average="macro", zero_division=0)),
    }
    try:
        proba = pipe.predict_proba(X)
        if len(labels) == 2:
            metrics["roc_auc"] = float(roc_auc_score(y, proba[:, 1]))
        elif len(labels) > 2:
            metrics["roc_auc_ovr"] = float(roc_auc_score(y, proba, multi_class="ovr", labels=labels))
    except Exception:
        pass
    cm = confusion_matrix(y, pred, labels=labels)
    per_class = {}
    for lab in labels:
        mask = np.asarray(y) == lab
        per_class[str(lab)] = {
            "support": int(mask.sum()),
            "precision": float(precision_score(mask, np.asarray(pred) == lab, zero_division=0)),
            "recall": float(recall_score(mask, np.asarray(pred) == lab, zero_division=0)),
            "f1": float(f1_score(mask, np.asarray(pred) == lab, zero_division=0)),
        }
    return {"labels": [str(l) for l in labels], "confusion_matrix": cm.tolist(),
            "per_class": per_class, "metrics": metrics}


def _regression_extras(pipe: Pipeline, X: pd.DataFrame, y: pd.Series) -> dict:
    pred = pipe.predict(X)
    pred = np.ravel(pred)
    yy = np.asarray(y, dtype=float)
    mae = float(mean_absolute_error(yy, pred))
    rmse = float(math.sqrt(mean_squared_error(yy, pred)))
    extras = {"mae": mae, "rmse": rmse, "r2": float(r2_score(yy, pred))}
    if np.all(yy > 0):
        extras["mape"] = float(np.mean(np.abs((yy - pred) / yy)) * 100)
    residuals = yy - pred
    return {
        "metrics": extras,
        "residual_std": float(np.std(residuals)),
        "worst_index": int(np.argmax(np.abs(residuals))),
    }


def _json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.ndarray,)):
        return _json_safe(obj.tolist())
    if isinstance(obj, (pd.Timestamp,)):
        return obj.isoformat()
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    if isinstance(obj, (Path,)):
        return str(obj)
    return obj


def model_bundle(task: str, target: str, features: list[str], numeric: list[str], categorical: list[str],
                 pipe: Pipeline, df: pd.DataFrame, cfg: TrainConfig, metrics: dict, extras: dict) -> dict:
    """One object that contains preprocessing *and* estimator: no train/predict drift."""
    categories = {c: sorted(map(str, df[c].dropna().unique().tolist())) for c in categorical}
    ranges = {
        c: [float(pd.to_numeric(df[c], errors="coerce").min()), float(pd.to_numeric(df[c], errors="coerce").max())]
        for c in numeric
    }
    return {
        "kind": "mlw.model",
        "format_version": 1,
        "app_version": VERSION,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "task": task,
        "target": target,
        "features": features,
        "numeric": numeric,
        "categorical": categorical,
        "categories": categories,
        "ranges": ranges,
        "classes": [str(c) for c in getattr(pipe.named_steps["model"], "classes_", [])] if task == "classification" else [],
        "metrics": metrics,
        "extras": {k: v for k, v in extras.items() if k in ("labels", "per_class", "mae", "rmse", "r2", "mape", "residual_std")},
        "config": cfg.as_dict(),
        "pipeline": pipe,
    }


def train_and_evaluate(
    df: pd.DataFrame,
    cfg: TrainConfig,
    out_root: str | Path = "runs",
    name: str = "run",
    log: Callable[[str], None] | None = None,
    progress: Callable[[float, str], None] | None = None,
    source_path: str | None = None,
) -> dict:
    """Train every requested model, evaluate honestly, pick one, and write a run folder."""
    log = log or (lambda m: None)
    progress = progress or (lambda p, m: None)

    if cfg.target not in df.columns:
        raise ValueError(f"Target '{cfg.target}' is not a column of the data.")
    missing = [f for f in cfg.features if f not in df.columns]
    if missing:
        raise ValueError(f"Feature(s) not found: {missing}")
    if cfg.target in cfg.features:
        raise ValueError("The target cannot also be a feature — that is leakage by construction.")

    df = df.dropna(subset=[cfg.target]).reset_index(drop=True)
    X = df[cfg.features].copy()
    y = df[cfg.target].copy()

    numeric = [c for c in cfg.features if pd.api.types.is_numeric_dtype(X[c])]
    categorical = [c for c in cfg.features if c not in numeric]
    # numbers stored as text -> coerce, and report it
    for c in numeric:
        X[c] = pd.to_numeric(X[c], errors="coerce")

    models = cfg.models or AUTO_MODELS[cfg.task]
    models = [m for m in models if m in ZOO[cfg.task]]
    if not models:
        raise ValueError("No valid models selected for this task.")

    run_dir = Path(out_root) / f"{time.strftime('%Y%m%d_%H%M%S')}_{_slug(name)}"
    (run_dir / "models").mkdir(parents=True, exist_ok=True)
    (run_dir / "plots").mkdir(parents=True, exist_ok=True)

    log(f"Data: {len(df)} rows, {len(cfg.features)} features "
        f"({len(numeric)} numeric, {len(categorical)} categorical), target '{cfg.target}' [{cfg.task}]")
    log(f"Models: {', '.join(models)}")
    log(f"CV: {cfg.cv_strategy}, {cfg.folds} folds, seed {cfg.seed}"
        + (f", holdout {cfg.holdout_frac:.0%}" if cfg.holdout_frac else ""))

    # ---------------- holdout split ----------------------------------------
    test_df = None
    if cfg.holdout_frac and cfg.holdout_frac > 0:
        strat = y if cfg.task == "classification" else None
        try:
            X_tr, X_te, y_tr, y_te = train_test_split(
                X, y, test_size=cfg.holdout_frac, random_state=cfg.seed, stratify=strat
            )
        except ValueError:
            X_tr, X_te, y_tr, y_te = train_test_split(
                X, y, test_size=cfg.holdout_frac, random_state=cfg.seed
            )
        df_train = df.loc[X_tr.index]
        test_df = (X_te, y_te)
        log(f"Holdout: {len(X_tr)} train / {len(X_te)} test rows (test rows never used for tuning)")
    else:
        X_tr, y_tr = X, y
        df_train = df

    groups = df_train[cfg.group_col] if (cfg.cv_strategy == "group" and cfg.group_col in df.columns) else None
    cv, cv_groups = _make_cv(cfg.task, cfg, len(X_tr), groups)

    results: list[dict] = []
    primary = PRIMARY_METRIC[cfg.task]

    # ---------------- baseline --------------------------------------------
    progress(0.02, "baseline")
    baseline_model = DummyRegressor(strategy="mean") if cfg.task == "regression" else DummyClassifier(strategy="most_frequent")
    base_pipe = Pipeline([("prep", _preprocessor(numeric, categorical, False)), ("model", baseline_model)])
    with _parallel_ctx():
        base_scores = _clean_scores(cross_validate(base_pipe, X_tr, y_tr, cv=cv, groups=cv_groups,
                                                  scoring=SCORERS[cfg.task], n_jobs=cfg.n_jobs))
    baseline = {
        "model": "BASELINE (mean)" if cfg.task == "regression" else "BASELINE (majority class)",
        "complexity": -1,
        "cv": base_scores,
        "holdout": None,
        "params": {},
        "notes": "Always compare against this.",
    }
    if test_df is not None:
        base_pipe.fit(X_tr, y_tr)
        baseline["holdout"] = (
            _regression_extras(base_pipe, *test_df)["metrics"] if cfg.task == "regression"
            else _classification_extras(base_pipe, *test_df)["metrics"]
        )
    log(f"Baseline {primary}: {base_scores[primary]['mean']:.4f}")

    # ---------------- per model -------------------------------------------
    fitted: dict[str, Any] = {}
    n_models = len(models)
    for i, name_m in enumerate(models):
        t0 = time.time()
        progress(0.05 + 0.85 * i / n_models, f"training {name_m}")
        log(f"→ {name_m}")
        pipe, params, complexity = build_pipeline(cfg.task, name_m, numeric, categorical, cfg.balance)
        best_params: dict = {}
        try:
            if cfg.search_iters and params:
                search = RandomizedSearchCV(
                    pipe, params, n_iter=min(cfg.search_iters, _space_size(params)),
                    cv=_inner_cv(cfg.task, cfg.seed), scoring=primary, random_state=cfg.seed,
                    n_jobs=cfg.n_jobs, refit=True,
                )
                with _parallel_ctx():
                    search.fit(X_tr, y_tr)
                pipe = search.best_estimator_
                best_params = {k.replace("model__", ""): v for k, v in search.best_params_.items()}
                log(f"   tuned: {best_params}")
            with _parallel_ctx():
                cv_scores = _clean_scores(cross_validate(pipe, X_tr, y_tr, cv=cv, groups=cv_groups,
                                                         scoring=SCORERS[cfg.task], n_jobs=cfg.n_jobs))
            pipe.fit(X_tr, y_tr)
        except Exception as exc:  # noqa: BLE001
            log(f"   FAILED: {exc}")
            results.append({"model": name_m, "complexity": complexity, "cv": {}, "holdout": None,
                            "params": {}, "error": str(exc), "fit_seconds": time.time() - t0,
                            "n_params": 0})
            continue

        extras = (
            _regression_extras(pipe, *test_df)["metrics"] if (cfg.task == "regression" and test_df is not None)
            else _classification_extras(pipe, *test_df)["metrics"] if (cfg.task == "classification" and test_df is not None)
            else None
        )
        row = {
            "model": name_m,
            "complexity": complexity,
            "cv": cv_scores,
            "holdout": extras,
            "params": best_params or _default_params(pipe),
            "fit_seconds": round(time.time() - t0, 2),
            "n_params": int(_n_params(pipe)),
        }
        results.append(row)
        fitted[name_m] = pipe
        score = cv_scores.get(primary, {}).get("mean", float("nan"))
        log(f"   {primary} = {score:.4f} (+/- {cv_scores.get(primary, {}).get('std', float('nan')):.4f}) "
            f"in {row['fit_seconds']}s")

    progress(0.92, "selecting model")
    ordered = sorted(
        [r for r in results if r["cv"] and primary in r["cv"]],
        key=lambda r: -r["cv"][primary]["mean"],
    )
    if not ordered:
        raise RuntimeError("Every model failed. See the log for the reason.")

    best = ordered[0]
    chosen = _choose_with_one_se(ordered, primary)
    baseline_mean = base_scores[primary]["mean"]
    gain = chosen["cv"][primary]["mean"] - baseline_mean
    recommendation = _recommendation(chosen, best, baseline_mean, primary)

    log(f"Best score: {best['model']} ({best['cv'][primary]['mean']:.4f})")
    log(f"Recommended: {chosen['model']} ({chosen['cv'][primary]['mean']:.4f})")
    log(f"Gain over baseline: {gain:+.4f}")

    # ---------------- final fit on all data + artefacts --------------------
    progress(0.94, "saving model")
    final_pipe = fit_final(cfg.task, chosen["model"], chosen["params"], numeric, categorical, cfg, df[cfg.features], df[cfg.target])

    bundle = model_bundle(cfg.task, cfg.target, cfg.features, numeric, categorical,
                          final_pipe, df, cfg, chosen["cv"], {})
    chosen_path = run_dir / "models" / f"{_slug(chosen['model'])}.pkl"
    joblib.dump(bundle, chosen_path)

    saved_models = [str(chosen_path)]
    if cfg.save_all_models:
        for name_m, pipe in fitted.items():
            if name_m == chosen["model"]:
                continue
            b = model_bundle(cfg.task, cfg.target, cfg.features, numeric, categorical,
                             pipe, df, cfg, next(r["cv"] for r in results if r["model"] == name_m), {})
            p = run_dir / "models" / f"{_slug(name_m)}.pkl"
            joblib.dump(b, p)
            saved_models.append(str(p))

    # importance on the final model
    progress(0.96, "importance")
    importance_rows: list[dict] = []
    try:
        imp = permutation_importance_manual(final_pipe, df[cfg.features], df[cfg.target], cfg.task, seed=cfg.seed)
        importance_rows = imp.to_dict(orient="records")
        imp.to_csv(run_dir / "feature_importance.csv", index=False)
        log("Feature importance: " + ", ".join(f"{r['feature']}={r['importance']:+.3f}" for r in importance_rows[:4]))
    except Exception as exc:  # noqa: BLE001
        log(f"importance failed: {exc}")

    coefs = _coefficients(final_pipe, cfg.features)

    # metrics table
    metrics_df = _metrics_table(baseline, results, primary)
    metrics_df.to_csv(run_dir / "metrics.csv", index=False)

    # holdout predictions
    if test_df is not None:
        X_te, y_te = test_df
        pred_df = X_te.copy()
        pred_df["actual"] = y_te.to_numpy()
        pred_df["prediction"] = np.ravel(final_pipe.predict(X_te))
        pred_df.to_csv(run_dir / "holdout_predictions.csv", index=False)

    # plots
    plots: dict[str, str] = {}
    try:
        from . import plots as P

        plots["confusion_matrix"] = P.confusion_matrix_plot(final_pipe, df[cfg.features], df[cfg.target], run_dir / "plots") if cfg.task == "classification" else ""
        plots["predicted_vs_actual"] = P.pred_vs_actual_plot(final_pipe, df[cfg.features], df[cfg.target], run_dir / "plots", cfg.task)
        if cfg.task == "regression":
            plots["residuals"] = P.residual_plot(final_pipe, df[cfg.features], df[cfg.target], run_dir / "plots")
        plots["correlation"] = P.correlation_heatmap(df[cfg.features], run_dir / "plots")
        if importance_rows:
            plots["importance"] = P.importance_plot(pd.DataFrame(importance_rows), run_dir / "plots")
        plots = {k: str(v) for k, v in plots.items() if v}
    except Exception as exc:  # noqa: BLE001
        log(f"plots failed: {exc}")

    # run record + report
    run_record = {
        "app": "ML Workbench",
        "app_version": VERSION,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "source_file": str(source_path) if source_path else None,
        "source_hash": dataset_hash(source_path) if source_path else None,
        "rows": int(len(df)),
        "features": cfg.features,
        "numeric": numeric,
        "categorical": categorical,
        "target": cfg.target,
        "task": cfg.task,
        "config": cfg.as_dict(),
        "baseline": baseline,
        "results": results,
        "chosen_model": chosen["model"],
        "best_model": best["model"],
        "recommendation": recommendation,
        "importance": importance_rows,
        "coefficients": coefs,
        "plots": plots,
        "models_saved": saved_models,
        "environment": environment(),
    }
    (run_dir / "run.json").write_text(json.dumps(_json_safe(run_record), indent=2), encoding="utf-8")
    (run_dir / "report.md").write_text(_report_md(run_record, baseline, results, primary), encoding="utf-8")

    progress(1.0, "done")
    log(f"Run written to {run_dir}")

    return {
        "run_dir": str(run_dir),
        "task": cfg.task,
        "target": cfg.target,
        "features": cfg.features,
        "baseline": baseline,
        "results": results,
        "metrics_table": metrics_df,
        "chosen": chosen["model"],
        "best": best["model"],
        "recommendation": recommendation,
        "importance": importance_rows,
        "coefficients": coefs,
        "plots": plots,
        "model_path": str(chosen_path),
        "report_path": str(run_dir / "report.md"),
        "holdout": test_df is not None,
        "primary_metric": primary,
    }


def _default_params(pipe: Pipeline) -> dict:
    params = pipe.named_steps["model"].get_params()
    keep = ("alpha", "C", "n_estimators", "max_depth", "learning_rate", "min_samples_leaf",
            "n_neighbors", "n_components", "hidden_layer_sizes", "max_iter", "gamma", "kernel",
            "criterion", "l1_ratio", "var_smoothing", "reg_param", "solver", "epsilon")
    return {k: v for k, v in params.items() if k in keep and not isinstance(v, type(None))}


def _n_params(pipe: Pipeline) -> int:
    model = pipe.named_steps["model"]
    total = 0
    for attr in ("coef_", "feature_importances_", "intercept_"):
        v = getattr(model, attr, None)
        if v is not None:
            total += int(np.size(v))
    return total


def _space_size(params: dict) -> int:
    total = 1
    for values in params.values():
        total *= max(1, len(values)) if isinstance(values, (list, tuple)) else 1
    return total


def _inner_cv(task: str, seed: int):
    if task == "classification":
        return StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    return KFold(n_splits=3, shuffle=True, random_state=seed)


def _choose_with_one_se(ordered: list[dict], primary: str) -> dict:
    """The 'one standard error' rule: prefer the simplest model that is statistically
    indistinguishable from the best one."""
    best = ordered[0]
    threshold = best["cv"][primary]["mean"] - best["cv"][primary]["std"]
    candidates = [r for r in ordered if r["cv"][primary]["mean"] >= threshold]
    return min(candidates, key=lambda r: (r["complexity"], -r["cv"][primary]["mean"]))


def _recommendation(chosen: dict, best: dict, baseline_mean: float, primary: str) -> str:
    score = chosen["cv"][primary]["mean"]
    spread = chosen["cv"][primary]["std"]
    if score <= baseline_mean + max(0.01, spread):
        return (f"No model beats the baseline meaningfully (best {score:.3f} vs baseline "
                f"{baseline_mean:.3f}). The features do not carry a usable signal yet — "
                f"collect better features or more rows before trusting any model.")
    if chosen["model"] == best["model"]:
        return (f"{chosen['model']} is both the best and the simplest within one standard "
                f"error ({score:.3f} ± {spread:.3f} vs baseline {baseline_mean:.3f}). Use it.")
    return (f"{best['model']} scores highest ({best['cv'][primary]['mean']:.3f}), but "
            f"{chosen['model']} is within one standard error ({score:.3f} ± {spread:.3f}) and "
            f"simpler. Prefer {chosen['model']} unless the extra accuracy is worth the complexity.")


def _metrics_table(baseline: dict, results: list[dict], primary: str) -> pd.DataFrame:
    rows = []
    for r in [baseline] + results:
        row = {"model": r["model"], "complexity": r["complexity"]}
        if not r["cv"]:
            row["error"] = r.get("error", "")
            rows.append(row)
            continue
        for metric, stats in r["cv"].items():
            row[metric] = round(stats["mean"], 4)
            row[f"{metric}_std"] = round(stats["std"], 4)
        if r.get("holdout"):
            for metric, value in r["holdout"].items():
                row[f"holdout_{metric}"] = round(float(value), 4)
        rows.append(row)
    df = pd.DataFrame(rows)
    if primary in df.columns:
        df = df.sort_values(primary, ascending=False, na_position="last").reset_index(drop=True)
    return df


def fit_final(task: str, name: str, params: dict, numeric: list[str], categorical: list[str],
              cfg: TrainConfig, X: pd.DataFrame, y: pd.Series) -> Pipeline:
    pipe, _, _ = build_pipeline(task, name, numeric, categorical, cfg.balance)
    if params:
        clean = {f"model__{k}" if not k.startswith("model__") else k: v for k, v in params.items()}
        try:
            pipe.set_params(**clean)
        except Exception:
            pass
    pipe.fit(X, y)
    return pipe


def environment() -> dict:
    import sklearn

    env = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit-learn": sklearn.__version__,
        "xgboost": getattr(__import__("xgboost"), "__version__", None) if HAS_XGB else None,
    }
    try:
        import matplotlib

        env["matplotlib"] = matplotlib.__version__
    except Exception:
        pass
    return env


def _slug(text: str) -> str:
    keep = [c if (c.isalnum() or c in "-_") else "_" for c in str(text)]
    return "".join(keep)[:60] or "item"


# =============================================================================
# 4.  Report
# =============================================================================


def _report_md(rec: dict, baseline: dict, results: list[dict], primary: str) -> str:
    cfg = rec["config"]
    L: list[str] = []
    A = L.append
    A(f"# ML Workbench report — {rec['created_at']}\n")
    A("## 1. What was asked\n")
    A(f"- **Source file:** `{rec['source_file']}` (sha256 `{rec['source_hash']}`, {rec['rows']} rows)")
    A(f"- **Task:** {rec['task']} → target **`{rec['target']}`**")
    A(f"- **Features ({len(rec['features'])}):** {', '.join(f'`{f}`' for f in rec['features'])}")
    A(f"- **Categorical (one-hot encoded inside the pipeline):** "
      f"{', '.join(f'`{c}`' for c in rec['categorical']) if rec['categorical'] else 'none'}")
    A(f"- **Validation:** {cfg['cv_strategy']} CV with {cfg['folds']} folds, seed {cfg['seed']}"
      + (f", plus a {cfg['holdout_frac']:.0%} holdout" if cfg.get("holdout_frac") else ""))
    if cfg["search_iters"]:
        A(f"- **Hyperparameter search:** RandomizedSearchCV, {cfg['search_iters']} candidates, inner 3-fold")
    else:
        A("- **Hyperparameter search:** none (library defaults)")

    A("\n## 2. Results\n")
    A(f"Primary metric: `{primary}`. The baseline is what you beat by doing nothing clever.\n")
    A(f"| model | {primary} | std | holdout {primary} | complexity |")
    A("|---|---|---|---|---|")
    def line(r: dict) -> str:
        if not r["cv"]:
            return f"| {r['model']} | failed | | | |"
        m = r["cv"][primary]["mean"]
        sd = r["cv"][primary]["std"]
        hold = ""
        if r.get("holdout") and primary in r["holdout"]:
            hold = f"{r['holdout'][primary]:.4f}"
        return f"| {r['model']} | {m:.4f} | {sd:.4f} | {hold} | {r['complexity']} |"
    A(line(baseline))
    for r in sorted([x for x in results if x['cv']], key=lambda x: -x["cv"][primary]["mean"]):
        A(line(r))

    A(f"\n**Recommended model: `{rec['chosen_model']}`**\n")
    A(f"> {rec['recommendation']}\n")

    if cfg["task"] == "classification":
        A("\n### Why accuracy alone is not enough\n")
        A("Compare the recommended model's accuracy with the majority-class baseline above. "
          "A model only adds value above that line, and `f1_macro` shows whether rare classes "
          "are being ignored.\n")

    if rec.get("importance"):
        A("\n## 3. What drives the prediction\n")
        A("Permutation importance: how much the score drops when one column is shuffled.\n")
        A("| feature | importance | std |")
        A("|---|---|---|")
        for row in rec["importance"][:12]:
            A(f"| `{row['feature']}` | {row['importance']:+.4f} | {row['std']:.4f} |")
        neg = [r["feature"] for r in rec["importance"] if r["importance"] < 0]
        if neg:
            A(f"\nNegative importance (noise, or correlated with something better): "
              f"{', '.join(f'`{n}`' for n in neg[:8])}")

    if rec.get("coefficients"):
        A("\n### Standardised-size coefficients (linear models)\n")
        A("| feature | coefficient |")
        A("|---|---|")
        for row in rec["coefficients"][:15]:
            A(f"| `{row['feature']}` | {row['coefficient']:+.4f} |")

    A("\n## 4. Files in this run\n")
    A("| file | contents |")
    A("|---|---|")
    A("| `run.json` | every setting, metric, seed, library version and the data hash |")
    A("| `metrics.csv` | the results table above, machine-readable |")
    A("| `models/*.pkl` | the complete pipeline (impute + encode + scale + model) |")
    A("| `plots/*.png` | confusion matrix, predicted-vs-actual, residuals, correlation, importance |")
    A("| `feature_importance.csv` | permutation importance |")
    A("| `holdout_predictions.csv` | actual vs predicted on the untouched holdout rows |")

    A("\n## 5. How to reuse this model\n")
    A("```bash")
    A(f"mlw predict {_slug(rec['chosen_model'])}.pkl new_data.csv -o predictions.csv")
    A("```")
    A("\nFrom Python (no GUI, no app needed):\n")
    A("```python")
    A("import joblib, pandas as pd")
    A(f'bundle = joblib.load("{_slug(rec["chosen_model"])}.pkl")')
    A("model = bundle['pipeline']          # preprocessing included")
    A("new = pd.read_csv('new_data.csv')")
    A("print(model.predict(new[bundle['features']]))")
    A("```")

    A("\n## 6. Honest caveats\n")
    A("- The CV score is the average over folds; quote the spread, not just the mean.")
    A("- If a hyperparameter search ran and the score was measured on the same folds, the "
      "score is mildly optimistic — the holdout column (if present) is the unbiased one.")
    A("- A model cannot extrapolate: keep prediction inputs inside the training ranges "
      "recorded in `run.json`.")
    A("\n## 7. Environment\n")
    for k, v in rec["environment"].items():
        A(f"- {k}: {v}")
    return "\n".join(L) + "\n"


# =============================================================================
# 5.  Prediction with strict validation
# =============================================================================


def load_model(path: str | Path) -> dict:
    bundle = joblib.load(path)
    if not isinstance(bundle, dict) or bundle.get("kind") != "mlw.model":
        raise ValueError("This file is not an ML Workbench model (or was made by another tool).")
    return bundle


def predict_frame(bundle: dict, df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Predict + validate. Returns (output frame, validation report)."""
    features = bundle["features"]
    numeric = bundle["numeric"]
    categorical = bundle["categorical"]
    report: dict[str, Any] = {
        "rows": int(len(df)),
        "missing_columns": [f for f in features if f not in df.columns],
        "extra_columns": [c for c in df.columns if c not in features],
        "coerced_to_numeric": [],
        "unseen_categories": {},
        "out_of_range": {},
        "imputed_cells": int(df[features].isna().sum().sum()) if all(f in df.columns for f in features) else None,
        "warnings": [],
    }
    if report["missing_columns"]:
        raise ValueError(
            "The prediction file is missing required column(s): "
            + ", ".join(report["missing_columns"])
            + f"\nModel was trained on: {', '.join(features)}"
        )

    X = df[features].copy()

    # --- numeric columns: coerce text, remember it -------------------------
    for c in numeric:
        original = X[c]
        if not pd.api.types.is_numeric_dtype(original):
            converted = pd.to_numeric(original.astype(str).str.replace(",", "", regex=False), errors="coerce")
            if converted.notna().sum() >= original.notna().sum() * 0.5:
                report["coerced_to_numeric"].append(c)
            X[c] = converted
        else:
            X[c] = pd.to_numeric(X[c], errors="coerce")

    # --- categorical: normalise strings, count unseen ----------------------
    for c in categorical:
        known = set(bundle["categories"].get(c, []))
        vals = X[c].astype(str).str.strip().where(X[c].notna())
        unseen = sorted(set(vals.dropna()) - known)
        if unseen:
            report["unseen_categories"][c] = unseen
            report["warnings"].append(
                f"'{c}' contains {len(unseen)} value(s) never seen in training "
                f"({', '.join(unseen[:5])}): they are treated as 'none of the above'."
            )
        X[c] = vals

    # --- applicability domain: range checks -------------------------------
    for c in numeric:
        lo, hi = bundle["ranges"].get(c, [None, None])
        if lo is None:
            continue
        col = X[c]
        outside = col[(col < lo) | (col > hi)]
        if len(outside):
            report["out_of_range"][c] = {
                "count": int(len(outside)),
                "training_range": [lo, hi],
                "examples": [float(v) for v in outside.head(3).tolist()],
            }
    if report["out_of_range"]:
        report["warnings"].append(
            "Some rows fall outside the range seen in training — predictions there are "
            "extrapolation and should be treated as unreliable."
        )

    n_missing = int(X.isna().sum().sum())
    if n_missing:
        report["warnings"].append(
            f"{n_missing} empty cell(s) in the prediction file will be imputed with training-time "
            f"medians / most-frequent values."
        )

    model = bundle["pipeline"]
    pred = np.ravel(model.predict(X))
    out = df.copy()
    if bundle["task"] == "regression":
        out[f"predicted_{bundle['target']}"] = pred
    else:
        pred = pd.Series(pred).astype(str).to_numpy()
        out[f"predicted_{bundle['target']}"] = pred
        try:
            proba = model.predict_proba(X)
            for i, cls in enumerate(model.named_steps["model"].classes_):
                out[f"proba_{cls}"] = np.round(proba[:, i], 4)
        except Exception:
            pass

    if report["out_of_range"] or report["unseen_categories"]:
        for idx in out.index:
            f = []
            for c, info in report["out_of_range"].items():
                v = X.at[idx, c]
                lo, hi = info["training_range"]
                if pd.notna(v) and (v < lo or v > hi):
                    f.append(f"{c} outside")
            out.loc[idx, "_warning"] = "; ".join(f)
        report["flagged_rows"] = int((out.get("_warning", pd.Series(dtype=str)).fillna("") != "").sum())

    return out, report


# =============================================================================
# 6.  Optimiser
# =============================================================================


def optimise_grid(
    bundle: dict,
    ranges: dict[str, tuple[float, float, int]],
    categorical_levels: dict[str, list[str]] | None = None,
    maximise: bool = True,
    top_n: int = 25,
    log: Callable[[str], None] | None = None,
) -> pd.DataFrame:
    """Fill the gap left by the step/array grid file: ranges in, ranked candidates out."""
    log = log or (lambda m: None)
    feature_cols = list(ranges.keys()) + list((categorical_levels or {}).keys())
    unknown = [c for c in feature_cols if c not in bundle["features"]]
    if unknown:
        raise ValueError(f"Not features of this model: {unknown}")

    axes: list[list] = []
    for c, (lo, hi, steps) in ranges.items():
        steps = max(1, int(steps))
        axes.append([round(lo + (hi - lo) * i / steps, 6) for i in range(steps + 1)])
    for c, levels in (categorical_levels or {}).items():
        axes.append(list(levels))

    combos = list(itertools.product(*axes))
    if len(combos) > 2_000_000:
        raise ValueError(f"Grid too large ({len(combos):,} rows). Use fewer steps or fewer features.")
    log(f"Grid: {len(combos):,} candidate combinations")

    grid = pd.DataFrame(combos, columns=feature_cols)
    # every other feature is held at its training median / mode
    for f in bundle["features"]:
        if f in grid.columns:
            continue
        if f in bundle["numeric"]:
            lo, hi = bundle["ranges"][f]
            grid[f] = (lo + hi) / 2
        else:
            cats = bundle["categories"].get(f) or ["unknown"]
            grid[f] = cats[0]
    grid = grid[bundle["features"]]

    model = bundle["pipeline"]
    pred = np.ravel(model.predict(grid))
    grid[f"predicted_{bundle['target']}"] = pred
    grid = grid.sort_values(f"predicted_{bundle['target']}", ascending=not maximise).reset_index(drop=True)

    # boundary flags: a "best" point at the edge of the grid means the optimum is outside
    boundary = []
    for _, row in grid.head(max(top_n, 50)).iterrows():
        at_edge = []
        for c, (lo, hi, steps) in ranges.items():
            vals = sorted(grid[c].unique())
            if len(vals) > 1 and row[c] in (vals[0], vals[-1]):
                at_edge.append(c)
        boundary.append(", ".join(at_edge))
    grid["_at_grid_boundary"] = ""
    for i, b in enumerate(boundary):
        grid.loc[i, "_at_grid_boundary"] = b
    grid["_inside_training_range"] = grid.apply(
        lambda r: all(
            bundle["ranges"][c][0] <= r[c] <= bundle["ranges"][c][1]
            for c in bundle["numeric"] if c in ranges
        ),
        axis=1,
    )
    return grid


# =============================================================================
# 7.  Time-series helpers (lag / rolling features)
# =============================================================================


def add_lag_features(df: pd.DataFrame, target: str, group_col: str | None = None,
                     lags: Sequence[int] = (1, 2, 3), rolls: Sequence[int] = (3,),
                     sort_col: str | None = None) -> pd.DataFrame:
    """Turn a demand-forecasting problem into a supervised one — something the
    original app cannot do at all."""
    df = df.copy()
    if sort_col and sort_col in df.columns:
        df = df.sort_values(sort_col).reset_index(drop=True)
    grouped = df.groupby(group_col, sort=False) if group_col else df.groupby(lambda _: 0)
    parts = []
    for _, chunk in grouped:
        chunk = chunk.copy()
        for lag in lags:
            chunk[f"{target}_lag{lag}"] = chunk[target].shift(lag)
        for w in rolls:
            chunk[f"{target}_roll{w}_mean"] = chunk[target].shift(1).rolling(w).mean()
            chunk[f"{target}_roll{w}_std"] = chunk[target].shift(1).rolling(w).std()
        parts.append(chunk)
    out = pd.concat(parts).sort_index()
    return out.dropna().reset_index(drop=True)


# =============================================================================
# 8.  Molecules
# =============================================================================


SPECIAL_DESCRIPTORS = {
    "HeavyAtoms": lambda m: m.GetNumHeavyAtoms(),
}

DESCRIPTOR_SET = [
    "MolWt", "MolLogP", "TPSA", "NumHDonors", "NumHAcceptors", "NumRotatableBonds",
    "RingCount", "FractionCSP3", "HeavyAtoms", "qed", "NumAromaticRings",
    "MaxAbsEStateIndex", "MinAbsEStateIndex", "BalabanJ", "BertzCT",
]


def smiles_features(
    smiles: Iterable[str],
    mode: str = "descriptors",
    bits: int = 1024,
    radius: int = 2,
    max_path: int = 7,
    standardize: bool = True,
) -> tuple[pd.DataFrame, list[dict]]:
    """Canonicalise + standardise molecules, then descriptors or fingerprints.
    Returns (feature frame, per-row notes)."""
    from rdkit import Chem, RDLogger
    from rdkit.Chem import Descriptors, rdMolDescriptors as D, rdFingerprintGenerator

    RDLogger.DisableLog("rdApp.*")

    def standardise(mol):
        if mol is None:
            return None, "unparsable"
        note = ""
        if standardize:
            frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
            if len(frags) > 1:
                mol = max(frags, key=lambda m: m.GetNumHeavyAtoms())
                note = "salt/mixture: kept largest fragment"
        return mol, note

    notes: list[dict] = []
    rows: list[dict] = []
    mols: list[Any] = []
    for i, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(str(smi)) if isinstance(smi, str) else None
        mol, note = standardise(mol)
        mols.append(mol)
        notes.append({"row": i, "smiles": smi, "status": "ok" if mol is not None else "failed", "note": note})
        if mol is None:
            rows.append({})
            continue
        if mode == "descriptors":
            vals = {}
            for name in DESCRIPTOR_SET:
                fn = SPECIAL_DESCRIPTORS.get(name) or getattr(Descriptors, name, None) or getattr(D, name, None)
                try:
                    vals[name] = float(fn(mol)) if fn else np.nan
                except Exception:
                    vals[name] = np.nan
            rows.append(vals)
        else:
            if mode == "morgan":
                gen = rdFingerprintGenerator.GetMorganGenerator(radius=int(radius), fpSize=int(bits))
                vec = gen.GetFingerprintAsNumPy(mol)
            else:
                gen = rdFingerprintGenerator.GetRDKitFPGenerator(maxPath=int(max_path), fpSize=int(bits))
                vec = gen.GetFingerprintAsNumPy(mol)
            rows.append({f"fp{i:04d}": int(v) for i, v in enumerate(vec)})

    feats = pd.DataFrame(rows)
    if mode == "descriptors" and len(feats):
        empty = [c for c in feats.columns if feats[c].isna().all()]
        if empty:
            feats = feats.drop(columns=empty)
            notes.append({"row": -1, "smiles": "", "status": "note",
                          "note": f"descriptors unavailable in this RDKit build and dropped: {', '.join(empty)}"})
    feats.insert(0, "canonical_smiles", [Chem.MolToSmiles(m) if m is not None else None for m in mols])
    return feats, notes


def applicability_domain(train_smiles: Sequence[str], query_smiles: Sequence[str], radius: int = 2, bits: int = 2048) -> pd.DataFrame:
    """Max Tanimoto similarity of each query molecule to the training set."""
    from rdkit import Chem, DataStructs, RDLogger
    from rdkit.Chem import rdFingerprintGenerator

    RDLogger.DisableLog("rdApp.*")
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=bits)
    train_fps = []
    for s in train_smiles:
        m = Chem.MolFromSmiles(str(s)) if isinstance(s, str) else None
        if m is not None:
            train_fps.append(gen.GetFingerprint(m))
    out = []
    for s in query_smiles:
        m = Chem.MolFromSmiles(str(s)) if isinstance(s, str) else None
        if m is None or not train_fps:
            out.append({"query": s, "max_similarity": np.nan, "nearest_index": None, "in_domain": False})
            continue
        sims = DataStructs.BulkTanimotoSimilarity(gen.GetFingerprint(m), train_fps)
        i = int(np.argmax(sims))
        out.append({"query": s, "max_similarity": float(sims[i]), "nearest_index": i,
                    "in_domain": bool(sims[i] >= 0.4)})
    return pd.DataFrame(out)
