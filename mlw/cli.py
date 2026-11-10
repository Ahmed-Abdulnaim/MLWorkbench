"""mlw.cli — the same engine as the desktop app, for scripts and automation.

Examples
--------
    mlw audit data.csv
    mlw train data.csv --target yield --task auto --search 20 --holdout 0.2
    mlw train data.csv --target grade --models Logistic,RandomForest --balance
    mlw train sales.csv --target sales --lag 1,2,7 --group store --time-series
    mlw predict runs/xxx/models/ridge.pkl new.csv -o predictions.csv
    mlw optimise runs/xxx/models/rf.pkl --maximise -r temperature:70:110:9 -o grid.csv
    mlw smiles molecules.csv --column SMILES --mode descriptors -o features.csv
    mlw gui
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from . import core


def _log(msg: str) -> None:
    """Never crash on a character the Windows console cannot encode."""
    try:
        print(msg, flush=True)
    except UnicodeEncodeError:
        enc = getattr(sys.stdout, "encoding", None) or "ascii"
        print(str(msg).encode(enc, "replace").decode(enc, "replace"), flush=True)


def cmd_audit(args) -> int:
    df, meta = core.load_csv(args.data)
    _log(f"{args.data}: {meta['rows']} rows x {meta['cols']} cols, encoding={meta['encoding']}, sep={meta['sep']!r}")
    issues = core.audit(df, target=args.target)
    if not issues:
        _log("No findings — the file is clean for modelling.")
    for issue in issues:
        _log(f"[{issue.level.upper():7s}] {issue.code:18s} {issue.message}")
        if issue.hint:
            _log(f"{'':27s}-> {issue.hint}")
    if args.target:
        _log(f"Inferred task for '{args.target}': {core.infer_task(df[args.target])}")
    return 0


def _parse_models(text: str | None, task: str) -> list[str] | None:
    if not text or text == "auto":
        return None
    if text == "all":
        return core.available_models(task)
    return [m.strip() for m in text.split(",") if m.strip()]


def cmd_train(args) -> int:
    df, _ = core.load_csv(args.data)
    task = args.task
    if task == "auto":
        task = core.infer_task(df[args.target])
    if args.lag:
        lags = [int(x) for x in args.lag.split(",") if x.strip()]
        before = len(df)
        df = core.add_lag_features(df, args.target, group_col=args.group,
                                   lags=lags, rolls=tuple(int(r) for r in (args.roll.split(",") if args.roll else [])),
                                   sort_col=args.sort_by)
        _log(f"Lag features: {before} -> {len(df)} rows (rows without history dropped)")

    features = [c for c in df.columns if c not in {args.target, args.sort_by}]
    if args.features:
        features = [f.strip() for f in args.features.split(",") if f.strip()]

    cfg = core.TrainConfig(
        target=args.target,
        features=features,
        task=task,
        folds=args.folds,
        cv_strategy=args.cv,
        group_col=args.group,
        holdout_frac=args.holdout,
        seed=args.seed,
        search_iters=args.search,
        models=_parse_models(args.models, task),
        balance=args.balance,
        save_all_models=args.save_all,
    )
    result = core.train_and_evaluate(df, cfg, out_root=args.out, name=Path(args.data).stem,
                                     log=_log, source_path=args.data)
    _log("\n" + result["metrics_table"].to_string(index=False))
    _log(f"\nRecommended: {result['chosen']}")
    _log(f"Model:   {result['model_path']}")
    _log(f"Report:  {result['report_path']}")
    if args.json:
        Path(args.json).write_text(json.dumps({k: v for k, v in result.items()
                                               if k not in ("metrics_table",)}, indent=2, default=str))
    return 0


def cmd_predict(args) -> int:
    bundle = core.load_model(args.model)
    df, _ = core.load_csv(args.data)
    out, report = core.predict_frame(bundle, df)
    path = Path(args.output) if args.output else Path(args.data).with_suffix("").with_name(
        Path(args.data).stem + "_predictions.csv")
    out.to_csv(path, index=False)
    _log(f"{len(out)} row(s) written to {path}")
    _log(f"predicted column: predicted_{bundle['target']}")
    for w in report["warnings"]:
        _log(f"WARNING: {w}")
    if report["missing_columns"]:
        return 2
    if report["unseen_categories"]:
        _log(f"Unseen categories: {report['unseen_categories']}")
    if report["out_of_range"]:
        _log(f"Extrapolated rows: {report['out_of_range']}")
    return 0


def cmd_optimise(args) -> int:
    bundle = core.load_model(args.model)
    ranges = {}
    for spec in args.range:
        name, lo, hi, steps = spec.split(":")
        ranges[name] = (float(lo), float(hi), int(steps))
    levels = {}
    for spec in args.level or []:
        name, values = spec.split(":")
        levels[name] = [v for v in values.split("|")]
    grid = core.optimise_grid(bundle, ranges, levels, maximise=args.maximise,
                              top_n=args.top, log=_log)
    out = Path(args.output or "optimised_candidates.csv")
    grid.to_csv(out, index=False)
    _log(grid.head(args.top).to_string(index=False))
    _log(f"\n{len(grid):,} candidates written to {out}")
    _log(f"Target column: predicted_{bundle['target']}")
    return 0


def cmd_smiles(args) -> int:
    df, _ = core.load_csv(args.data)
    if args.column not in df.columns:
        _log(f"Column '{args.column}' not found. Available: {list(df.columns)}")
        return 2
    feats, notes = core.smiles_features(df[args.column], mode=args.mode, bits=args.bits,
                                        radius=args.radius, max_path=args.max_path,
                                        standardize=not args.no_standardise)
    merged = pd.concat([df.reset_index(drop=True), feats.reset_index(drop=True)], axis=1)
    out = Path(args.output or "molecule_features.csv")
    merged.to_csv(out, index=False)
    failed = [n for n in notes if n["status"] != "ok"]
    changed = [n for n in notes if n["note"]]
    _log(f"{len(feats)} molecules -> {feats.shape[1]} feature columns, written to {out}")
    if failed:
        _log(f"WARNING: {len(failed)} SMILES could not be parsed (rows {[f['row'] for f in failed][:10]})")
    if changed:
        _log(f"{len(changed)} molecule(s) were standardised (salts/mixtures -> largest fragment)")
    if args.ad_against:
        ref = pd.read_csv(args.ad_against)
        ad = core.applicability_domain(ref[args.column], df[args.column])
        ad_path = out.with_name(out.stem + "_applicability_domain.csv")
        ad.to_csv(ad_path, index=False)
        _log(f"Applicability domain vs {args.ad_against}: {ad_path}")
        _log(ad.describe(include="all").to_string())
    return 0


def cmd_gui(_args) -> int:
    from .gui import main

    main()
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="mlw", description="ML Workbench — honest no-code machine learning.")
    p.add_argument("--version", action="version", version=f"ML Workbench {core.VERSION}")
    sub = p.add_subparsers(dest="command", required=True)

    a = sub.add_parser("audit", help="report data problems before training")
    a.add_argument("data")
    a.add_argument("--target")
    a.set_defaults(func=cmd_audit)

    t = sub.add_parser("train", help="train, cross-validate, compare, pick and save")
    t.add_argument("data")
    t.add_argument("--target", required=True)
    t.add_argument("--features", help="comma separated; default = all other columns")
    t.add_argument("--task", choices=["auto", "regression", "classification"], default="auto")
    t.add_argument("--models", default="auto", help="auto | all | comma separated names")
    t.add_argument("--folds", type=int, default=5)
    t.add_argument("--cv", choices=["kfold", "group", "timeseries"], default="kfold")
    t.add_argument("--group", help="group column for group-aware CV / lag grouping")
    t.add_argument("--holdout", type=float, default=0.0, help="fraction held back (0-0.5)")
    t.add_argument("--seed", type=int, default=77)
    t.add_argument("--search", type=int, default=0, help="RandomizedSearchCV budget (0 = off)")
    t.add_argument("--balance", action="store_true", help="class_weight='balanced' where supported")
    t.add_argument("--lag", help="comma separated lags for time series, e.g. 1,2,7")
    t.add_argument("--roll", help="comma separated rolling window sizes")
    t.add_argument("--sort-by", dest="sort_by", help="date/time column to sort before lagging")
    t.add_argument("--out", default="runs")
    t.add_argument("--json", help="also write the result summary here")
    t.add_argument("--save-all", action="store_true")
    t.set_defaults(func=cmd_train)

    pr = sub.add_parser("predict", help="apply a saved model to new rows")
    pr.add_argument("model")
    pr.add_argument("data")
    pr.add_argument("-o", "--output")
    pr.set_defaults(func=cmd_predict)

    o = sub.add_parser("optimise", help="rank candidate conditions with a saved model")
    o.add_argument("model")
    o.add_argument("-r", "--range", action="append", required=True,
                   help="name:min:max:steps (repeatable)")
    o.add_argument("-l", "--level", action="append", help="name:v1|v2|v3 (repeatable)")
    o.add_argument("--maximise", action="store_true", default=True)
    o.add_argument("--minimise", dest="maximise", action="store_false")
    o.add_argument("--top", type=int, default=25)
    o.add_argument("-o", "--output")
    o.set_defaults(func=cmd_optimise)

    s = sub.add_parser("smiles", help="SMILES -> descriptors or fingerprints")
    s.add_argument("data")
    s.add_argument("--column", default="SMILES")
    s.add_argument("--mode", choices=["descriptors", "morgan", "rdkit"], default="descriptors")
    s.add_argument("--bits", type=int, default=1024)
    s.add_argument("--radius", type=int, default=2)
    s.add_argument("--max-path", dest="max_path", type=int, default=7)
    s.add_argument("--no-standardise", action="store_true")
    s.add_argument("--ad-against", help="CSV of training SMILES for applicability domain")
    s.add_argument("-o", "--output")
    s.set_defaults(func=cmd_smiles)

    g = sub.add_parser("gui", help="open the desktop app")
    g.set_defaults(func=cmd_gui)
    return p


def main(argv: list[str] | None = None) -> int:
    # Windows consoles default to a legacy code page; make output tolerant.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
        except Exception:
            pass
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except KeyboardInterrupt:
        return 130
    except Exception as exc:  # noqa: BLE001
        _log(f"ERROR: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
