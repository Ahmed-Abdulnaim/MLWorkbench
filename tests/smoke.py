"""End-to-end smoke test of the core (no GUI needed)."""
import json, sys, tempfile, time
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mlw import core

DATA = Path(__file__).resolve().parents[1] / "sample_data"  # the repo's bundled CSVs
tmp = Path(tempfile.mkdtemp())

print("== load + audit ==")
df, meta = core.load_csv(DATA / "reactor_yield.csv")
print(meta, df.shape)
issues = core.audit(df, target="yield_pct")
for i in issues:
    print(f"  [{i.level}] {i.code}: {i.message}")
print("inferred task:", core.infer_task(df["yield_pct"]))

print("\n== regression training ==")
cfg = core.TrainConfig(target="yield_pct",
                       features=["temperature_C", "pressure_bar", "catalyst_wt", "residence_min"],
                       task="regression", folds=5, seed=77, search_iters=6,
                       models=["Linear", "Ridge", "RandomForest", "GradientBoosting"],
                       holdout_frac=0.2, save_all_models=True)
t0 = time.time()
res = core.train_and_evaluate(df, cfg, out_root=tmp, log=print)
print(f"\n-> {time.time()-t0:.1f}s  chosen={res['chosen']} best={res['best']}")
print(res["metrics_table"].to_string(index=False))
print("rec:", res["recommendation"])
print("importance:", [(r["feature"], round(r["importance"], 3)) for r in res["importance"][:4]])
print("plots:", list(res["plots"]))
print("run dir contents:", sorted(p.name for p in Path(res["run_dir"]).iterdir()))

print("\n== predict ==")
bundle = core.load_model(res["model_path"])
new, rep = core.predict_frame(bundle, pd.read_csv(DATA / "reactor_predict.csv"))
print(new.to_string(index=False))
print({k: v for k, v in rep.items() if v not in ([], {}, None)})

print("\n== predict with broken schema ==")
bad = pd.read_csv(DATA / "reactor_predict.csv").rename(columns={"temperature_C": "Temperature_C"})
try:
    core.predict_frame(bundle, bad)
except ValueError as e:
    print("caught as expected:", str(e).splitlines()[0])

print("\n== optimise ==")
grid = core.optimise_grid(bundle,
    ranges={"temperature_C": (70, 110, 8), "pressure_bar": (1, 5, 4),
            "catalyst_wt": (1, 3, 4), "residence_min": (40, 120, 4)}, maximise=True, log=print)
print(grid.head(5).to_string(index=False))
print("grid rows:", len(grid))

print("\n== classification ==")
cdf, _ = core.load_csv(DATA / "reactor_grade.csv")
ccfg = core.TrainConfig(target="grade",
                        features=["temperature_C","pressure_bar","catalyst_wt","residence_min","reactor","solvent","shift"],
                        task="classification", folds=5, seed=77, search_iters=4,
                        models=["Logistic","RandomForest","DecisionTree"], holdout_frac=0.2, balance=True)
cres = core.train_and_evaluate(cdf, ccfg, out_root=tmp, log=print)
print(cres["metrics_table"].to_string(index=False))
print("rec:", cres["recommendation"])

print("\n== lag features ==")
ts = pd.DataFrame({"day": range(40), "sales": [10 + (i % 7) + i * 0.3 for i in range(40)]})
lagged = core.add_lag_features(ts, "sales", lags=(1, 2), rolls=(3,))
print(lagged.head(3).to_string(index=False), lagged.shape)

print("\n== smiles ==")
try:
    smi = pd.read_csv(DATA / "molecules.csv")
    feats, notes = core.smiles_features(smi["SMILES"], mode="descriptors")
    print("descriptor table:", feats.shape, "| failures:", sum(n["status"] != "ok" for n in notes))
    print(feats.head(3).to_string(index=False))
    fp, _ = core.smiles_features(smi["SMILES"].head(5), mode="morgan", bits=256)
    print("morgan fp table:", fp.shape)
    ad = core.applicability_domain(smi["SMILES"].head(30), smi["SMILES"].tail(4))
    print(ad.to_string(index=False))
except ImportError:
    print("rdkit not available in this environment")

print("\n== report.md head ==")
print(Path(res["report_path"]).read_text()[:1200])
print("\nALL CORE CHECKS DONE ->", tmp)
