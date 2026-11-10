"""Generate the screenshot set used inside the User Guide PDF.

Runs the real Tkinter app on a virtual display, drives it exactly like a user
would, and grabs each tab. Output: docs/guide_assets/shot_*.png
"""
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mlw import core          # noqa: E402
from mlw.gui import App       # noqa: E402

ASSETS = ROOT / "docs" / "guide_assets"
ASSETS.mkdir(parents=True, exist_ok=True)
RUNS = ROOT / "guide_runs"
SAMPLE = ROOT / "sample_data"

app = App()
app.geometry("1400x880+0+0")
app.update()


def shot(name: str):
    for _ in range(3):
        app.update()
        app.update_idletasks()
        time.sleep(0.35)
    subprocess.run(["import", "-window", "root", str(ASSETS / f"shot_{name}.png")], check=False)
    print("shot", name, flush=True)


def load(path: Path, target: str, untick=()):
    app.df, app.meta = core.load_csv(path)
    app.data_path = str(path)
    app.lbl_file.configure(text=f"{path.name} ({app.meta['encoding']})")
    app.header_status.configure(text=f"{path.name} \u00b7 {app.meta['rows']} rows \u00d7 {app.meta['cols']} cols")
    app._populate_preview()
    app._populate_roles()
    app.cmb_target.set(target)
    app._on_target_change()
    for col, var in app.feature_vars.items():
        var.set(col not in untick)
    app._update_feature_count()
    app.update()


# ---------------------------------------------------------------- 1. Data tab
load(SAMPLE / "reactor_yield.csv", "yield_pct", untick=("purity_pct",))
shot("01_data")

# messy file -> the health-check list with real findings
load(ASSETS / "messy_demo.csv", "grade")
app.nb.select(0)
shot("02_audit")

# ---------------------------------------------------------------- 2. Train tab
load(SAMPLE / "reactor_yield.csv", "yield_pct", untick=("purity_pct",))
app.nb.select(1)
app.var_holdout.set(True)
app.spn_folds.set(5)
app.spn_seed.set(77)
app.spn_search.set(12)
for name, var in app.model_vars.items():
    var.set(name in {"Linear", "Ridge", "RandomForest", "GradientBoosting"})
app.update()
shot("03_train")

cfg = core.TrainConfig(
    target="yield_pct",
    features=["temperature_C", "pressure_bar", "catalyst_wt", "residence_min"],
    task="regression", folds=5, seed=77, search_iters=12, holdout_frac=0.2,
    models=["Linear", "Ridge", "RandomForest", "GradientBoosting"],
)
app.set_status("training \u2026")
res = core.train_and_evaluate(app.df, cfg, out_root=RUNS, log=app.log,
                              name="reactor_yield", source_path=app.data_path)
app._training_done(res)
app.nb.select(2)
app.update()
shot("04_results")

# ------------------------------------------------------------- 4. Predict tab
app.ent_pred_model.delete(0, "end"); app.ent_pred_model.insert(0, res["model_path"])
app.ent_pred_data.delete(0, "end"); app.ent_pred_data.insert(0, str(SAMPLE / "reactor_predict.csv"))
app.ent_pred_out.delete(0, "end"); app.ent_pred_out.insert(0, str(ASSETS / "guide_predictions.csv"))
app.nb.select(3)
app.update()
app.run_predict()
app.update()
shot("05_predict")

# ------------------------------------------------------------ 5. Optimise tab
app.ent_opt_model.delete(0, "end"); app.ent_opt_model.insert(0, res["model_path"])
app.nb.select(4)
app.update()
app.load_ranges()
app.update()
shot("06_optimise_setup")
app.run_optimise()
app.update()
shot("07_optimise_result")

# ----------------------------------------------------------- 6. Molecules tab
app.nb.select(5)
app.ent_mol_csv.delete(0, "end"); app.ent_mol_csv.insert(0, str(SAMPLE / "molecules.csv"))
app.update()
app.run_molecules()
app.update()
shot("08_molecules")

# --------------------------------------------- classification run for the guide
load(SAMPLE / "reactor_grade.csv", "grade", untick=("shift",))
cfg2 = core.TrainConfig(
    target="grade",
    features=["temperature_C", "pressure_bar", "catalyst_wt", "residence_min", "reactor", "solvent"],
    task="classification", folds=5, seed=77, search_iters=12, holdout_frac=0.2,
    models=["Logistic", "RandomForest"], balance=True,
)
res2 = core.train_and_evaluate(app.df, cfg2, out_root=RUNS, log=app.log,
                               name="reactor_grade", source_path=app.data_path)
app._training_done(res2)
app.nb.select(2)
app.update()
shot("09_results_classification")

print("RUN_DIR", res["run_dir"], flush=True)
print("RUN_DIR_CLF", res2["run_dir"], flush=True)
print("MODEL", res["model_path"], flush=True)
print("METRICS:\n", res["metrics_table"].to_string(index=False), flush=True)
print("METRICS_CLF:\n", res2["metrics_table"].to_string(index=False), flush=True)
print("IMPORTANCE:", res["importance"], flush=True)
print("DONE", flush=True)
