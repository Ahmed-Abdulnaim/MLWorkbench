"""Drive the GUI headlessly, then screenshot each tab (used for visual QA)."""
import sys, time, subprocess
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mlw.gui import App
from mlw import core

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "build_logs" / "shots"; OUT.mkdir(parents=True, exist_ok=True)
app = App()
app.geometry("1400x880+0+0")
app.update()

def shot(name):
    app.update(); app.update_idletasks(); time.sleep(0.6); app.update()
    subprocess.run(["import", "-window", "root", str(OUT / f"{name}.png")], check=False)
    print("shot", name)

# load data
app.data_path = str(ROOT / "sample_data" / "reactor_yield.csv")
app.df, app.meta = core.load_csv(app.data_path)
app.lbl_file.configure(text=f"{app.data_path} (utf-8-sig)")
app.header_status.configure(text="lab1a_training_numeric.csv · 140 rows × 6 cols")
app._populate_preview(); app._populate_roles()
app.cmb_target.set("yield_pct"); app._on_target_change()
app.update()
shot("01_data")

# training tab
app.nb.select(1); app.update()
shot("02_train")

# run a real training job inline (small + fast) to fill results
cfg = core.TrainConfig(target="yield_pct",
                       features=["temperature_C","pressure_bar","catalyst_wt","residence_min"],
                       task="regression", folds=5, seed=77, search_iters=6,
                       models=["Linear","Ridge","RandomForest","GradientBoosting"],
                       holdout_frac=0.2)
res = core.train_and_evaluate(app.df, cfg, out_root=str(ROOT / "MLWorkbench_runs"),
                              log=app.log, name="reactor", source_path=app.data_path)
app._training_done(res)
app.nb.select(2); app.update()
shot("03_results")

# predict tab with a real model
app.ent_pred_model.delete(0,"end"); app.ent_pred_model.insert(0, res["model_path"])
app.ent_pred_data.delete(0,"end"); app.ent_pred_data.insert(0, str(ROOT / "sample_data" / "reactor_predict.csv"))
app.ent_pred_out.delete(0,"end"); app.ent_pred_out.insert(0, str(OUT.parent / "preds.csv"))
app.nb.select(3); app.update()
app.run_predict(); app.update()
shot("04_predict")

# optimise tab
app.ent_opt_model.delete(0,"end"); app.ent_opt_model.insert(0, res["model_path"])
app.nb.select(4); app.update()
app.load_ranges(); app.update()
app.run_optimise(); app.update()
shot("05_optimise")

# molecules tab
app.nb.select(5); app.update()
app.ent_mol_csv.delete(0,"end"); app.ent_mol_csv.insert(0, str(ROOT / "sample_data" / "molecules.csv"))
app.cmb_mol_col.set("SMILES"); app.update()
app.run_molecules(); app.update()
shot("06_molecules")

print("done")
app.destroy()
