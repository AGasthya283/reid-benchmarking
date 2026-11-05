"""TrackEval (HOTA / CLEAR / Identity) on MOT-format results."""
from __future__ import annotations

from pathlib import Path


def evaluate(gt_dir: Path, seqmap: Path, trackers_dir: Path, tracker_names: list[str], threads: int = 4) -> dict:
    from .third_party import trackeval

    eval_cfg = trackeval.Evaluator.get_default_eval_config()
    eval_cfg.update({"USE_PARALLEL": threads > 1, "NUM_PARALLEL_CORES": threads, "PRINT_RESULTS": False,
                     "PRINT_ONLY_COMBINED": True, "PRINT_CONFIG": False, "TIME_PROGRESS": False,
                     "DISPLAY_LESS_PROGRESS": True, "OUTPUT_SUMMARY": False, "OUTPUT_DETAILED": False,
                     "PLOT_CURVES": False, "OUTPUT_EMPTY_CLASSES": False, "LOG_ON_ERROR": None,
                     "BREAK_ON_ERROR": True, "RETURN_ON_ERROR": False})
    ds_cfg = trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
    ds_cfg.update({"GT_FOLDER": str(gt_dir), "TRACKERS_FOLDER": str(trackers_dir), "OUTPUT_FOLDER": None,
                   "TRACKERS_TO_EVAL": tracker_names, "CLASSES_TO_EVAL": ["pedestrian"], "BENCHMARK": "MOT17",
                   "SPLIT_TO_EVAL": "train", "SEQMAP_FILE": str(seqmap), "SKIP_SPLIT_FOL": True,
                   "TRACKER_SUB_FOLDER": "data", "DO_PREPROC": True, "PRINT_CONFIG": False})
    evaluator = trackeval.Evaluator(eval_cfg)
    dataset = trackeval.datasets.MotChallenge2DBox(ds_cfg)
    metrics = [trackeval.metrics.HOTA(), trackeval.metrics.CLEAR({"PRINT_CONFIG": False}),
               trackeval.metrics.Identity({"PRINT_CONFIG": False})]
    res, _ = evaluator.evaluate([dataset], metrics)
    out = {}
    for name in tracker_names:
        r = res["MotChallenge2DBox"][name]
        out[name] = {seq: _summ(v["pedestrian"]) for seq, v in r.items()}
    return out


def _summ(r: dict) -> dict:
    h, c, i = r["HOTA"], r["CLEAR"], r["Identity"]
    return {
        "HOTA": 100 * float(h["HOTA"].mean()), "DetA": 100 * float(h["DetA"].mean()),
        "AssA": 100 * float(h["AssA"].mean()), "LocA": 100 * float(h["LocA"].mean()),
        "MOTA": 100 * float(c["MOTA"]), "IDF1": 100 * float(i["IDF1"]),
        "IDSW": int(c["IDSW"]), "FP": int(c["CLR_FP"]), "FN": int(c["CLR_FN"]), "Frag": int(c["Frag"]),
    }
