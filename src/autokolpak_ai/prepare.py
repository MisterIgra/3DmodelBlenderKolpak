from __future__ import annotations
import argparse, json
from .validate import validate_dataset
from .dataset import load_annotations, split_by_model
from pathlib import Path


def prepare_dataset(data_dir, output_dir="./prepared_dataset", seed=42,
                    validation_fraction=.15, test_fraction=.15):
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    report = validate_dataset(data_dir, out / "dataset_report.json")
    if not report["ok"]:
        raise ValueError(f"Датасет не прошёл проверку: {report['errors_count']} ошибок. Смотрите dataset_report.json")
    rows = load_annotations(data_dir)
    tr, va, te, ids = split_by_model(rows, validation_fraction, test_fraction, seed)
    payload = {name: {"model_ids": ids[name], "images": len(part), "records": [r["image_path"] for r in part]}
               for name, part in (("train", tr), ("validation", va), ("test", te))}
    (out / "splits.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"validation": report, "splits": {k: {"images": v["images"], "models": len(v["model_ids"])} for k,v in payload.items()}}


def main():
    p=argparse.ArgumentParser(description="Validate AutoKolpak dataset and create model-level split")
    p.add_argument("--data", required=True); p.add_argument("--output", default="./prepared_dataset")
    p.add_argument("--seed", type=int, default=42); p.add_argument("--validation-fraction", type=float, default=.15)
    p.add_argument("--test-fraction", type=float, default=.15); a=p.parse_args()
    print(json.dumps(prepare_dataset(a.data,a.output,a.seed,a.validation_fraction,a.test_fraction),ensure_ascii=False,indent=2))
if __name__ == "__main__": main()
