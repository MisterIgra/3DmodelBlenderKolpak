from pathlib import Path
import argparse, json, sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from autokolpak_ai.validate import validate_dataset


def main():
    p = argparse.ArgumentParser(description="Deep-check annotations, labels, images and model uniqueness")
    p.add_argument("--data", required=True)
    p.add_argument("--report", default="./runs/dataset_report.json")
    args = p.parse_args()
    report = validate_dataset(args.data, args.report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["ok"]:
        raise SystemExit(2)

if __name__ == "__main__": main()
