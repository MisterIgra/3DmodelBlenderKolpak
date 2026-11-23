# Blender Dataset Generator v4

See the root `README.md` for exact Windows commands. `generate_dataset.py` writes `metadata/annotations.jsonl`, `metadata/hubcap_*.json` and `images/*.png`. Each image annotation includes `model_id` and the labels consumed by the training package. `--renders-per-model 2` is a reasonable baseline; `--no-export-models` skips STL/OBJ export. The training split is grouped by `model_id` so multiple views of one model cannot leak into different subsets.
