from pathlib import Path
import json
from .geometry import create_parametric_mesh


def export_model(params, output_dir, stem="hubcap"):
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    mesh, sanitized = create_parametric_mesh(params)
    stl, glb = out / f"{stem}.stl", out / f"{stem}.glb"
    mesh.export(stl)
    scene = mesh.scene()
    scene.units = "mm"
    scene.export(glb)
    metadata = out / f"{stem}_parameters.json"
    metadata.write_text(json.dumps({"parameters": sanitized, "units": "mm",
                                    "note": "Conceptual parametric mesh. Inspect in CAD; not manufacturing-certified."},
                                   ensure_ascii=False, indent=2), encoding="utf-8")
    return {"stl": str(stl), "glb": str(glb), "parameters": str(metadata)}
