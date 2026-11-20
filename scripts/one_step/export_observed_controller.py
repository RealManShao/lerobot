"""Export the fitted head and switch coefficients with a simulation-independent manifest."""

import argparse
import hashlib
import json
import tarfile
import tempfile
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--heads", type=Path, required=True)
    p.add_argument("--switches", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        for s in (1, 42):
            src = a.heads / f"seed{s}"
            manifest = json.loads((src / "object_constant.json").read_text())
            out = root / f"seed{s}"
            out.mkdir()
            payload = (src / manifest["weights"]).read_bytes()
            digest = hashlib.sha256(payload).hexdigest()
            (out / "head.pt").write_bytes(payload)
            (out / "observed_head.json").write_text(
                json.dumps({"weights": "head.pt", "model": manifest["model"]}, indent=2) + "\n"
            )
            for mode in ("short", "delayed", "motion", "events", "bounded"):
                spec = json.loads((a.switches / f"seed{s}" / f"{mode}_switch.json").read_text())
                if spec["head_sha256"] != digest:
                    raise ValueError("Mismatched switch and head")
                (out / f"{mode}_switch.json").write_text(json.dumps(spec, indent=2) + "\n")
        dest = a.out.with_suffix(".tmp")
        with tarfile.open(dest, "w:gz") as tar:
            for file in sorted(root.glob("*/*")):
                tar.add(file, arcname=file.relative_to(root))
        dest.replace(a.out)


if __name__ == "__main__":
    main()
