"""Score consequence metrics on random directions unused by the coordinate probes."""

import argparse
import json
from pathlib import Path

import numpy as np
from probe import DIM, FIT, RADII
from scipy.stats import spearmanr


def jacobian(row, mask):
    return np.stack([((np.array(r["plus"]) - r["minus"]) / (2 * FIT))[mask] for r in row["fit"]], axis=1)


def mask_for(row, outcome):
    size = len(row["fit"][0]["plus"])
    arm = np.tile(np.arange(size // 2) < 3, 2)
    return {"joint": np.ones(size, dtype=bool), "object": ~arm, "eef": arm}[outcome]


def energy(r, mask):
    return (
        np.sum(np.square(np.asarray(r["plus"])[mask])) + np.sum(np.square(np.asarray(r["minus"])[mask]))
    ) / (2 * r["eps"] ** 2)


def rmse(pred, truth):
    return float(np.sqrt(np.mean((pred - truth) ** 2)) / np.mean(truth))


def rho(x, y):
    if np.ptp(x) < 1e-12 * max(np.max(np.abs(x)), 1e-20) or np.ptp(y) == 0:
        return None
    return float(spearmanr(x, y).statistic)


def analyze(row, donor, weights, contrast, outcome, scale_data):
    mask = mask_for(row, outcome)
    jac = jacobian(row, mask)
    gram = jac.T @ jac
    eig, vec = np.linalg.eigh(gram)
    trace = np.trace(gram)
    basis = np.array(row["basis"])
    flat = basis.reshape(DIM, -1)
    iso = flat @ flat.T
    diagonal = np.mean(weights["weights"], axis=0).reshape(-1)
    idp = (flat * diagonal) @ flat.T
    # Restrict the fixed 5/20-step mean operators to this H=16 perturbation;
    # coordinates of the original 40-step chunk after step 16 have zero error.
    blocks = np.pad(basis, ((0, 0), (0, 40 - basis.shape[1]), (0, 0)))
    block_metric = idp / (2 * 40 * 7)
    for size in (5, 20):
        means = blocks.reshape(DIM, 40 // size, size, 3).mean(axis=2).reshape(DIM, -1)
        block_metric += 0.25 * (means @ means.T) / means.shape[1]
    matrix = jacobian(donor, mask_for(donor, outcome))
    matrices = {
        "isotropic": iso,
        "expert": idp,
        "expert_blocks": block_metric,
        "physics_diagonal": np.diag(gram.diagonal()),
        "physics_rank3": (vec[:, -3:] * eig[-3:].clip(0)) @ vec[:, -3:].T,
        "physics_full": gram,
        "other_demo": matrix.T @ matrix,
    }
    result = {
        "task": row["task"],
        "demo": row["demo"],
        "phase": row["phase"],
        "outcome": outcome,
        "start": row["start"],
        "base_contact_steps": row["base_contact_steps"],
        "reference_movement_um": float(1e6 * FIT * np.sqrt(trace / DIM)),
        "eef_restore_error_m": row["eef_restore_error_m"],
        "basis_replay_error": row["replay_error"],
        "rank90": int(np.searchsorted(np.cumsum(eig[::-1].clip(0)), 0.9 * trace) + 1) if trace > 0 else None,
        "fraction_offdiag_sq": float(
            np.linalg.norm(gram - np.diag(gram.diagonal())) ** 2 / max(np.linalg.norm(gram) ** 2, 1e-50)
        ),
        "tests": {},
    }
    for radius in RADII:
        tests = [r for r in row["tests"] if r["eps"] == radius]
        v = np.array([r["v"] for r in tests])
        truth = np.array([energy(r, mask) for r in tests])
        if truth.mean() == 0:
            continue
        scores = {}
        probes = scale_data["basis_probes"][str(radius)]
        matched_j = np.stack(
            [(np.array(s["plus"]) - s["minus"])[mask] / (2 * radius) for s in probes], axis=1
        )
        for name, g in {**matrices, "matched_radius": matched_j.T @ matched_j}.items():
            estimate = np.einsum("ij,jk,ik->i", v, g, v)
            corr = rho(estimate, truth)
            if name in ("isotropic", "expert", "expert_blocks"):
                estimate *= trace / max(np.trace(g), 1e-30)
            scores[name] = {"energy_nrmse": rmse(estimate, truth), "spearman": corr}
        plus = np.array([r["plus"] for r in tests])[:, mask]
        minus = np.array([r["minus"] for r in tests])[:, mask]
        scores["field"] = {
            "relative_vector_error": float(
                np.linalg.norm((plus - minus) / 2 - radius * v @ jac.T)
                / max(np.linalg.norm((plus - minus) / 2), 1e-30)
            ),
            "even_response_fraction": float(
                np.sqrt(np.mean(np.sum(((plus + minus) / 2) ** 2, axis=1)) / (radius**2 * truth.mean()))
            ),
            "changed_contacts": sum(r["contact_changed"] for r in tests),
        }
        result["tests"][str(radius)] = scores
    cases = contrast["contrasts"]
    result["sign_pairs"] = [
        [energy(cases[f"{axis}_{s}"][0], mask) for s in ("same", "alternating")] for axis in range(3)
    ]
    result["extreme_displacements_um"] = {
        s: [1e6 * r["eps"] * np.sqrt(energy(r, mask)) for r in cases[s]]
        for s in ("most_sensitive", "least_sensitive")
    }
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    p.add_argument("out", type=Path)
    args = p.parse_args()
    raw = {f.stem: json.loads(f.read_text()) for f in (args.root / "raw").glob("*.json")}
    results = []
    for key, row in sorted(raw.items()):
        task, demo, phase = key.split("--")
        donor = raw["--".join([task, "demo_1" if demo == "demo_0" else "demo_0", phase])]
        weights = json.loads((args.root / "idp" / (task + ".json")).read_text())["anchors"][
            demo + "--" + phase
        ]
        assert weights["start"] == row["start"]
        contrasts = json.loads((args.root / "contrasts" / (key + ".json")).read_text())
        scale_data = json.loads((args.root / "scales" / (key + ".json")).read_text())
        assert scale_data["cross_run_error"] <= 1e-9
        for outcome in ("joint", "object", "eef"):
            results.append(analyze(row, donor, weights, contrasts, outcome, scale_data))
    assert len(raw) == 60 and len({r["task"] for r in results}) == 10
    args.out.write_text(json.dumps(results, separators=(",", ":"), allow_nan=False))


if __name__ == "__main__":
    main()
