"""Check a fitted controller from recorded robot observations without constructing a simulator."""

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np

from lerobot.policies.demonstration_chunk.observed_control import ObservedController


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for field in ("vision", "head", "switch", "demo", "out"):
        p.add_argument("--" + field, type=Path, required=True)
    a = p.parse_args()
    policy = ObservedController.from_checkpoints(a.vision, a.head, a.switch)
    calls = []
    hook = policy.head.register_forward_hook(lambda *_: calls.append(1))
    with h5py.File(a.demo) as f:
        sensors = f["data/demo_0/obs"]
        for i in range(10):
            state = np.concatenate([sensors[k][i] for k in ("ee_pos", "ee_ori", "gripper_states")])
            rgb = sensors["agentview_rgb"][i]
            wrist = sensors["eye_in_hand_rgb"][i]
            action = policy.act(rgb, wrist, state)
            assert action.shape == (7,) and np.isfinite(action).all() and (abs(action) <= 1).all()
    hook.remove()
    if len(calls) != 2 or policy.features.shape != (20,):
        raise ValueError("Incorrect planning frequency or observation features")
    try:
        policy.act(rgb, wrist, np.r_[state, 1.0])
    except ValueError:
        pass
    else:
        raise ValueError("Unexpected extra state input accepted")
    loaded = [name for name in ("libero", "mujoco", "robosuite") if name in sys.modules]
    a.out.write_text(
        json.dumps(
            {
                "control_ticks": 10,
                "head_evaluations": len(calls),
                "feature_count": 20,
                "simulator_modules_loaded": loaded,
                "extra_state_rejected": True,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
