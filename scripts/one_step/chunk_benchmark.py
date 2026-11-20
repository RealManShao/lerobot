"""Shared episode-split data loading and action metrics for the LIBERO chunk experiments."""

import torch
from cache_libero import TASKS
from torch.nn.functional import one_hot


def dataset(root, split, n_tasks=len(TASKS)):
    datasets = [torch.load(root / f"{i}-{split}.pt", weights_only=True) for i in range(n_tasks)]
    windows = []
    for i, item in enumerate(datasets):
        vision, state, actions, episodes = item["data"]
        # Supply task ID explicitly because the frozen encoder supplies only visual tokens.
        task = torch.full((len(state),), i, dtype=torch.long)
        state = torch.cat([state, one_hot(task, n_tasks).float()], dim=1)
        windows.append((vision, state, actions, task, episodes))
    return tuple(torch.cat(parts).cuda() for parts in zip(*windows, strict=True)), datasets


def fit_mask(data, records, reserved):
    """Reserve the last demonstrations in the shuffled training list for fitting the router."""
    mask = torch.zeros_like(data[3], dtype=torch.bool)
    for task, record in enumerate(records):
        names = record["demos"]
        if not 0 <= reserved < len(names):
            raise ValueError("Reservation must leave at least one fitting demonstration")
        numbers = [int(name.split("_")[-1]) for name in names[: len(names) - reserved]]
        mask |= (data[3] == task) & torch.isin(data[4], data[4].new_tensor(numbers))
    return mask


def make_batch(data, indices):
    vision, state, target = [tensor[indices] for tensor in data[:3]]
    valid = torch.ones(vision.shape[:2], device=vision.device, dtype=torch.bool)
    return (
        {"backbone_features": vision.float(), "backbone_attention_mask": valid, "image_mask": valid},
        {
            "state": state[:, None],
            "action": target,
            "action_mask": torch.ones_like(target),
            "embodiment_id": torch.zeros(len(indices), dtype=torch.long, device=state.device),
        },
    )


def match_events(truth, prediction, tolerance=1):
    """Maximum ordered, one-to-one matching within tolerance for one switch direction."""
    i = j = matched = 0
    while i < len(truth) and j < len(prediction):
        if prediction[j] < truth[i] - tolerance:
            j += 1
        elif prediction[j] > truth[i] + tolerance:
            i += 1
        else:
            matched += 1
            i += 1
            j += 1
    return matched


@torch.inference_mode()
def score(head, data, records):
    head.eval()
    result = []
    for i, item in enumerate(records):
        indices = (data[3] == i).nonzero().flatten()
        predictions = []
        for batch_ids in indices.split(64):
            obs, batch = make_batch(data, batch_ids)
            del batch["action"]
            predictions.append(head.get_action(obs, batch)["action_pred"])
        pred = torch.cat(predictions)
        target = data[2][indices]
        err = (pred[:, :, :6] - target[:, :, :6]).square()
        if not torch.isfinite(pred).all():
            raise ValueError("Non-finite validation prediction")
        expected = (target[:, :, 6] > 0).int().diff(dim=1).cpu().numpy()
        observed = (pred[:, :, 6] > 0).int().diff(dim=1).cpu().numpy()
        hits = 0
        for expert, policy in zip(expected, observed, strict=True):
            for sign in (-1, 1):
                hits += match_events((expert == sign).nonzero()[0], (policy == sign).nonzero()[0])
        n_expert = int((expected != 0).sum())
        n_pred = int((observed != 0).sum())
        result.append(
            {
                "task": item["task"],
                "validation_demos": item["demos"],
                "windows": len(indices),
                "arm_first5_mse": err[:, :5].mean().item(),
                "arm_last20_mse": err[:, 20:].mean().item(),
                "arm_diff_mse": (pred[:, :, :6].diff(dim=1) - target[:, :, :6].diff(dim=1))
                .square()
                .mean()
                .item(),
                "gripper_accuracy": ((target[:, :, 6] > 0) == (pred[:, :, 6] > 0)).float().mean().item(),
                "switch_f1": 2 * hits / max(1, n_expert + n_pred),
                "matched_events": hits,
                "expert_events": n_expert,
                "predicted_events": n_pred,
            }
        )
    return result
