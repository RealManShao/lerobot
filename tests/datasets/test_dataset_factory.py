from types import SimpleNamespace

import torch

from lerobot.datasets import factory
from lerobot.processor.converters import batch_to_transition
from lerobot.utils.constants import IMAGENET_STATS


def test_make_dataset_adds_missing_camera_stats(monkeypatch):
    metadata = SimpleNamespace(features={}, camera_keys=["image"], depth_keys=[])
    dataset = SimpleNamespace(meta=SimpleNamespace(camera_keys=["image"], depth_keys=[], stats={}))

    monkeypatch.setattr(factory, "LeRobotDatasetMetadata", lambda *args, **kwargs: metadata)
    monkeypatch.setattr(factory, "LeRobotDataset", lambda *args, **kwargs: dataset)

    dataset_config = SimpleNamespace(
        repo_id="lerobot/robomme",
        root=None,
        episodes=None,
        feature_rename_map={},
        image_transforms=SimpleNamespace(enable=False),
        revision=None,
        use_imagenet_stats=True,
        video_backend="torchcodec",
        depth_output_unit="m",
        streaming=False,
    )
    trainable_config = SimpleNamespace(
        reward_delta_indices=None,
        action_delta_indices=None,
        observation_delta_indices=None,
    )
    config = SimpleNamespace(dataset=dataset_config, trainable_config=trainable_config, tolerance_s=1e-4)

    result = factory.make_dataset(config)

    assert result is dataset
    for stats_type, stats in IMAGENET_STATS.items():
        torch.testing.assert_close(result.meta.stats["image"][stats_type], torch.tensor(stats))


def test_feature_renamed_dataset_exposes_canonical_batch_and_metadata():
    class SourceDataset:
        meta = SimpleNamespace(
            features={"image": {"dtype": "image"}, "actions": {"dtype": "float32"}},
            stats={"actions": {"mean": torch.zeros(2)}},
            camera_keys=["image"],
            depth_keys=[],
            has_language_columns=False,
        )

        def __len__(self):
            return 1

        def __getitem__(self, index):
            return {
                "image": torch.zeros(3, 8, 8),
                "actions": torch.zeros(4, 2),
            }

    dataset = factory._FeatureRenamedDataset(
        SourceDataset(),
        {"image": "observation.images.image", "actions": "action"},
    )
    transition = batch_to_transition(dataset[0])

    assert dataset.meta.camera_keys == ["observation.images.image"]
    assert "observation.images.image" in dataset.meta.features
    assert "action" in dataset.meta.stats
    assert transition["observation"]["observation.images.image"].shape == (3, 8, 8)
    assert transition["action"].shape == (4, 2)
