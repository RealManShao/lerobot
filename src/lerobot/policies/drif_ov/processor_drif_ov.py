#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any

import torch

from lerobot.policies.groot.processor_groot import (
    make_groot_pre_post_processors,
    make_groot_pre_post_processors_from_pretrained,
)

from .configuration_drif_ov import DrifOvConfig


def make_drif_ov_pre_post_processors(
    config: DrifOvConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
    dataset_meta: Any | None = None,
):
    return make_groot_pre_post_processors(
        config=config,
        dataset_stats=dataset_stats,
        dataset_meta=dataset_meta,
    )


def make_drif_ov_pre_post_processors_from_pretrained(
    config: DrifOvConfig,
    pretrained_path: str,
    **kwargs: Any,
):
    return make_groot_pre_post_processors_from_pretrained(
        config=config,
        pretrained_path=pretrained_path,
        **kwargs,
    )
