# __init__.py
"""Custom policy package for LeRobot."""

try:
    import lerobot  # noqa: F401
except ImportError:
    raise ImportError(
        "lerobot is not installed. Please install lerobot to use this policy package."
    )

from .configuration_my_custom_policy import MyCustomPolicyConfig
from .modeling_my_custom_policy import MyCustomPolicy
from .processor_my_custom_policy import make_my_custom_policy_pre_post_processors

__all__ = [
    "MyCustomPolicyConfig",
    "MyCustomPolicy",
    "make_my_custom_policy_pre_post_processors",
]