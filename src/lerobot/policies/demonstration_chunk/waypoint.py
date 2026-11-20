"""Cartesian feedback for a waypoint held fixed between neural replans."""


def waypoint_translation(position, goal, inverse_motion, remaining_steps):
    """Compute a bounded translation command for row-vector NumPy or PyTorch inputs."""
    return (((goal - position) @ inverse_motion) / remaining_steps).clip(-1, 1)
