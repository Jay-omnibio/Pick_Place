"""
Environment module: Simulation and sensor interfaces.
"""

from .sensors import SensorSuite

try:
    # Importing MuJoCo can fail in minimal environments (CI, linting, docs builds).
    from .simulator import MujocoSimulator  # type: ignore

    __all__ = ["MujocoSimulator", "SensorSuite"]
except Exception:
    __all__ = ["SensorSuite"]
