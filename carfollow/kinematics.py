"""The one update rule for moving a car forward by 0.1 s.

The table is built so that feeding the real accel_mps2 into `advance`
reproduces the real position and speed exactly. Any simulator must use this
rule, otherwise even a perfect model gets a non-zero Drive error.
"""

from .schema import DT


def advance(position_m: float, speed_mps: float, accel_mps2: float, dt: float = DT) -> tuple[float, float]:
    """Return (position, speed) after one step. Position uses the old speed."""
    return position_m + speed_mps * dt, max(speed_mps + accel_mps2 * dt, 0.0)
