"""Lane A baselines: keep current speed, and the Intelligent Driver Model (IDM)."""

from dataclasses import dataclass

import numpy as np

MAX_BRAKE_MPS2 = 9.0


class KeepSpeed:
    """Never accelerates or brakes. The floor every model must beat."""

    name = "keep_speed"

    def accel(self, history: np.ndarray) -> float:
        return 0.0


@dataclass
class IDM:
    """Treiber, Hennecke & Helbing (2000).

    a = a_max * [1 - (v / v0)^delta - (s* / s)^2]
    s* = s0 + max(0, v*T + v*dv / (2*sqrt(a_max*b)))
    with s = gap, v = speed, dv = closing speed.
    """

    v0: float = 30.0      # desired speed, m/s
    T: float = 1.5        # desired time headway, s
    s0: float = 2.0       # jam distance, m
    a_max: float = 1.0    # max acceleration, m/s^2
    b: float = 1.5        # comfortable braking, m/s^2
    delta: float = 4.0
    name: str = "idm"

    @classmethod
    def textbook(cls) -> "IDM":
        return cls(name="idm_textbook")

    @classmethod
    def calibrated(cls) -> "IDM":
        """One shared set fit on pairs_full (see lanes/A_data_idm/idm_calibrated.json)."""
        return cls(
            v0=19.256901,
            T=0.525609,
            s0=1.504281,
            a_max=0.5,
            b=3.5,
            name="idm_calibrated",
        )

    def accel(self, history: np.ndarray) -> float:
        gap, v, dv = (float(x) for x in history[-1])
        if gap <= 0.0:
            return -MAX_BRAKE_MPS2
        s_star = self.s0 + max(0.0, v * self.T + v * dv / (2.0 * np.sqrt(self.a_max * self.b)))
        a = self.a_max * (1.0 - (max(v, 0.0) / self.v0) ** self.delta - (s_star / gap) ** 2)
        return float(max(a, -MAX_BRAKE_MPS2))
