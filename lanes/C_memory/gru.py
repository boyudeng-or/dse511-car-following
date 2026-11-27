# Intended to be run by the scorer
"""Lane C: GRU car-following model wrapped in the shared plug

from lanes.C_memory.gru import load_model
model = load_model("lanes/C_memory/outputs/gru_w30_0_handoff.pt)
model.accel(history) e.g., (30, 3) array of [gap_m, speed_mps, closing_speed_mps]
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from carfollow.baselines import MAX_BRAKE_MPS2

HIDDEN = 32


class GRUFollower(nn.Module):
    def __init__(self, hidden=HIDDEN):
        super().__init__()
        self.gru = nn.GRU(input_size=3, hidden_size=hidden, batch_first=True)
        # making 1 prediction
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        out, _ = self.gru(x)
        return self.head(out[:, -1]).squeeze(-1)


class GRUDriver:
    "Plug needed for the trained GRU so it can be ran and scored. Will be given raw histories and using mean and std will be normalized. Will look at only last lookback amount of rows and returns accelerations"

    def __init__(self, net, mean, std, lookback, name=None):
        self.net = net.eval()
        self.mean = np.asarray(mean, dtype=np.float32)
        self.std = np.asarray(std, dtype=np.float32)
        self.lookback = lookback
        self.name = name or f"gru_w{lookback}"

    def accel_batch(self, histories):
        """Batching multiple histories at once: (n, 30, 3) returning n accelerations

        Scorer calls this when available
        """

        x = np.asarray(histories, dtype=np.float32)
        # keep newest lookback rows
        x = x[:, -self.lookback :, :]
        x = (x - self.mean) / self.std
        with torch.no_grad():
            a = self.net(torch.from_numpy(x)).numpy()
        return np.clip(a, -MAX_BRAKE_MPS2, MAX_BRAKE_MPS2)  # within physical limits

    def accel(self, history):
        "Runs accel_batch but with a batch of 1"
        batch_one = np.asarray(history)[None]
        return float(self.accel_batch(batch_one)[0])


def save_model(driver, path):
    """This saves everything we need to rebuild the GRU driver in one file, since we also need the mean, std"""
    torch.save(
        {
            "state_dict": driver.net.state_dict(),
            "hidden": driver.net.gru.hidden_size,
            "mean": driver.mean.tolist(),
            "std": driver.std.tolist(),
            "lookback": driver.lookback,
            "name": driver.name,
        },
        Path(path),
    )


def load_model(path):
    """Loads a driver model from a specified path and rebuild so that it is ready to use"""
    saved = torch.load(Path(path))
    net = GRUFollower(hidden=saved["hidden"])
    net.load_state_dict(saved["state_dict"])
    return GRUDriver(net, saved["mean"], saved["std"], saved["lookback"], saved["name"])
