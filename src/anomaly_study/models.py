"""Small CPU models; equal classifier architecture/optimizer/epoch budgets."""

from copy import deepcopy
from time import perf_counter_ns

import numpy as np
import torch
from torch import nn


class Autoencoder(nn.Module):
    def __init__(self, dimension, hidden, latent):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(dimension, hidden), nn.ReLU(), nn.Linear(hidden, latent)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent, hidden), nn.ReLU(), nn.Linear(hidden, dimension)
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


def classifier(dimension, hidden, classes):
    return nn.Sequential(nn.Linear(dimension, hidden), nn.ReLU(), nn.Linear(hidden, classes))


def representation(x, z, kind):
    if kind == "raw":
        return x
    if kind == "encoded":
        return z
    if kind == "concat":
        return torch.cat([x, z], dim=1)
    raise ValueError(f"Unknown representation: {kind}")


def train_model(
    model, x, target, val_x, val_target, config, seed, reconstruction=False, class_weights=None
):
    """Fixed full budget; select checkpoint with lowest validation loss, never test loss."""
    if len(x) == 0 or len(val_x) == 0:
        raise ValueError("Training and validation data must be nonempty")
    generator = torch.Generator().manual_seed(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    loss_fn = nn.MSELoss() if reconstruction else nn.CrossEntropyLoss(weight=class_weights)
    best_loss, best_epoch, best_state = float("inf"), None, None
    history = []
    for epoch in range(config["epochs"]):
        model.train()
        order = torch.randperm(len(x), generator=generator)
        for idx in order.split(config["batch_size"]):
            optimizer.zero_grad()
            loss = loss_fn(model(x[idx]), target[idx])
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.inference_mode():
            val_loss = float(loss_fn(model(val_x), val_target))
        history.append(val_loss)
        if val_loss < best_loss:
            best_loss, best_epoch = val_loss, epoch + 1
            best_state = deepcopy(model.state_dict())
    if best_state is None:
        raise ValueError("Training produced no finite validation checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    return {
        "selected_epoch": best_epoch,
        "validation_loss": best_loss,
        "history": history,
        "training_samples": len(x),
        "epochs_run": config["epochs"],
        "optimizer_steps": config["epochs"] * int(np.ceil(len(x) / config["batch_size"])),
    }


def parameter_count(model):
    return sum(p.numel() for p in model.parameters())


def benchmark(fn, batch_size, repeats=100):
    """CPU wall clock, pre-materialized inputs, inference_mode, no I/O or preprocessing."""
    with torch.inference_mode():
        for _ in range(20):
            fn()
        elapsed = []
        for _ in range(repeats):
            start = perf_counter_ns()
            fn()
            elapsed.append((perf_counter_ns() - start) / 1e6)
    return {
        "batch_size": batch_size,
        "warmup": 20,
        "repeats": repeats,
        "median_ms_per_batch": float(np.median(elapsed)),
        "p95_ms_per_batch": float(np.quantile(elapsed, 0.95)),
        "median_us_per_window": float(np.median(elapsed) * 1000 / batch_size),
    }
