"""
OceanEmbed — Training Pipeline
==============================
End-to-end training loop with cosine LR schedule, early stopping,
checkpointing, and metric logging.

Usage
-----
    python -m src.train --config configs/default.yaml

Or programmatically::

    from src.train import Trainer
    trainer = Trainer(cfg)
    trainer.run()
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

try:
    import yaml
except ImportError:
    yaml = None  # type: ignore

from src.data import SyntheticOceanDataset
from src.model import SatEmbedNet


# ------------------------------------------------------------------ #
def load_config(path: str) -> dict:
    with open(path) as f:
        if yaml is not None:
            return yaml.safe_load(f)
        return json.load(f)


# ------------------------------------------------------------------ #
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device,
             depths=None) -> Dict[str, float]:
    """Compute per-depth RMSE, correlation, bias and overall loss."""
    model.eval()
    all_pred, all_true = [], []
    criterion = nn.MSELoss()
    total_loss = 0.0
    n = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)["temp_profile"]
            loss = criterion(out, y)
            total_loss += loss.item() * x.size(0)
            n += x.size(0)
            all_pred.append(out.cpu().numpy())
            all_true.append(y.cpu().numpy())

    preds = np.concatenate(all_pred)
    trues = np.concatenate(all_true)
    n_depths = preds.shape[1]

    rmse = np.sqrt(np.mean((preds - trues) ** 2, axis=0))
    bias = np.mean(preds - trues, axis=0)
    corr = np.array([
        float(np.corrcoef(preds[:, i], trues[:, i])[0, 1])
        for i in range(n_depths)
    ])

    return {
        "loss": total_loss / max(n, 1),
        "rmse_mean": float(rmse.mean()),
        "rmse_per_depth": rmse.tolist(),
        "corr_mean": float(np.nanmean(corr)),
        "corr_per_depth": corr.tolist(),
        "bias_mean": float(bias.mean()),
        "bias_per_depth": bias.tolist(),
    }


# ------------------------------------------------------------------ #
class Trainer:
    def __init__(self, cfg: dict, output_dir: str = "outputs"):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        tc = cfg["training"]
        self.epochs = tc["epochs"]
        self.lr = tc["learning_rate"]
        self.weight_decay = tc["weight_decay"]
        self.patience = tc["early_stopping_patience"]
        self.seed = tc["seed"]

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        self._build_data()
        self._build_model()

    # -- data -------------------------------------------------------
    def _build_data(self):
        tc = self.cfg["training"]
        ds_cls = SyntheticOceanDataset if self.cfg.get("data_source", "synthetic") == "synthetic" else None
        if ds_cls is None:
            raise ValueError("Only 'synthetic' data source is wired in this MVP.")

        self.train_ds = ds_cls(self.cfg, split="train")
        self.val_ds = ds_cls(self.cfg, split="val")
        self.test_ds = ds_cls(self.cfg, split="test")

        self.train_loader = DataLoader(self.train_ds, batch_size=tc["batch_size"], shuffle=True)
        self.val_loader = DataLoader(self.val_ds, batch_size=tc["batch_size"], shuffle=False)
        self.test_loader = DataLoader(self.test_ds, batch_size=tc["batch_size"], shuffle=False)

    # -- model ------------------------------------------------------
    def _build_model(self):
        self.model = SatEmbedNet(self.cfg).to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        steps = self.epochs * len(self.train_loader)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=steps
        )

    # -- loop -------------------------------------------------------
    def run(self) -> Dict:
        print(f"OceanEmbed — training on {self.device}")
        print(f"  train={len(self.train_ds)}  val={len(self.val_ds)}  test={len(self.test_ds)}")

        best_val = float("inf")
        best_epoch = 0
        history = []

        for epoch in range(1, self.epochs + 1):
            t0 = time.time()
            self.model.train()
            train_loss = 0.0
            for x, y in self.train_loader:
                x, y = x.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()
                out = self.model(x)["temp_profile"]
                loss = self.criterion(out, y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                self.scheduler.step()
                train_loss += loss.item() * x.size(0)
            train_loss /= len(self.train_ds)

            val_metrics = evaluate(self.model, self.val_loader, self.device)
            dt = time.time() - t0

            print(
                f"Epoch {epoch:3d}/{self.epochs}  "
                f"train_loss={train_loss:.4f}  val_loss={val_metrics['loss']:.4f}  "
                f"val_rmse={val_metrics['rmse_mean']:.3f}°C  val_corr={val_metrics['corr_mean']:.3f}  "
                f"({dt:.1f}s)"
            )

            history.append({
                "epoch": epoch,
                "train_loss": train_loss,
                **{k: v for k, v in val_metrics.items() if k != "rmse_per_depth" and k != "corr_per_depth" and k != "bias_per_depth"},
            })

            if val_metrics["loss"] < best_val:
                best_val = val_metrics["loss"]
                best_epoch = epoch
                torch.save(
                    {"model_state": self.model.state_dict(), "cfg": self.cfg, "epoch": epoch},
                    self.output_dir / "best_model.pt",
                )
            elif epoch - best_epoch >= self.patience:
                print(f"  Early stopping at epoch {epoch} (no improvement for {self.patience} epochs).")
                break

        # test
        ckpt = torch.load(self.output_dir / "best_model.pt", weights_only=False)
        self.model.load_state_dict(ckpt["model_state"])
        test_metrics = evaluate(self.model, self.test_loader, self.device,
                                depths=self.cfg["target"]["depths"])

        print("\n=== Test set ===")
        print(f"  RMSE (mean)  = {test_metrics['rmse_mean']:.3f} °C")
        print(f"  Corr (mean)  = {test_metrics['corr_mean']:.3f}")
        print(f"  Bias (mean)  = {test_metrics['bias_mean']:.3f} °C")

        # save metrics
        results = {
            "best_epoch": best_epoch,
            "history": history,
            "test_metrics": test_metrics,
            "depths": self.cfg["target"]["depths"],
        }
        with open(self.output_dir / "metrics.json", "w") as f:
            json.dump(results, f, indent=2)

        return results


# ------------------------------------------------------------------ #
def main():
    parser = argparse.ArgumentParser(description="Train OceanEmbed MVP")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--output_dir", default="outputs")
    args = parser.parse_args()

    cfg = load_config(args.config)
    trainer = Trainer(cfg, output_dir=args.output_dir)
    trainer.run()


if __name__ == "__main__":
    main()
