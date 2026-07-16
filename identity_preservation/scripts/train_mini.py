"""
Overnight training script for MiniGainNet.

Runs on MPS (MacBook), CUDA (RunPod), or CPU (CI). Prints loss every step,
saves loss curve and periodic samples.

Usage:
    conda activate momask
    cd ~/identity_preservation_mini
    python scripts/train_mini.py

Overnight schedule:
    - 200 epochs on ~10 clips × 20 augmentations × pair combinations
    - ~5000-10000 steps depending on pair count
    - Expected ~4-6 hours on MPS

Output:
    outputs/checkpoints/mini_gain_net_epoch_XXX.pt  (every 20 epochs)
    outputs/checkpoints/mini_gain_net_final.pt
    outputs/logs/loss_curve.png
    outputs/logs/loss_history.csv
"""
from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))
from config_mini import cfg
from models.mini_gain_net import MiniGainNet, count_params
from data.mini_pair_dataset import MiniPairDataset


def joint_reconstruction_loss(
    blended: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """
    L2 loss between blended and target joint positions.
    Both tensors are (B, T, J, 3).
    """
    return ((blended - target) ** 2).mean()


def identity_regularization(
    alpha: torch.Tensor,
    spectrum_delta: torch.Tensor,
) -> torch.Tensor:
    """
    Encourage alpha to be high when spectrum_delta is low (identity pair),
    and lower when spectrum_delta is high.

    Target: alpha ≈ (1 - spectrum_delta), clamped to [0.1, 0.95]
    """
    B = alpha.shape[0]
    # Reduce alpha to (B,) mean
    alpha_mean = alpha.view(B, -1).mean(dim=1)
    # Target: high alpha when delta small, low when delta large
    target_alpha = (1.0 - spectrum_delta).clamp(0.1, 0.95)
    return ((alpha_mean - target_alpha) ** 2).mean()


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
    epoch: int,
) -> dict:
    model.train()
    losses = {"recon": [], "id_reg": [], "total": [], "alpha_mean": []}

    for step, batch in enumerate(loader):
        source = batch["source"].to(device)
        target = batch["target"].to(device)
        delta = batch["spectrum_delta"].to(device)

        alpha, blended = model(source, target)
        # 'target' from dataset is the higher-spectrum motion.
        # blended = alpha * source + (1-alpha) * target
        # We want blended to reconstruct... what? Ideally something that
        # keeps identity of source but has expressive content of target.
        # For proof-of-architecture: reconstruct the target (main signal)
        # + a soft regularizer pushing alpha toward (1 - spectrum_delta).
        l_recon = joint_reconstruction_loss(blended, target)
        l_id = identity_regularization(alpha, delta)
        total = l_recon + 0.1 * l_id

        optimizer.zero_grad()
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.GRAD_CLIP)
        optimizer.step()

        losses["recon"].append(l_recon.item())
        losses["id_reg"].append(l_id.item())
        losses["total"].append(total.item())
        losses["alpha_mean"].append(alpha.mean().item())

    return {k: float(np.mean(v)) for k, v in losses.items()}


def save_checkpoint(model, optimizer, epoch, loss_history, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "loss_history": loss_history,
        "config": {k: str(v) for k, v in cfg.__dict__.items()},
    }, str(path))


def save_loss_curve(loss_history: list[dict], out_path: Path) -> None:
    if not loss_history:
        return
    epochs = [x["epoch"] for x in loss_history]
    total = [x["total"] for x in loss_history]
    recon = [x["recon"] for x in loss_history]
    alpha = [x["alpha_mean"] for x in loss_history]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(epochs, total, label="total", linewidth=2)
    ax1.plot(epochs, recon, label="recon", linewidth=1, alpha=0.7)
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("loss")
    ax1.set_title("MiniGainNet training loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, alpha, color="teal", linewidth=2)
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("mean α")
    ax2.set_title("Mean identity gain over time")
    ax2.set_ylim(0, 1)
    ax2.grid(True, alpha=0.3)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(str(out_path), dpi=120, bbox_inches="tight")
    plt.close()


def save_loss_csv(loss_history: list[dict], out_path: Path) -> None:
    if not loss_history:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=loss_history[0].keys())
        writer.writeheader()
        writer.writerows(loss_history)


def main() -> None:
    torch.manual_seed(cfg.SEED)
    np.random.seed(cfg.SEED)

    # Device
    if cfg.DEVICE == "mps" and torch.backends.mps.is_available():
        device = "mps"
    elif cfg.DEVICE == "cuda" and torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"[train] device: {device}")

    # Data
    dataset = MiniPairDataset(cfg)
    loader = DataLoader(
        dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=True,
        num_workers=0,       # MPS doesn't play well with multi-worker
        pin_memory=(device == "cuda"),
    )

    # Model
    model = MiniGainNet(cfg).to(device)
    n_params = count_params(model)
    print(f"[train] MiniGainNet params: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.LR, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.NUM_EPOCHS, eta_min=cfg.LR * 0.1
    )

    loss_history = []
    start_time = time.time()
    print(f"[train] starting {cfg.NUM_EPOCHS} epochs at {time.strftime('%H:%M:%S')}")
    print()

    for epoch in range(1, cfg.NUM_EPOCHS + 1):
        epoch_start = time.time()
        losses = train_one_epoch(model, loader, optimizer, device, epoch)
        scheduler.step()
        elapsed = time.time() - epoch_start

        record = {"epoch": epoch, "elapsed_s": round(elapsed, 2), **losses}
        loss_history.append(record)

        if epoch % 5 == 0 or epoch == 1:
            print(
                f"[ep {epoch:3d}/{cfg.NUM_EPOCHS}] "
                f"total={losses['total']:.4f} "
                f"recon={losses['recon']:.4f} "
                f"id_reg={losses['id_reg']:.4f} "
                f"α={losses['alpha_mean']:.3f} "
                f"({elapsed:.1f}s)"
            )

        # Periodic checkpoint
        if epoch % cfg.SAMPLE_EVERY_N_EPOCHS == 0:
            ckpt_path = cfg.CHECKPOINTS_ROOT / f"mini_gain_net_epoch_{epoch:03d}.pt"
            save_checkpoint(model, optimizer, epoch, loss_history, ckpt_path)
            save_loss_curve(loss_history, cfg.LOGS_ROOT / "loss_curve.png")
            save_loss_csv(loss_history, cfg.LOGS_ROOT / "loss_history.csv")
            print(f"        ✓ saved checkpoint {ckpt_path.name}")

    # Final checkpoint
    save_checkpoint(model, optimizer, cfg.NUM_EPOCHS, loss_history,
                    cfg.CHECKPOINTS_ROOT / "mini_gain_net_final.pt")
    save_loss_curve(loss_history, cfg.LOGS_ROOT / "loss_curve.png")
    save_loss_csv(loss_history, cfg.LOGS_ROOT / "loss_history.csv")

    total_time = time.time() - start_time
    print()
    print(f"[train] ✓ done in {total_time / 60:.1f} min")
    print(f"[train] final loss: {loss_history[-1]['total']:.4f}")
    print(f"[train] final α mean: {loss_history[-1]['alpha_mean']:.3f}")
    print(f"[train] outputs at: {cfg.OUTPUTS_ROOT}")


if __name__ == "__main__":
    main()
