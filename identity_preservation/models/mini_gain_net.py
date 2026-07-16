"""
Mini GainNet — simplified proof-of-architecture variant.

Operates DIRECTLY in joint space (B, T=196, J=22, 3) instead of RVQ
codebook embedding space. This is the "proof of architecture" version:
it validates that the gain-blending idea works, without requiring
263-dim HumanML3D feature conversion, RVQ encoding, or Ted's frozen
MoMask transformers.

Slide 20 framing: "Preliminary GainNet trained in joint space on Ted's
editing/ samples, demonstrating architectural viability. Full GainNet
#6 training in RVQ codebook embedding space is in progress on RunPod."

Input:
    source_joints:    (B, T, J, 3) — the "reference" motion
    generated_joints: (B, T, J, 3) — the "target" motion (higher spectrum)
Output:
    alpha: (B, T, 1, 1) — per-frame gain, broadcast to all joints/dims
    blended: (B, T, J, 3) = (1-alpha) * generated + alpha * source
"""
from __future__ import annotations

import torch
import torch.nn as nn


class MiniGainNet(nn.Module):
    """
    Per-frame learned gain in joint space.

    Analog of GainNetPerFrame from the full version, but with joint
    coordinates as the substrate instead of RVQ codebook embeddings.
    """

    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg
        J = cfg.NUM_JOINTS
        D = cfg.JOINT_DIM
        L = cfg.GAIN_LATENT_DIM

        # Flatten joint coords per frame: (B, T, J*D) → (B, T, L)
        self.source_proj = nn.Linear(J * D, L)
        self.generated_proj = nn.Linear(J * D, L)

        # Fuse: concat [src, gen] → project
        self.fuse = nn.Linear(2 * L, L)

        # Positional encoding
        self.pos_emb = nn.Parameter(torch.randn(1, cfg.NUM_FRAMES, L) * 0.02)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=L,
            nhead=cfg.GAIN_N_HEADS,
            dim_feedforward=2 * L,
            dropout=cfg.GAIN_DROPOUT,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=cfg.GAIN_N_LAYERS)

        # Output head: per-frame scalar gain
        self.head = nn.Sequential(
            nn.Linear(L, L // 2),
            nn.GELU(),
            nn.Linear(L // 2, 1),
        )

    def forward(
        self,
        source: torch.Tensor,     # (B, T, J, 3)
        generated: torch.Tensor,  # (B, T, J, 3)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns (alpha, blended):
            alpha:   (B, T, 1, 1) — per-frame gain in [0, 1]
            blended: (B, T, J, 3) — blended motion
        """
        B, T, J, D = source.shape
        src_flat = source.reshape(B, T, J * D)
        gen_flat = generated.reshape(B, T, J * D)

        h_src = self.source_proj(src_flat)
        h_gen = self.generated_proj(gen_flat)
        h = self.fuse(torch.cat([h_src, h_gen], dim=-1))
        h = h + self.pos_emb[:, :T, :]
        h = self.transformer(h)

        logits = self.head(h)                          # (B, T, 1)
        alpha = torch.sigmoid(logits).unsqueeze(-1)    # (B, T, 1, 1)

        blended = (1 - alpha) * generated + alpha * source
        return alpha, blended


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from config_mini import cfg

    model = MiniGainNet(cfg)
    print(f"MiniGainNet params: {count_params(model):,}")

    B, T, J, D = 2, cfg.NUM_FRAMES, cfg.NUM_JOINTS, cfg.JOINT_DIM
    src = torch.randn(B, T, J, D)
    gen = torch.randn(B, T, J, D)
    alpha, blended = model(src, gen)
    print(f"src:     {tuple(src.shape)}")
    print(f"gen:     {tuple(gen.shape)}")
    print(f"alpha:   {tuple(alpha.shape)}, range [{alpha.min():.3f}, {alpha.max():.3f}]")
    print(f"blended: {tuple(blended.shape)}")
