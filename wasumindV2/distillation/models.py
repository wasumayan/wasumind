"""
Student architectures for continuous-action policy distillation.
All models: (B, T, obs_dim) → (B, T, act_dim)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Re-use spectral filters from existing codebase
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from spectramem.models.memory.spectral_filters import get_spectral_filters, apply_filter_fft


class GRUStudent(nn.Module):
    def __init__(self, obs_dim, act_dim, d_model=64, n_layers=2):
        super().__init__()
        self.input_proj = nn.Linear(obs_dim, d_model)
        self.gru = nn.GRU(d_model, d_model, num_layers=n_layers, batch_first=True)
        self.output_proj = nn.Linear(d_model, act_dim)

    def forward(self, obs):
        h = self.input_proj(obs)
        h, _ = self.gru(h)
        return self.output_proj(h)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class LSTMStudent(nn.Module):
    def __init__(self, obs_dim, act_dim, d_model=64, n_layers=2):
        super().__init__()
        self.input_proj = nn.Linear(obs_dim, d_model)
        self.lstm = nn.LSTM(d_model, d_model, num_layers=n_layers, batch_first=True)
        self.output_proj = nn.Linear(d_model, act_dim)

    def forward(self, obs):
        h = self.input_proj(obs)
        h, _ = self.lstm(h)
        return self.output_proj(h)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class TransformerStudent(nn.Module):
    def __init__(self, obs_dim, act_dim, d_model=64, n_layers=2, n_heads=4, window_size=64):
        super().__init__()
        self.input_proj = nn.Linear(obs_dim, d_model)
        self.pos_enc = nn.Parameter(torch.randn(1, 2048, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.output_proj = nn.Linear(d_model, act_dim)
        self.window_size = window_size

    def forward(self, obs):
        B, T, _ = obs.shape
        h = self.input_proj(obs) + self.pos_enc[:, :T, :]

        # Causal + sliding window mask
        mask = torch.ones(T, T, device=obs.device, dtype=torch.bool)
        mask = torch.triu(mask, diagonal=1)  # causal: can't attend to future
        for i in range(T):
            start = max(0, i - self.window_size + 1)
            mask[i, :start] = True  # can't attend beyond window
        mask = mask.float() * -1e9

        h = self.transformer(h, mask=mask)
        return self.output_proj(h)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class STUStudent(nn.Module):
    """Full STU (non-tensordot) student for continuous actions."""
    def __init__(self, obs_dim, act_dim, d_model=64, n_layers=2, num_filters=16, seq_len=1000):
        super().__init__()
        self.input_proj = nn.Linear(obs_dim, d_model)
        self.num_filters = num_filters
        self.d_model = d_model

        # Spectral filters (fixed, not learned)
        filters = get_spectral_filters(seq_len, num_filters)
        self.register_buffer("spectral_filters", filters)
        actual_k = filters.shape[0]

        # Learnable projection matrices per layer
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(nn.ModuleDict({
                "M_pos": nn.Linear(actual_k * d_model, d_model, bias=False),
                "M_neg": nn.Linear(actual_k * d_model, d_model, bias=False),
                "M_direct": nn.Linear(d_model, d_model),
                "norm": nn.LayerNorm(d_model),
                "mlp": nn.Sequential(
                    nn.Linear(d_model, d_model * 4),
                    nn.SiLU(),
                    nn.Linear(d_model * 4, d_model),
                ),
                "mlp_norm": nn.LayerNorm(d_model),
            }))

        self.output_proj = nn.Linear(d_model, act_dim)

    def forward(self, obs):
        B, L, _ = obs.shape
        h = self.input_proj(obs)  # (B, L, d_model)

        # Clamp filters to sequence length
        filters = self.spectral_filters
        K = filters.shape[0]
        if filters.shape[1] < L:
            # Pad filters
            pad = torch.zeros(K, L - filters.shape[1], device=filters.device)
            filters = torch.cat([filters, pad], dim=1)
        elif filters.shape[1] > L:
            filters = filters[:, :L]

        for layer in self.layers:
            residual = h

            # Spectral convolution via FFT
            u_pos = apply_filter_fft(h, filters)       # (B, L, K, d_model)
            neg_filters = filters * ((-1) ** torch.arange(filters.shape[1], device=filters.device)).unsqueeze(0)
            u_neg = apply_filter_fft(h, neg_filters)    # (B, L, K, d_model)

            # Flatten and project
            u_pos_flat = u_pos.reshape(B, L, -1)  # (B, L, K*d_model)
            u_neg_flat = u_neg.reshape(B, L, -1)

            h_spectral = layer["M_pos"](u_pos_flat) + layer["M_neg"](u_neg_flat) + layer["M_direct"](h)
            h = layer["norm"](residual + h_spectral)

            # MLP block
            residual = h
            h = layer["mlp_norm"](residual + layer["mlp"](h))

        return self.output_proj(h)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class MLPStudent(nn.Module):
    """Feedforward baseline — no temporal model, processes each timestep independently."""
    def __init__(self, obs_dim, act_dim, d_model=64, n_layers=2, **kwargs):
        super().__init__()
        layers = [nn.Linear(obs_dim, d_model), nn.SiLU()]
        for _ in range(n_layers - 1):
            layers.extend([nn.Linear(d_model, d_model), nn.SiLU()])
        layers.append(nn.Linear(d_model, act_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, obs):
        return self.net(obs)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class FrameStackStudent(nn.Module):
    """Stack last k frames and feed to MLP. Simplest history-based baseline."""
    def __init__(self, obs_dim, act_dim, d_model=64, n_layers=2, frame_stack=8, **kwargs):
        super().__init__()
        self.frame_stack = frame_stack
        input_dim = obs_dim * frame_stack
        layers = [nn.Linear(input_dim, d_model), nn.SiLU()]
        for _ in range(n_layers - 1):
            layers.extend([nn.Linear(d_model, d_model), nn.SiLU()])
        layers.append(nn.Linear(d_model, act_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, obs):
        B, T, D = obs.shape
        k = self.frame_stack
        padded = torch.zeros(B, k - 1 + T, D, device=obs.device, dtype=obs.dtype)
        padded[:, k - 1:, :] = obs
        frames = [padded[:, i:i + T, :] for i in range(k)]
        stacked = torch.cat(frames, dim=-1)  # (B, T, k*D)
        return self.net(stacked)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


STUDENT_REGISTRY = {
    "gru": GRUStudent,
    "lstm": LSTMStudent,
    "transformer": TransformerStudent,
    "stu": STUStudent,
    "mlp": MLPStudent,
    "framestack": FrameStackStudent,
}


def create_student(arch, obs_dim, act_dim, d_model=64, n_layers=2, **kwargs):
    cls = STUDENT_REGISTRY[arch]
    return cls(obs_dim=obs_dim, act_dim=act_dim, d_model=d_model, n_layers=n_layers, **kwargs)


if __name__ == "__main__":
    # Smoke test all architectures
    B, T, obs_dim, act_dim = 4, 100, 8, 6
    x = torch.randn(B, T, obs_dim)

    for name, cls in STUDENT_REGISTRY.items():
        kwargs = {"seq_len": T} if name == "stu" else {}
        model = cls(obs_dim=obs_dim, act_dim=act_dim, d_model=64, n_layers=2, **kwargs)
        out = model(x)
        n_params = model.count_parameters()
        print(f"{name:>12}: params={n_params:>8,}  output={out.shape}")
        assert out.shape == (B, T, act_dim), f"Shape mismatch: {out.shape}"

    print("\nAll student architectures verified.")
