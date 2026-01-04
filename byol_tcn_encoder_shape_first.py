# byol_tcn_encoder_shape_first.py
import os
import glob
import random
from dataclasses import dataclass, field
from typing import Tuple, List, Optional, Dict, Any

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


# -----------------------------
# Reproducibility (seed)
# -----------------------------
def seed_everything(seed: int = 42, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# -----------------------------
# Feature construction
# -----------------------------
def make_features_from_ohlcv(
    df: pd.DataFrame,
    window: int = 50,
    vol_lookback: int = 10,
    smoothing_window: int = 0,
) -> np.ndarray:
    """
    Output: (T, 2) array: [log_return(smoothed close), rolling_std(log_return)]
    """
    close = df["Close"].astype(float)

    if smoothing_window and smoothing_window > 1:
        close = close.rolling(window=smoothing_window, min_periods=1).mean()

    close = close.values
    lr = np.diff(np.log(close), prepend=np.nan)

    vol = pd.Series(lr).rolling(vol_lookback).std().values
    feats = np.stack([lr, vol], axis=1)
    return feats


# -----------------------------
# Augmentations for BYOL (channel-aware)
# -----------------------------
@dataclass
class AugmentConfig:
    max_time_shift: int = 4

    # return channel (shape)
    ret_scale_jitter_std: float = 0.08
    ret_noise_std: float = 0.01
    ret_drop_prob: float = 0.08

    # vol channel (state) - keep stable
    vol_noise_std: float = 0.001
    vol_drop_prob: float = 0.0

    # clamp for stability
    clamp_ret: float = 0.25
    clamp_vol: float = 1.0


def _time_shift_pad(x: torch.Tensor, shift: int) -> torch.Tensor:
    if shift == 0:
        return x
    C, L = x.shape
    if shift > 0:
        pad = x[:, :1].repeat(1, shift)
        return torch.cat([pad, x[:, :-shift]], dim=1)
    else:
        shift = -shift
        pad = x[:, -1:].repeat(1, shift)
        return torch.cat([x[:, shift:], pad], dim=1)


def augment_view(x: torch.Tensor, cfg: AugmentConfig) -> torch.Tensor:
    """
    x: (C=2, L) where:
      x[0] = log_return
      x[1] = rolling_std(log_return)
    """
    if x.dim() != 2 or x.shape[0] != 2:
        raise ValueError(f"Expected x shape (2, L), got {tuple(x.shape)}")

    out = x.clone()
    _, L = out.shape

    # 1) time shift on BOTH channels
    if cfg.max_time_shift > 0:
        shift = int(torch.randint(-cfg.max_time_shift, cfg.max_time_shift + 1, (1,)).item())
        out = _time_shift_pad(out, shift)

    ret = out[0:1, :]
    vol = out[1:2, :]

    # 2) scale jitter on RETURN only
    if cfg.ret_scale_jitter_std and cfg.ret_scale_jitter_std > 0:
        scale = 1.0 + torch.randn(1).item() * cfg.ret_scale_jitter_std
        ret = ret * scale

    # 3) noise (channel specific)
    if cfg.ret_noise_std and cfg.ret_noise_std > 0:
        ret = ret + torch.randn_like(ret) * cfg.ret_noise_std
    if cfg.vol_noise_std and cfg.vol_noise_std > 0:
        vol = vol + torch.randn_like(vol) * cfg.vol_noise_std

    # 4) dropout/mask (RETURN only by default)
    if cfg.ret_drop_prob and cfg.ret_drop_prob > 0:
        mask = (torch.rand(L) > cfg.ret_drop_prob).float().view(1, L)
        ret = ret * mask

    if cfg.vol_drop_prob and cfg.vol_drop_prob > 0:
        vmask = (torch.rand(L) > cfg.vol_drop_prob).float().view(1, L)
        vol = vol * vmask

    # 5) clamp
    if cfg.clamp_ret and cfg.clamp_ret > 0:
        ret = torch.clamp(ret, -cfg.clamp_ret, cfg.clamp_ret)
    if cfg.clamp_vol and cfg.clamp_vol > 0:
        vol = torch.clamp(vol, 0.0, cfg.clamp_vol)

    return torch.cat([ret, vol], dim=0)


# -----------------------------
# Dataset
# -----------------------------
class TimeSeriesWindowDataset(Dataset):
    def __init__(
        self,
        parquet_dir: str,
        window: int = 50,
        vol_lookback: int = 10,
        min_length: int = 300,
        augment_cfg: Optional[AugmentConfig] = None,
        max_tickers: Optional[int] = None,
        smoothing_window: int = 0,
    ):
        self.window = window
        self.vol_lookback = vol_lookback
        self.augment_cfg = augment_cfg or AugmentConfig()
        self.smoothing_window = smoothing_window

        files = sorted(glob.glob(os.path.join(parquet_dir, "*.parquet")))
        if max_tickers is not None:
            files = files[:max_tickers]

        self.series: List[np.ndarray] = []
        for fp in files:
            df = pd.read_parquet(fp)
            if "Close" not in df.columns:
                continue

            feats = make_features_from_ohlcv(
                df,
                window=window,
                vol_lookback=vol_lookback,
                smoothing_window=self.smoothing_window,
            )
            feats = feats[~np.isnan(feats).any(axis=1)]
            if len(feats) >= max(min_length, window + 1):
                self.series.append(feats.astype(np.float32))

        self.index: List[Tuple[int, int]] = []
        for sid, feats in enumerate(self.series):
            for t in range(0, len(feats) - window):
                self.index.append((sid, t))

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int):
        sid, t = self.index[idx]
        feats = self.series[sid][t:t + self.window]  # (L,2)
        x = torch.from_numpy(feats).transpose(0, 1)  # (2,L)

        v1 = augment_view(x, self.augment_cfg)
        v2 = augment_view(x, self.augment_cfg)
        return v1, v2


# -----------------------------
# Model: small TCN/Conv1D encoder
# -----------------------------
class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int = 5, dilation: int = 1, dropout: float = 0.1):
        super().__init__()
        pad = (k - 1) * dilation // 2
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=k, padding=pad, dilation=dilation)
        self.bn = nn.BatchNorm1d(out_ch)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = F.gelu(x)
        return self.drop(x)


class SmallTCNEncoder(nn.Module):
    def __init__(self, in_ch: int = 2, hidden: int = 64, layers: int = 4, emb_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        blocks = []
        ch = in_ch
        for i in range(layers):
            blocks.append(ConvBlock(ch, hidden, k=5, dilation=2 ** i, dropout=dropout))
            ch = hidden
        self.net = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(hidden, emb_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.net(x)
        h = self.pool(h).squeeze(-1)
        z = self.fc(h)
        z = F.normalize(z, dim=-1)
        return z


class MLP(nn.Module):
    def __init__(self, dim: int, hidden: int = 256, out: Optional[int] = None):
        super().__init__()
        out = out or dim
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.GELU(),
            nn.Linear(hidden, out),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# -----------------------------
# BYOL wrapper
# -----------------------------
@torch.no_grad()
def ema_update(target: nn.Module, online: nn.Module, m: float) -> None:
    for tp, op in zip(target.parameters(), online.parameters()):
        tp.data.mul_(m).add_(op.data, alpha=1 - m)


class BYOL(nn.Module):
    def __init__(self, encoder: nn.Module, emb_dim: int = 64, proj_dim: int = 128, m: float = 0.99):
        super().__init__()
        self.m = m

        self.online_encoder = encoder
        self.online_proj = MLP(emb_dim, hidden=256, out=proj_dim)
        self.online_pred = MLP(proj_dim, hidden=256, out=proj_dim)

        import copy
        self.target_encoder = copy.deepcopy(self.online_encoder)
        self.target_proj = copy.deepcopy(self.online_proj)

        for p in self.target_encoder.parameters():
            p.requires_grad = False
        for p in self.target_proj.parameters():
            p.requires_grad = False

    def forward(self, v1: torch.Tensor, v2: torch.Tensor) -> torch.Tensor:
        z1 = self.online_encoder(v1)
        z2 = self.online_encoder(v2)
        p1 = self.online_pred(self.online_proj(z1))
        p2 = self.online_pred(self.online_proj(z2))
        p1 = F.normalize(p1, dim=-1)
        p2 = F.normalize(p2, dim=-1)

        with torch.no_grad():
            t1 = self.target_proj(self.target_encoder(v1))
            t2 = self.target_proj(self.target_encoder(v2))
            t1 = F.normalize(t1, dim=-1)
            t2 = F.normalize(t2, dim=-1)

        loss = (2 - 2 * (p1 * t2).sum(dim=-1)) + (2 - 2 * (p2 * t1).sum(dim=-1))
        return loss.mean()

    @torch.no_grad()
    def update_target(self) -> None:
        ema_update(self.target_encoder, self.online_encoder, self.m)
        ema_update(self.target_proj, self.online_proj, self.m)

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.online_encoder(x)


# -----------------------------
# Train loop
# -----------------------------
@dataclass
class TrainConfig:
    parquet_dir: str = "data/yahoo_parquet"
    window: int = 50
    vol_lookback: int = 10
    smoothing_window: int = 0

    emb_dim: int = 64
    batch_size: int = 256
    lr: float = 2e-4
    epochs: int = 5
    num_workers: int = 4

    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    save_path: str = "byol_tcn_encoder_shape_first.pt"

    max_tickers: Optional[int] = 800

    seed: int = 42
    deterministic: bool = True

    # ✅ FIX: use default_factory for mutable default
    aug: AugmentConfig = field(default_factory=AugmentConfig)

    hidden: int = 64
    layers: int = 4
    dropout: float = 0.1

    proj_dim: int = 128
    ema_m: float = 0.99


def train_byol(cfg: TrainConfig) -> None:
    seed_everything(cfg.seed, deterministic=cfg.deterministic)

    ds = TimeSeriesWindowDataset(
        parquet_dir=cfg.parquet_dir,
        window=cfg.window,
        vol_lookback=cfg.vol_lookback,
        smoothing_window=cfg.smoothing_window,
        augment_cfg=cfg.aug,
        max_tickers=cfg.max_tickers,
    )

    g = torch.Generator()
    g.manual_seed(cfg.seed)

    dl = DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        drop_last=True,
        worker_init_fn=seed_worker,
        generator=g,
        pin_memory=(cfg.device.startswith("cuda")),
        persistent_workers=(cfg.num_workers > 0),
    )

    encoder = SmallTCNEncoder(
        in_ch=2,
        hidden=cfg.hidden,
        layers=cfg.layers,
        emb_dim=cfg.emb_dim,
        dropout=cfg.dropout,
    )
    model = BYOL(
        encoder=encoder,
        emb_dim=cfg.emb_dim,
        proj_dim=cfg.proj_dim,
        m=cfg.ema_m,
    ).to(cfg.device)

    opt = torch.optim.AdamW(
        list(model.online_encoder.parameters()) +
        list(model.online_proj.parameters()) +
        list(model.online_pred.parameters()),
        lr=cfg.lr
    )

    model.train()
    for epoch in range(cfg.epochs):
        pbar = tqdm(dl, desc=f"epoch {epoch+1}/{cfg.epochs}")
        running = 0.0
        for i, (v1, v2) in enumerate(pbar):
            v1 = v1.to(cfg.device, non_blocking=True)
            v2 = v2.to(cfg.device, non_blocking=True)

            loss = model(v1, v2)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            model.update_target()

            running = 0.98 * running + 0.02 * float(loss.item()) if i > 0 else float(loss.item())
            pbar.set_postfix(loss=running)

    encoder_cpu = model.online_encoder.to("cpu").eval()
    torch.save(
        {
            "state_dict": encoder_cpu.state_dict(),
            "emb_dim": cfg.emb_dim,
            "window": cfg.window,
            "vol_lookback": cfg.vol_lookback,
            "smoothing_window": cfg.smoothing_window,
            "seed": cfg.seed,
            "deterministic": cfg.deterministic,
            "augment": cfg.aug.__dict__,
            "model": {"hidden": cfg.hidden, "layers": cfg.layers, "dropout": cfg.dropout},
        },
        cfg.save_path
    )
    print(f"Saved encoder to {cfg.save_path}")


if __name__ == "__main__":
    cfg = TrainConfig(
        window=50,
        vol_lookback=10,
        smoothing_window=5,
        emb_dim=64,
        batch_size=512, # 256
        epochs=10, # 5  15
        max_tickers=800,     # 50 MVP quick test
        seed=42,
        deterministic=True,
        save_path="byol_tcn_encoder_shape_first_v03.pt",
    )
    # 기존
    # cfg.aug = AugmentConfig(
    #     max_time_shift=5,
    #     ret_scale_jitter_std=0.10,
    #     ret_noise_std=0.012,
    #     ret_drop_prob=0.10,
    #     vol_noise_std=0.001,
    #     vol_drop_prob=0.0,
    #     clamp_ret=0.25,
    #     clamp_vol=1.0,
    # )
    
    # v02
    # cfg.aug = AugmentConfig(
    #     max_time_shift=5,
    #     ret_scale_jitter_std=0.05, 
    #     ret_noise_std=0.006, 
    #     ret_drop_prob=0.05, 
    #     vol_noise_std=0.005,
    #     vol_drop_prob=0.0,
    #     clamp_ret=0.25,
    #     clamp_vol=1.0,)

    # v03: Symmetric augmentation (양 채널 동등 처리)
    cfg.aug = AugmentConfig(
        max_time_shift=4,
        ret_scale_jitter_std=0.04,   # 약간 낮춤
        ret_noise_std=0.005,
        ret_drop_prob=0.04,
        vol_noise_std=0.005,         # 수익률과 동일
        vol_drop_prob=0.04,          # 수익률과 동일 (핵심 변경)
        clamp_ret=0.25,
        clamp_vol=1.0,
    )

    train_byol(cfg)
