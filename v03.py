# optuna_byol_tuning.py
"""
BYOL Shape-First Representation Learning - Optuna Hyperparameter Optimization
Based on v03 configuration as baseline
"""

import os
import glob
import random
import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, List, Tuple
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

import optuna
from optuna.trial import Trial
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

# =============================================================================
# 재현성 설정
# =============================================================================
def seed_everything(seed: int = 42, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# =============================================================================
# Feature 생성
# =============================================================================
def make_features_from_ohlcv(
    df: pd.DataFrame,
    window: int = 50,
    vol_lookback: int = 10,
    smoothing_window: int = 0,
) -> np.ndarray:
    close = df["Close"].astype(float)
    if smoothing_window and smoothing_window > 1:
        close = close.rolling(window=smoothing_window, min_periods=1).mean()
    close = close.values
    lr = np.diff(np.log(close), prepend=np.nan)
    vol = pd.Series(lr).rolling(vol_lookback).std().values
    feats = np.stack([lr, vol], axis=1)
    return feats


# =============================================================================
# 증강 설정
# =============================================================================
@dataclass
class AugmentConfig:
    max_time_shift: int = 4
    ret_scale_jitter_std: float = 0.05
    ret_noise_std: float = 0.006
    ret_drop_prob: float = 0.05
    vol_noise_std: float = 0.005
    vol_drop_prob: float = 0.0
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
    if x.dim() != 2 or x.shape[0] != 2:
        raise ValueError(f"Expected x shape (2, L), got {tuple(x.shape)}")

    out = x.clone()
    _, L = out.shape

    if cfg.max_time_shift > 0:
        shift = int(torch.randint(-cfg.max_time_shift, cfg.max_time_shift + 1, (1,)).item())
        out = _time_shift_pad(out, shift)

    ret = out[0:1, :]
    vol = out[1:2, :]

    if cfg.ret_scale_jitter_std and cfg.ret_scale_jitter_std > 0:
        scale = 1.0 + torch.randn(1).item() * cfg.ret_scale_jitter_std
        ret = ret * scale

    if cfg.ret_noise_std and cfg.ret_noise_std > 0:
        ret = ret + torch.randn_like(ret) * cfg.ret_noise_std
    if cfg.vol_noise_std and cfg.vol_noise_std > 0:
        vol = vol + torch.randn_like(vol) * cfg.vol_noise_std

    if cfg.ret_drop_prob and cfg.ret_drop_prob > 0:
        mask = (torch.rand(L) > cfg.ret_drop_prob).float().view(1, L)
        ret = ret * mask
    if cfg.vol_drop_prob and cfg.vol_drop_prob > 0:
        vmask = (torch.rand(L) > cfg.vol_drop_prob).float().view(1, L)
        vol = vol * vmask

    if cfg.clamp_ret and cfg.clamp_ret > 0:
        ret = torch.clamp(ret, -cfg.clamp_ret, cfg.clamp_ret)
    if cfg.clamp_vol and cfg.clamp_vol > 0:
        vol = torch.clamp(vol, 0.0, cfg.clamp_vol)

    return torch.cat([ret, vol], dim=0)


# =============================================================================
# Dataset
# =============================================================================
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
                df, window=window, vol_lookback=vol_lookback,
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
        feats = self.series[sid][t:t + self.window]
        x = torch.from_numpy(feats).transpose(0, 1)
        v1 = augment_view(x, self.augment_cfg)
        v2 = augment_view(x, self.augment_cfg)
        return v1, v2


# =============================================================================
# 모델 정의
# =============================================================================
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


# =============================================================================
# 평가 지표
# =============================================================================
def compute_participation_ratio(embeddings: np.ndarray) -> float:
    pca = PCA().fit(embeddings)
    eig = pca.explained_variance_
    return float((eig.sum() ** 2) / (eig ** 2).sum())


def compute_neighbor_overlap(
    embeddings: np.ndarray,
    reference: np.ndarray,
    k: int = 10,
    emb_metric: str = 'cosine',
    ref_metric: str = 'euclidean'
) -> float:
    n = len(embeddings)
    k = min(k, n - 1)
    
    nn_emb = NearestNeighbors(n_neighbors=k, metric=emb_metric).fit(embeddings)
    nn_ref = NearestNeighbors(n_neighbors=k, metric=ref_metric).fit(reference)
    
    _, idx_emb = nn_emb.kneighbors(embeddings)
    _, idx_ref = nn_ref.kneighbors(reference)
    
    overlaps = [len(set(idx_emb[i]) & set(idx_ref[i])) / k for i in range(n)]
    return float(np.mean(overlaps))


def compute_cosine_stats(embeddings: np.ndarray, n_samples: int = 3000) -> Dict[str, float]:
    from sklearn.metrics.pairwise import cosine_distances
    n = len(embeddings)
    if n > 500:
        idx = np.random.choice(n, min(500, n), replace=False)
        embeddings = embeddings[idx]
    
    cos_dist = cosine_distances(embeddings)
    upper_tri = cos_dist[np.triu_indices_from(cos_dist, k=1)]
    
    return {
        'mean': float(np.mean(upper_tri)),
        'std': float(np.std(upper_tri)),
    }


# =============================================================================
# 학습 함수
# =============================================================================
def train_and_evaluate(
    parquet_dir: str,
    aug_config: AugmentConfig,
    window: int = 50,
    vol_lookback: int = 10,
    smoothing_window: int = 5,
    emb_dim: int = 64,
    hidden: int = 64,
    layers: int = 4,
    dropout: float = 0.1,
    proj_dim: int = 128,
    ema_m: float = 0.993,          # ✅ 0.996 -> 0.993 (압축 완화 쪽)
    batch_size: int = 256,         # ✅ 256 유지 (BN/안정성/속도 균형)
    lr: float = 1.5e-4,
    epochs: int = 10,
    max_tickers_train: int = 400,
    max_tickers_eval: int = 800,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    seed: int = 42,
    verbose: bool = False,
    trial: Optional[Trial] = None,
) -> Dict[str, float]:
    """모델 학습 및 평가, 지표 반환"""
    
    seed_everything(seed)
    

    t0 = time.time()
    print("[1] building dataset...")
    # Dataset
    ds = TimeSeriesWindowDataset(
        parquet_dir=parquet_dir,
        window=window,
        vol_lookback=vol_lookback,
        min_length=300,
        augment_cfg=aug_config,
        max_tickers=max_tickers_train,
        smoothing_window=smoothing_window,
    )
    print(f"[1] dataset done: len(ds)={len(ds)} series={len(ds.series)} sec={time.time()-t0:.1f}")

    if len(ds) == 0:
        return {'error': True}
    
    g = torch.Generator()
    g.manual_seed(seed)
    
    print("[2] building dataloader...")
    dl = DataLoader(
        ds, batch_size=batch_size, shuffle=True,
        num_workers=4, drop_last=True,
        worker_init_fn=seed_worker, generator=g,
        pin_memory=(device == 'cuda'),
        persistent_workers=True,
    )
    print("[2] dataloader done")

    # Model
    print("[3] building model/optim...")
    encoder = SmallTCNEncoder(in_ch=2, hidden=hidden, layers=layers, emb_dim=emb_dim, dropout=dropout)
    model = BYOL(encoder=encoder, emb_dim=emb_dim, proj_dim=proj_dim, m=ema_m).to(device)
    print("[3] model done")

    opt = torch.optim.AdamW(
        list(model.online_encoder.parameters()) +
        list(model.online_proj.parameters()) +
        list(model.online_pred.parameters()),
        lr=lr
    )
    
    # Training
    print("[4] start training loop (first batch probe)...")
    it = iter(dl)
    v1, v2 = next(it) 
    print("[4] got first batch", v1.shape, v2.shape)

    model.train()
    for epoch in range(epochs):
        pbar = tqdm(dl, desc=f"Epoch {epoch+1}/{epochs}", disable=not verbose)
        running_loss = 0.0
        
        for i, (v1, v2) in enumerate(pbar):
            v1 = v1.to(device, non_blocking=True)
            v2 = v2.to(device, non_blocking=True)
            
            loss = model(v1, v2)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            model.update_target()
            
            running_loss = 0.95 * running_loss + 0.05 * loss.item() if i > 0 else loss.item()
            if verbose:
                pbar.set_postfix(loss=f"{running_loss:.4f}")
        
        # 중간 평가 및 pruning (epoch 5 이후)
        if trial is not None and epoch >= 4:
            encoder_eval = model.online_encoder.cpu().eval()
            quick_metrics = quick_evaluate(encoder_eval, parquet_dir, window, vol_lookback, smoothing_window, max_tickers=200)
            model.to(device)
            model.train()
            
            trial.report(quick_metrics.get('neighbor_overlap_returns', 0), epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()
    
    # Final evaluation
    encoder_final = model.online_encoder.cpu().eval()
    metrics = full_evaluate(encoder_final, parquet_dir, window, vol_lookback, smoothing_window, max_tickers_eval)
    
    return metrics


def quick_evaluate(
    encoder: nn.Module,
    parquet_dir: str,
    window: int,
    vol_lookback: int,
    smoothing_window: int,
    max_tickers: int = 200
) -> Dict[str, float]:
    """빠른 평가 (pruning용)"""
    embeddings, returns_mat, vol_mat, _ = extract_embeddings(
        encoder, parquet_dir, window, vol_lookback, smoothing_window, max_tickers
    )
    
    if len(embeddings) < 30:
        return {'neighbor_overlap_returns': 0.0}
    
    overlap_ret = compute_neighbor_overlap(embeddings, returns_mat, k=10)
    pr = compute_participation_ratio(embeddings)
    
    return {
        'neighbor_overlap_returns': overlap_ret,
        'participation_ratio': pr,
    }


def full_evaluate(
    encoder: nn.Module,
    parquet_dir: str,
    window: int,
    vol_lookback: int,
    smoothing_window: int,
    max_tickers: int = 800
) -> Dict[str, float]:
    """전체 평가"""
    embeddings, returns_mat, vol_mat, _ = extract_embeddings(
        encoder, parquet_dir, window, vol_lookback, smoothing_window, max_tickers
    )
    
    if len(embeddings) < 50:
        return {'error': True}
    
    pr = compute_participation_ratio(embeddings)
    overlap_ret = compute_neighbor_overlap(embeddings, returns_mat, k=10)
    overlap_vol = compute_neighbor_overlap(embeddings, vol_mat, k=10)
    cos_stats = compute_cosine_stats(embeddings)
    
    return {
        'participation_ratio': pr,
        'neighbor_overlap_returns': overlap_ret,
        'neighbor_overlap_volatility': overlap_vol,
        'cosine_mean': cos_stats['mean'],
        'cosine_std': cos_stats['std'],
    }


def extract_embeddings(
    encoder: nn.Module,
    parquet_dir: str,
    window: int,
    vol_lookback: int,
    smoothing_window: int,
    max_tickers: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """임베딩 및 특성 추출"""
    encoder.eval()
    files = sorted(glob.glob(os.path.join(parquet_dir, "*.parquet")))[:max_tickers]
    
    embeddings, returns_list, vol_list, tickers = [], [], [], []
    
    for fp in files:
        try:
            df = pd.read_parquet(fp)
            if "Close" not in df.columns:
                continue
            
            feats = make_features_from_ohlcv(df, window, vol_lookback, smoothing_window)
            feats = feats[~np.isnan(feats).any(axis=1)]
            
            if len(feats) < window:
                continue
            
            last_window = feats[-window:]
            x = torch.from_numpy(last_window.astype(np.float32)).T.unsqueeze(0)
            
            with torch.no_grad():
                z = encoder(x).squeeze().numpy()
            
            embeddings.append(z)
            returns_list.append(last_window[:, 0])
            vol_list.append(last_window[:, 1])
            tickers.append(os.path.basename(fp).replace('.parquet', ''))
        except:
            continue
    
    return np.array(embeddings), np.array(returns_list), np.array(vol_list), tickers


# =============================================================================
# Optuna Objective
# =============================================================================
def create_objective(
    parquet_dir: str,
    window: int = 50,
    vol_lookback: int = 10,
    smoothing_window: int = 5,
    epochs: int = 10,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
):
    """Optuna objective 함수 생성"""
    
    def objective(trial: Trial) -> float:
        print(f"[trial {trial.number}] start")
        # =====================
        # 하이퍼파라미터 샘플링
        # =====================
        
        # 증강 파라미터 (핵심 탐색 대상)
        # 증강 파라미터 (완화된 탐색 범위)
        aug_config = AugmentConfig(
            max_time_shift=trial.suggest_int('max_time_shift', 1, 4),                 # ✅ 3~6 -> 1~4
            ret_scale_jitter_std=trial.suggest_float('ret_scale_jitter_std', 0.01, 0.06),  # ✅ 0.02~0.10 -> 0.01~0.06
            ret_noise_std=trial.suggest_float('ret_noise_std', 0.0015, 0.008),        # ✅ 0.003~0.015 -> 0.0015~0.008
            ret_drop_prob=trial.suggest_float('ret_drop_prob', 0.00, 0.06),           # ✅ 0.02~0.12 -> 0~0.06
            vol_noise_std=trial.suggest_float('vol_noise_std', 0.001, 0.008),         # ✅ 0.002~0.012 -> 0.001~0.008
            vol_drop_prob=trial.suggest_float('vol_drop_prob', 0.0, 0.05),            # ✅ 0.0~0.08 -> 0.0~0.05
            clamp_ret=0.25,
            clamp_vol=1.0,
        )

        # 학습 파라미터
        lr = trial.suggest_float('lr', 5e-5, 3e-4, log=True)                          # ✅ 상한 살짝 낮춤(안정화)
        ema_m = trial.suggest_float('ema_m', 0.990, 0.995)                             # ✅ 0.990~0.999 -> 0.990~0.995
        batch_size = trial.suggest_categorical('batch_size', [128, 256])               # ✅ 256,512 -> 128,256

        
        # =====================
        # 학습 및 평가
        # =====================
        try:
            metrics = train_and_evaluate(
                parquet_dir=parquet_dir,
                aug_config=aug_config,
                window=window,
                vol_lookback=vol_lookback,
                smoothing_window=smoothing_window,
                emb_dim=64,  # 고정 (v02 기준)
                hidden=64,
                layers=4,
                dropout=0.1,
                proj_dim=128,
                ema_m=ema_m,
                batch_size=batch_size,
                lr=lr,
                epochs=epochs,
                max_tickers_train=150,
                max_tickers_eval=300,
                device=device,
                seed=42,
                verbose=True,
                trial=trial,
            )
        except optuna.TrialPruned:
            raise
        except Exception as e:
            print(f"Trial {trial.number} failed: {e}")
            return 0.0
        
        if metrics.get('error'):
            return 0.0
        
        # =====================
        # 지표 저장
        # =====================
        for key, value in metrics.items():
            trial.set_user_attr(key, value)
        
        # =====================
        # 복합 점수 계산
        # =====================
        overlap_ret = metrics['neighbor_overlap_returns']
        overlap_vol = metrics['neighbor_overlap_volatility']
        pr = metrics['participation_ratio']
        cos_mean = metrics['cosine_mean']
        
        # PR 페널티: PR < 2 (붕괴) 또는 PR > 10 (압축 부족)
        pr_penalty = 0.0
        if pr < 2.0:
            pr_penalty = (2.0 - pr) * 0.1
        elif pr > 10.0:
            pr_penalty = (pr - 10.0) * 0.02
        
        # 코사인 페널티: 너무 집중되면 (mean < 0.15) 페널티
        cos_penalty = 0.0
        if cos_mean < 0.15:
            cos_penalty = (0.15 - cos_mean) * 0.3
        
        # 복합 점수
        # PR 보너스: 3~6 구간에 보너스, 2 미만/10 초과 페널티는 유지
        pr_bonus = 0.0
        if 3.0 <= pr <= 6.0:
            pr_bonus = 0.05
        elif 2.0 <= pr < 3.0:
            pr_bonus = 0.02

        score = (
            overlap_ret * 0.6 +
            overlap_vol * 0.3 +
            pr_bonus -
            pr_penalty -
            cos_penalty
        )

        
        print(f"Trial {trial.number}: score={score:.4f}, "
              f"overlap_ret={overlap_ret:.3f}, overlap_vol={overlap_vol:.3f}, "
              f"PR={pr:.2f}, cos_mean={cos_mean:.3f}")
        
        return score
    
    return objective


# =============================================================================
# 메인 실행
# =============================================================================
def run_optimization(
    parquet_dir: str = "data/yahoo_parquet",
    study_name: str = "byol_v03_optimization",
    n_trials: int = 50,
    timeout_hours: float = 4.0,
    epochs: int = 10,
):
    """Optuna 최적화 실행"""
    
    storage = f"sqlite:///{study_name}.db"
    
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction='maximize',
        load_if_exists=True,
        sampler=TPESampler(seed=42),
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=5),
    )
    
    objective = create_objective(
        parquet_dir=parquet_dir,
        epochs=epochs,
    )
    
    print(f"\n{'='*70}")
    print(f"Starting Optuna optimization")
    print(f"Study: {study_name}")
    print(f"Trials: {n_trials}, Timeout: {timeout_hours}h")
    print(f"{'='*70}\n")
    
    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=int(timeout_hours * 3600),
        show_progress_bar=False,
    )
    
    return study


def print_results(study: optuna.Study):
    """결과 출력"""
    print(f"\n{'='*70}")
    print("OPTIMIZATION RESULTS")
    print(f"{'='*70}")
    
    print(f"\nCompleted trials: {len(study.trials)}")
    print(f"Best trial: #{study.best_trial.number}")
    print(f"Best score: {study.best_trial.value:.4f}")
    
    print("\n--- Best Hyperparameters ---")
    for key, value in study.best_trial.params.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.6f}")
        else:
            print(f"  {key}: {value}")
    
    print("\n--- Best Metrics ---")
    for key, value in study.best_trial.user_attrs.items():
        print(f"  {key}: {value:.4f}")
    
    # Top 5
    print("\n--- Top 5 Trials ---")
    trials = sorted(study.trials, key=lambda t: t.value if t.value else 0, reverse=True)[:5]
    for t in trials:
        if t.value:
            overlap = t.user_attrs.get('neighbor_overlap_returns', 0)
            pr = t.user_attrs.get('participation_ratio', 0)
            print(f"  #{t.number}: score={t.value:.4f}, overlap={overlap:.3f}, PR={pr:.2f}")


def save_best_config(study: optuna.Study, output_path: str = "best_config.json"):
    """최적 설정 저장"""
    config = {
        'best_score': study.best_trial.value,
        'params': study.best_trial.params,
        'metrics': study.best_trial.user_attrs,
        'timestamp': datetime.now().isoformat(),
    }
    
    with open(output_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\nBest config saved to: {output_path}")


def train_final_model(
    study: optuna.Study,
    parquet_dir: str = "data/yahoo_parquet",
    epochs: int = 20,
    save_path: Optional[str] = None,  # None이면 자동 생성
):
    """최적 파라미터로 최종 모델 학습 및 저장"""
    if save_path is None:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        save_path = f"byol_v03_optuna_{timestamp}.pt"

    params = study.best_trial.params
    
    aug_config = AugmentConfig(
        max_time_shift=params['max_time_shift'],
        ret_scale_jitter_std=params['ret_scale_jitter_std'],
        ret_noise_std=params['ret_noise_std'],
        ret_drop_prob=params['ret_drop_prob'],
        vol_noise_std=params['vol_noise_std'],
        vol_drop_prob=params['vol_drop_prob'],
    )
    
    print(f"\n{'='*70}")
    print("TRAINING FINAL MODEL")
    print(f"{'='*70}\n")
    
    # 직접 학습 (train_and_evaluate 대신)
    seed_everything(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    window = 50
    vol_lookback = 10
    smoothing_window = 5
    emb_dim = 64
    hidden = 64
    layers = 4
    dropout = 0.1
    proj_dim = 128
    
    # Dataset
    ds = TimeSeriesWindowDataset(
        parquet_dir=parquet_dir,
        window=window,
        vol_lookback=vol_lookback,
        min_length=300,
        augment_cfg=aug_config,
        max_tickers=800,
        smoothing_window=smoothing_window,
    )
    
    g = torch.Generator()
    g.manual_seed(42)
    
    dl = DataLoader(
        ds, 
        batch_size=params['batch_size'], 
        shuffle=True,
        num_workers=0,              # <= 변경
        drop_last=True,
        pin_memory=False,           # <= 일단 끄기
        persistent_workers=False,   # <= 일단 끄기
        )

    
    # Model
    encoder = SmallTCNEncoder(in_ch=2, hidden=hidden, layers=layers, emb_dim=emb_dim, dropout=dropout)
    model = BYOL(encoder=encoder, emb_dim=emb_dim, proj_dim=proj_dim, m=params['ema_m']).to(device)
    
    opt = torch.optim.AdamW(
        list(model.online_encoder.parameters()) +
        list(model.online_proj.parameters()) +
        list(model.online_pred.parameters()),
        lr=params['lr']
    )
    
    # Training
    model.train()
    for epoch in range(epochs):
        pbar = tqdm(dl, desc=f"Epoch {epoch+1}/{epochs}")
        running_loss = 0.0
        
        for i, (v1, v2) in enumerate(pbar):
            v1 = v1.to(device, non_blocking=True)
            v2 = v2.to(device, non_blocking=True)
            
            loss = model(v1, v2)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            model.update_target()
            
            running_loss = 0.95 * running_loss + 0.05 * loss.item() if i > 0 else loss.item()
            pbar.set_postfix(loss=f"{running_loss:.4f}")
    
    # Evaluation
    encoder_final = model.online_encoder.cpu().eval()
    metrics = full_evaluate(encoder_final, parquet_dir, window, vol_lookback, smoothing_window, 800)
    
    print("\n--- Final Model Metrics ---")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")
    
    # 모델 저장
    torch.save({
        'state_dict': encoder_final.state_dict(),
        'emb_dim': emb_dim,
        'hidden': hidden,
        'layers': layers,
        'dropout': dropout,
        'window': window,
        'vol_lookback': vol_lookback,
        'smoothing_window': smoothing_window,
        'aug_config': aug_config.__dict__,
        'params': params,
        'metrics': metrics,
    }, save_path)
    
    print(f"\n✅ Model saved to: {save_path}")
    
    return encoder_final, metrics


# =============================================================================
# CLI
# =============================================================================
if __name__ == "__main__":

    import multiprocessing as mp
    mp.freeze_support()
    # mp.set_start_method("spawn", force=True)

    import argparse
    
    parser = argparse.ArgumentParser(description="BYOL Optuna Optimization")
    parser.add_argument('--parquet_dir', type=str, default='data/yahoo_parquet')
    parser.add_argument('--study_name', type=str, default='byol_v32_tuning')
    parser.add_argument('--n_trials', type=int, default=50)
    parser.add_argument('--timeout_hours', type=float, default=4.0)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--train_final', action='store_true')
    parser.add_argument('--final_epochs', type=int, default=20)
    
    args = parser.parse_args()
    
    # 최적화 실행
    study = run_optimization(
        parquet_dir=args.parquet_dir,
        study_name=args.study_name,
        n_trials=args.n_trials,
        timeout_hours=args.timeout_hours,
        epochs=args.epochs,
    )
    
    # 결과 출력 및 저장
    print_results(study)
    save_best_config(study, f"{args.study_name}_best.json")
    
    # 최종 모델 학습 (선택)
    if args.train_final:
        train_final_model(
            study=study,
            parquet_dir=args.parquet_dir,
            epochs=args.final_epochs,
        )