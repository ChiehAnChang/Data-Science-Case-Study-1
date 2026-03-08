#!/usr/bin/env python3
"""
Dataset/generate_synthetic.py

Produces three augmented training datasets for the wildfire PM2.5 transformer:

  1. PCHIP 30-min temporal upsampling
       Output: Dataset/Outputs/CS2_pchip30min.csv
       Drop-in replacement for CS2_model_input.csv — use SEQ_LEN=144 in notebook.

  2. Conditional VAE (CVAE)
       Output: Dataset/Outputs/CS2_cvae_aug.npz
       Per-zone arrays of real + synthetic sequences, ready to concatenate.

  3. TimeGAN  (Yoon et al. 2019)
       Output: Dataset/Outputs/CS2_timegan_aug.npz
       Same NPZ format as CVAE.

Usage (run from repo root):
  python Dataset/generate_synthetic.py --all
  python Dataset/generate_synthetic.py --pchip
  python Dataset/generate_synthetic.py --cvae --timegan
  python Dataset/generate_synthetic.py --cvae --syn-multiplier 2.0

Loading NPZ results in CS2 Modeling.ipynb:
  data     = np.load("../Dataset/Outputs/CS2_cvae_aug.npz")
  zone_key = "Lower_Fraser_Valley"   # or "Southern_Interior"
  X_real   = data[f"{zone_key}_X_real"]   # (N, 72, 11)  original scale
  y_real   = data[f"{zone_key}_y_real"]   # (N,)
  X_syn    = data[f"{zone_key}_X_syn"]    # (N_syn, 72, 11) original scale
  y_syn    = data[f"{zone_key}_y_syn"]    # (N_syn,)
  X_aug    = np.concatenate([X_real, X_syn])
  y_aug    = np.concatenate([y_real, y_syn])
  # Then apply StandardScaler to X_aug before building the DataLoader.
"""

import argparse
import time
import warnings
from itertools import cycle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator
from sklearn.preprocessing import MinMaxScaler, StandardScaler

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).resolve().parent.parent
INPUT_CSV  = ROOT / "Dataset" / "Outputs" / "CS2_model_input.csv"
OUTPUT_DIR = ROOT / "Dataset" / "Outputs"
MODEL_DIR  = OUTPUT_DIR / "syn_models"

# ── Feature configuration (matches RAW_FEATURES in CS2 Modeling.ipynb) ────────
RAW_FEATURES = [
    "PM25",
    "fire_count_regional", "frp_regional_sum", "hfi_weighted", "fwi_mean",
    "fire_count_local",    "frp_local_sum",
    "hour", "day_of_week", "month", "is_wildfire_season",
]
PM25_IDX    = 0                   # index of PM25 within RAW_FEATURES
N_FEAT      = len(RAW_FEATURES)   # 11
TRAIN_YEARS = [2022, 2023, 2024]

# Sequence config — keep in sync with notebook
SEQ_LEN  = 72   # 72 h of history (hourly data)
HORIZON  = 1    # predict 1 h ahead

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ═══════════════════════════════════════════════════════════════════════════════
# Shared utilities
# ═══════════════════════════════════════════════════════════════════════════════

def _zone_key(zone: str) -> str:
    """Zone name → valid numpy archive key (spaces → underscores)."""
    return zone.replace(" ", "_")


def _season(month_s: pd.Series) -> pd.Series:
    mapping = {12: 1, 1: 1, 2: 1, 3: 2, 4: 2, 5: 2,
               6: 3, 7: 3, 8: 3, 9: 4, 10: 4, 11: 4}
    return month_s.map(mapping)


def _recompute_lag_features(d: pd.DataFrame, freq: str) -> pd.DataFrame:
    """Recompute all PM2.5 lag / rolling features after temporal resampling.

    pph = periods per hour (e.g. 2 for 30-min data, 4 for 15-min data).
    All window sizes are expressed in periods so they represent the same
    physical duration regardless of resolution.
    """
    pph = int(round(pd.Timedelta("1h") / pd.Timedelta(freq)))
    d   = d.sort_values("Datetime_UTC").copy()

    for n_h, col in [
        (1,  "PM25_lag_1h"),
        (6,  "PM25_lag_6h"),
        (24, "PM25_lag_24h"),
        (48, "PM25_lag_48h"),
    ]:
        d[col] = d["PM25"].shift(n_h * pph)

    w24 = 24 * pph
    w7d = 24 * 7 * pph
    d["PM25_roll_24h_mean"] = d["PM25"].rolling(w24, min_periods=1).mean()
    d["PM25_roll_24h_max"]  = d["PM25"].rolling(w24, min_periods=1).max()
    d["PM25_roll_7d_mean"]  = d["PM25"].rolling(w7d, min_periods=1).mean()
    d["frp_roll_24h_sum"]   = d["frp_local_sum"].rolling(24 * pph, min_periods=1).sum()
    d["frp_roll_72h_sum"]   = d["frp_local_sum"].rolling(72 * pph, min_periods=1).sum()
    return d


def _build_sequences(
    d: pd.DataFrame,
    features: list,
    seq_len: int = SEQ_LEN,
    horizon: int = HORIZON,
) -> tuple:
    """Build overlapping (X_seq, y) pairs from a zone DataFrame.

    Returns:
        X : np.ndarray  (N, seq_len, n_features)   — original scale
        y : np.ndarray  (N,)                        — PM25 at t+horizon
    """
    d = d.copy().sort_values("Datetime_UTC")
    for c in features + ["PM25"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna(subset=features + ["PM25"])

    X_arr = d[features].values.astype(np.float32)
    y_arr = d["PM25"].values.astype(np.float32)

    Xs, ys = [], []
    for i in range(seq_len - 1, len(d) - horizon):
        Xs.append(X_arr[i - seq_len + 1 : i + 1])
        ys.append(y_arr[i + horizon])
    return np.stack(Xs), np.array(ys, dtype=np.float32)


def _load_and_prep(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["Datetime_UTC"] = pd.to_datetime(df["Datetime_UTC"], utc=True, errors="coerce")
    df = df.dropna(subset=["Datetime_UTC"])
    for c in RAW_FEATURES:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.sort_values(["Zone", "Datetime_UTC"]).reset_index(drop=True)


def _header(title: str) -> None:
    print(f"\n{'='*62}")
    print(f"  {title}")
    print(f"{'='*62}")


def _device_info() -> None:
    """Print a full device report (GPU name, VRAM, CUDA version)."""
    print(f"\n  Device : {DEVICE.type.upper()}", end="")
    if DEVICE.type == "cuda":
        props    = torch.cuda.get_device_properties(0)
        vram_gb  = props.total_memory / 1024 ** 3
        print(f"  ({props.name}, {vram_gb:.1f} GB VRAM, "
              f"CUDA {torch.version.cuda}, "
              f"PyTorch {torch.__version__})")
    else:
        print(f"  (no CUDA detected — training will be significantly slower)")
        print(f"  PyTorch {torch.__version__}")


def _gpu_mem() -> str:
    """Return a short GPU memory string, or empty string on CPU."""
    if DEVICE.type == "cuda":
        used = torch.cuda.memory_allocated() / 1024 ** 2
        peak = torch.cuda.max_memory_allocated() / 1024 ** 2
        return f"  mem {used:.0f}/{peak:.0f} MB"
    return ""


def _pbar(step: int, total: int, prefix: str = "", width: int = 28) -> None:
    """Print an in-place ASCII progress bar.  Call with step==total for newline."""
    pct    = step / max(total, 1)
    filled = int(width * pct)
    bar    = "#" * filled + "-" * (width - filled)
    end    = "\n" if step >= total else ""
    print(f"\r    {prefix} [{bar}] {step:>{len(str(total))}}/{total} "
          f"({100*pct:5.1f}%)", end=end, flush=True)


def _eta(elapsed: float, step: int, total: int) -> str:
    """Return a human-readable ETA string."""
    if step == 0:
        return "ETA --:--"
    remaining = elapsed / step * (total - step)
    m, s = divmod(int(remaining), 60)
    return f"ETA {m:02d}:{s:02d}"


# ═══════════════════════════════════════════════════════════════════════════════
# 1. PCHIP 30-min upsampling
# ═══════════════════════════════════════════════════════════════════════════════

def generate_pchip(df: pd.DataFrame, freq: str = "30min") -> pd.DataFrame:
    """Upsample all zones from hourly to `freq`.

    Strategy per feature group:
      - PM2.5          : PCHIP  (smooth, monotone, no Runge overshoot)
      - HFI / FWI      : linear (daily fire-weather indices, slow-varying)
      - Fire counts/FRP: forward-fill  (satellite pass is piecewise-constant)
      - Calendar fields: recomputed exactly from new timestamps
      - All lag/rolling: recomputed from the interpolated PM2.5 series
    """
    _header(f"PCHIP  —  hourly → {freq}")
    t0 = time.time()

    zone_dfs = []
    for zone in sorted(df["Zone"].unique()):
        print(f"  [{zone}]", end=" ", flush=True)
        d = (df[df["Zone"] == zone]
             .sort_values("Datetime_UTC")
             .reset_index(drop=True))

        t_ns     = d["Datetime_UTC"].astype(np.int64).values
        new_idx  = pd.date_range(
            d["Datetime_UTC"].iloc[0],
            d["Datetime_UTC"].iloc[-1],
            freq=freq, tz="UTC",
        )
        t_new_ns = new_idx.astype(np.int64).values

        out       = pd.DataFrame({"Datetime_UTC": new_idx})
        out["Zone"] = zone

        # PM2.5 — PCHIP
        pm25_vals  = d["PM25"].ffill().bfill().values
        out["PM25"] = PchipInterpolator(t_ns, pm25_vals)(t_new_ns).clip(0)

        # Daily fire-weather indices — linear
        for col in ["hfi_weighted", "fwi_mean"]:
            vals    = d[col].ffill().bfill().fillna(0).values
            out[col] = np.interp(t_new_ns, t_ns, vals).clip(0)

        # Satellite-based fire observations — forward-fill (step function)
        for col in ["fire_count_regional", "frp_regional_sum",
                    "fire_count_local",    "frp_local_sum"]:
            vals    = d[col].ffill().bfill().fillna(0).values
            out[col] = np.interp(t_new_ns, t_ns, vals).clip(0)

        # Calendar features — exact from new timestamps
        # Use fractional hour (e.g. 0.5 for 00:30) so sub-hourly info is preserved
        out["hour"]               = (out["Datetime_UTC"].dt.hour
                                     + out["Datetime_UTC"].dt.minute / 60.0)
        out["day_of_week"]        = out["Datetime_UTC"].dt.dayofweek
        out["month"]              = out["Datetime_UTC"].dt.month
        out["is_wildfire_season"] = out["month"].isin([5, 6, 7, 8, 9]).astype(int)
        out["season"]             = _season(out["month"])
        out["alert"]              = (out["PM25"] >= 25.0).astype(int)

        # Derived lag / rolling features
        out = _recompute_lag_features(out, freq)

        zone_dfs.append(out)
        print(f"{len(out):,} rows")

    result   = pd.concat(zone_dfs, ignore_index=True)
    out_path = OUTPUT_DIR / "CS2_pchip30min.csv"
    result.to_csv(out_path, index=False)

    elapsed = time.time() - t0
    print(f"\n  Saved {len(result):,} rows  ->  {out_path.name}  ({elapsed:.1f}s)")
    print(f"  Tip: use SEQ_LEN=144 in the notebook (= 72 h at 30-min resolution).")
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Conditional VAE
# ═══════════════════════════════════════════════════════════════════════════════

class _CVAE(nn.Module):
    """MLP Conditional VAE over flattened (SEQ_LEN x N_FEAT) sequences.

    Condition vector c (dim 2):
      c[0] = mean is_wildfire_season over the sequence
      c[1] = log1p(mean frp_local_sum over the sequence)

    This lets us oversample high-fire-intensity sequences at generation time
    by biasing c toward the high-FRP regime.
    """

    def __init__(self, seq_len: int, n_feat: int,
                 cond_dim: int = 2, latent_dim: int = 64, hidden: int = 512):
        super().__init__()
        flat           = seq_len * n_feat
        self.seq_len   = seq_len
        self.n_feat    = n_feat
        self.latent_dim = latent_dim

        self.enc = nn.Sequential(
            nn.Linear(flat + cond_dim, hidden),
            nn.LayerNorm(hidden), nn.GELU(),
            nn.Linear(hidden, hidden // 2),
            nn.LayerNorm(hidden // 2), nn.GELU(),
        )
        self.mu_head  = nn.Linear(hidden // 2, latent_dim)
        self.var_head = nn.Linear(hidden // 2, latent_dim)

        self.dec = nn.Sequential(
            nn.Linear(latent_dim + cond_dim, hidden // 2),
            nn.LayerNorm(hidden // 2), nn.GELU(),
            nn.Linear(hidden // 2, hidden),
            nn.LayerNorm(hidden), nn.GELU(),
            nn.Linear(hidden, flat),
        )

    def encode(self, x_flat: torch.Tensor, cond: torch.Tensor):
        h      = self.enc(torch.cat([x_flat, cond], dim=-1))
        return self.mu_head(h), self.var_head(h)

    def decode(self, z: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        out = self.dec(torch.cat([z, cond], dim=-1))
        return out.view(-1, self.seq_len, self.n_feat)

    def forward(self, x: torch.Tensor, cond: torch.Tensor):
        """x : (B, seq_len, n_feat)"""
        x_flat        = x.view(x.size(0), -1)
        mu, logvar    = self.encode(x_flat, cond)
        std           = (0.5 * logvar).exp()
        z             = mu + std * torch.randn_like(std)
        return self.decode(z, cond), mu, logvar


def _cvae_loss(recon, x, mu, logvar, beta: float = 0.5):
    recon_l = F.mse_loss(recon, x)
    kl      = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).mean()
    return recon_l + beta * kl


def _make_cond_raw(X_raw: np.ndarray) -> np.ndarray:
    """Build raw (unscaled) 2-D condition vector from original-scale sequences."""
    frp_idx = RAW_FEATURES.index("frp_local_sum")
    ws_idx  = RAW_FEATURES.index("is_wildfire_season")
    mean_ws  = X_raw[:, :, ws_idx].mean(axis=1)
    mean_frp = X_raw[:, :, frp_idx].mean(axis=1)
    return np.stack([mean_ws, np.log1p(mean_frp)], axis=1).astype(np.float32)


def generate_cvae(
    df: pd.DataFrame,
    epochs: int        = 150,
    batch_size: int    = 256,
    lr: float          = 1e-3,
    latent_dim: int    = 64,
    beta: float        = 0.5,
    syn_multiplier: float = 1.0,
) -> dict:
    """Train a per-zone CVAE and generate synthetic sequences.

    For each zone:
      - Fits on 2022-2024 training sequences
      - Generates (syn_multiplier x N_real) synthetic sequences
      - Boosts fire-event sequences: 25% of synthetic are sampled
        with conditions from the top-20% FRP sequences
    """
    _header(f"CVAE  —  epochs={epochs}, latent={latent_dim}, syn_mult={syn_multiplier}x")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    t0      = time.time()
    archive = {}

    for zone in sorted(df["Zone"].unique()):
        key   = _zone_key(zone)
        print(f"\n  [{zone}]")

        train = df[(df["Zone"] == zone) &
                   (df["Datetime_UTC"].dt.year.isin(TRAIN_YEARS))].copy()

        # ── Build sequences (original scale) ──────────────────────────────────
        X_raw, y_raw = _build_sequences(train, RAW_FEATURES)
        N, T, F      = X_raw.shape
        print(f"    Real sequences : {N:,}  shape ({T}, {F})")

        # ── Condition vectors (unscaled, then standardised) ────────────────────
        cond_raw    = _make_cond_raw(X_raw)           # (N, 2)
        cond_scaler = StandardScaler()
        cond_scaled = cond_scaler.fit_transform(cond_raw).astype(np.float32)

        # ── Standardise features ──────────────────────────────────────────────
        feat_scaler = StandardScaler()
        X_scaled    = feat_scaler.fit_transform(
            X_raw.reshape(-1, F)
        ).reshape(N, T, F).astype(np.float32)

        # ── DataLoader ────────────────────────────────────────────────────────
        loader = DataLoader(
            TensorDataset(
                torch.tensor(X_scaled),
                torch.tensor(cond_scaled),
            ),
            batch_size=batch_size, shuffle=True,
            pin_memory=(DEVICE.type == "cuda"),
        )

        # ── Model ─────────────────────────────────────────────────────────────
        model = _CVAE(T, F, cond_dim=2, latent_dim=latent_dim).to(DEVICE)
        opt   = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

        # ── Training loop ─────────────────────────────────────────────────────
        log_every = max(1, epochs // 5)   # print ~5 times total
        t_zone    = time.time()
        for ep in range(1, epochs + 1):
            model.train()
            ep_loss = 0.0
            for xb, cb in loader:
                xb, cb = xb.to(DEVICE), cb.to(DEVICE)
                opt.zero_grad(set_to_none=True)
                recon, mu, logvar = model(xb, cb)
                loss = _cvae_loss(recon, xb, mu, logvar, beta)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                ep_loss += loss.item()
            sched.step()
            elapsed = time.time() - t_zone
            _pbar(ep, epochs, prefix=f"ep")
            if ep % log_every == 0 or ep == epochs:
                print(f"    ep {ep:03d}/{epochs} | "
                      f"loss {ep_loss / len(loader):.4f} | "
                      f"{_eta(elapsed, ep, epochs)}"
                      f"{_gpu_mem()}")

        torch.save(model.state_dict(), MODEL_DIR / f"cvae_{key}.pt")

        # ── Generate synthetic sequences ──────────────────────────────────────
        n_syn       = int(N * syn_multiplier)
        n_fire_boost = n_syn // 4    # 25% from high-FRP regime

        # Mask of top-20% FRP sequences (original scale)
        frp_threshold  = np.percentile(cond_raw[:, 1], 80)
        hi_fire_mask   = cond_raw[:, 1] >= frp_threshold

        model.eval()
        with torch.no_grad():
            # Base: resample conditions from training distribution
            idx_base   = np.random.choice(N, n_syn, replace=True)
            cond_base  = torch.tensor(
                cond_scaled[idx_base], dtype=torch.float32, device=DEVICE
            )
            z_base     = torch.randn(n_syn, latent_dim, device=DEVICE)
            X_syn_base = model.decode(z_base, cond_base).cpu().numpy()

            # Fire boost: sample only from high-FRP conditions
            src_pool   = np.where(hi_fire_mask)[0] if hi_fire_mask.sum() > 0 \
                         else np.arange(N)
            idx_fire   = np.random.choice(src_pool, n_fire_boost, replace=True)
            cond_fire  = torch.tensor(
                cond_scaled[idx_fire], dtype=torch.float32, device=DEVICE
            )
            z_fire     = torch.randn(n_fire_boost, latent_dim, device=DEVICE)
            X_syn_fire = model.decode(z_fire, cond_fire).cpu().numpy()

        X_syn_scaled = np.concatenate([X_syn_base, X_syn_fire], axis=0)

        # Inverse-transform back to original feature space
        n_total = X_syn_scaled.shape[0]
        X_syn   = feat_scaler.inverse_transform(
            X_syn_scaled.reshape(-1, F)
        ).reshape(n_total, T, F).astype(np.float32)
        X_syn[:, :, PM25_IDX] = X_syn[:, :, PM25_IDX].clip(0)

        # y_syn: PM2.5 at the last timestep (horizon-1 approximation — valid for H=1
        # since PM2.5 changes slowly; the transformer will use these as training targets)
        y_syn = X_syn[:, -1, PM25_IDX].astype(np.float32)

        archive[f"{key}_X_real"] = X_raw
        archive[f"{key}_y_real"] = y_raw
        archive[f"{key}_X_syn"]  = X_syn
        archive[f"{key}_y_syn"]  = y_syn
        print(f"    Synthetic seqs : {n_total:,}  "
              f"(incl. {n_fire_boost} fire-boosted)")

    out_path = OUTPUT_DIR / "CS2_cvae_aug.npz"
    np.savez_compressed(out_path, **archive)
    print(f"\n  Saved -> {out_path.name}  ({(time.time()-t0)/60:.1f} min)")
    return archive


# ═══════════════════════════════════════════════════════════════════════════════
# 3. TimeGAN
# ═══════════════════════════════════════════════════════════════════════════════

# Five-network architecture from Yoon et al. (2019).
# All networks use 3-layer GRUs except the Supervisor (2 layers).
# The Discriminator is bidirectional for stronger signal discrimination.

class _TGEmbedder(nn.Module):
    """X  (B,T,F)  ->  H  (B,T,Z)   Maps real data to latent space."""
    def __init__(self, n_feat: int, hidden: int, z_dim: int):
        super().__init__()
        self.rnn  = nn.GRU(n_feat, hidden, num_layers=3,
                           batch_first=True, dropout=0.1)
        self.proj = nn.Linear(hidden, z_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, _ = self.rnn(x)
        return torch.sigmoid(self.proj(h))


class _TGRecovery(nn.Module):
    """H  (B,T,Z)  ->  X_hat  (B,T,F)   Reconstructs data from latent."""
    def __init__(self, z_dim: int, hidden: int, n_feat: int):
        super().__init__()
        self.rnn  = nn.GRU(z_dim, hidden, num_layers=3,
                           batch_first=True, dropout=0.1)
        self.proj = nn.Linear(hidden, n_feat)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        out, _ = self.rnn(h)
        return self.proj(out)


class _TGGenerator(nn.Module):
    """Z_noise  (B,T,noise_dim)  ->  E_hat  (B,T,Z)   Generates fake latents."""
    def __init__(self, noise_dim: int, hidden: int, z_dim: int):
        super().__init__()
        self.rnn  = nn.GRU(noise_dim, hidden, num_layers=3,
                           batch_first=True, dropout=0.1)
        self.proj = nn.Linear(hidden, z_dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h, _ = self.rnn(z)
        return torch.sigmoid(self.proj(h))


class _TGSupervisor(nn.Module):
    """H_t  ->  H_{t+1}   Enforces temporal step-consistency in latent space."""
    def __init__(self, z_dim: int, hidden: int):
        super().__init__()
        self.rnn  = nn.GRU(z_dim, hidden, num_layers=2, batch_first=True)
        self.proj = nn.Linear(hidden, z_dim)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        out, _ = self.rnn(h)
        return torch.sigmoid(self.proj(out))


class _TGDiscriminator(nn.Module):
    """H  (B,T,Z)  ->  logit  (B,1)   Classifies real vs. fake latent sequences."""
    def __init__(self, z_dim: int, hidden: int):
        super().__init__()
        self.rnn  = nn.GRU(z_dim, hidden, num_layers=2,
                           batch_first=True, bidirectional=True)
        self.head = nn.Linear(hidden * 2, 1)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        out, _ = self.rnn(h)
        return self.head(out[:, -1, :])  # (B, 1)


def generate_timegan(
    df: pd.DataFrame,
    phase1_steps: int   = 600,
    phase2_steps: int   = 300,
    phase3_steps: int   = 2000,
    batch_size: int     = 128,
    hidden_dim: int     = 64,
    z_dim: int          = 64,
    noise_dim: int      = 32,
    syn_multiplier: float = 1.0,
) -> dict:
    """Train TimeGAN per zone (Yoon et al. 2019) and generate synthetic sequences.

    Three training phases:
      Phase 1 — Autoencoder pre-training (Embedder E + Recovery R)
                Loss: 10 * MSE(X, R(E(X)))

      Phase 2 — Supervisor pre-training (on real embeddings)
                Loss: MSE(H[:,1:], S(H[:,:-1]))

      Phase 3 — Joint adversarial training
                Generator G: fool D, match supervisor steps, match moments
                Discriminator D: distinguish E(real) from S(G(noise))
                                 — updated only when D loss > 0.15 threshold
                Embedder E: reconstruction + supervisor consistency
    """
    _header(
        f"TimeGAN  —  p1={phase1_steps}  p2={phase2_steps}  "
        f"p3={phase3_steps}  syn_mult={syn_multiplier}x"
    )
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    t0      = time.time()
    archive = {}

    for zone in sorted(df["Zone"].unique()):
        key   = _zone_key(zone)
        print(f"\n  [{zone}]")

        train = df[(df["Zone"] == zone) &
                   (df["Datetime_UTC"].dt.year.isin(TRAIN_YEARS))].copy()

        X_raw, y_raw = _build_sequences(train, RAW_FEATURES)
        N, T, F      = X_raw.shape
        print(f"    Real sequences : {N:,}  shape ({T}, {F})")

        # TimeGAN works best with [0,1]-normalised inputs (per feature)
        scaler   = MinMaxScaler()
        X_scaled = scaler.fit_transform(
            X_raw.reshape(-1, F)
        ).reshape(N, T, F).astype(np.float32)

        X_t     = torch.tensor(X_scaled)
        loader  = DataLoader(
            TensorDataset(X_t),
            batch_size=batch_size, shuffle=True,
            pin_memory=(DEVICE.type == "cuda"),
        )
        loader_cycle = cycle(loader)

        def _batch():
            """Pull one batch from the infinite cycle."""
            (xb,) = next(loader_cycle)
            return xb.to(DEVICE)

        def _noise(b: int) -> torch.Tensor:
            return torch.randn(b, T, noise_dim, device=DEVICE)

        # ── Instantiate networks ──────────────────────────────────────────────
        E = _TGEmbedder    (F,         hidden_dim, z_dim).to(DEVICE)
        R = _TGRecovery    (z_dim,     hidden_dim, F    ).to(DEVICE)
        G = _TGGenerator   (noise_dim, hidden_dim, z_dim).to(DEVICE)
        S = _TGSupervisor  (z_dim,     hidden_dim       ).to(DEVICE)
        D = _TGDiscriminator(z_dim,    hidden_dim       ).to(DEVICE)

        opt_ER = torch.optim.Adam(
            list(E.parameters()) + list(R.parameters()), lr=1e-3)
        opt_S  = torch.optim.Adam(S.parameters(), lr=1e-3)
        opt_GS = torch.optim.Adam(
            list(G.parameters()) + list(S.parameters()), lr=1e-3)
        opt_D  = torch.optim.Adam(D.parameters(), lr=1e-3)
        opt_E2 = torch.optim.Adam(
            list(E.parameters()) + list(R.parameters()), lr=1e-4)

        bce = nn.BCEWithLogitsLoss()
        mse = nn.MSELoss()

        # ── Phase 1: Autoencoder ──────────────────────────────────────────────
        print(f"    Phase 1/3  Autoencoder    ({phase1_steps} steps)")
        t_p1      = time.time()
        log_p1    = max(1, phase1_steps // 5)
        p1_loss   = 0.0
        for step in range(1, phase1_steps + 1):
            xb = _batch()
            opt_ER.zero_grad(set_to_none=True)
            loss = 10 * mse(R(E(xb)), xb)
            loss.backward()
            opt_ER.step()
            p1_loss += loss.item()
            _pbar(step, phase1_steps, prefix="p1")
            if step % log_p1 == 0 or step == phase1_steps:
                print(f"    step {step:4d}/{phase1_steps} | "
                      f"recon_loss {p1_loss/log_p1:.4f} | "
                      f"{_eta(time.time()-t_p1, step, phase1_steps)}"
                      f"{_gpu_mem()}")
                p1_loss = 0.0

        # ── Phase 2: Supervisor ───────────────────────────────────────────────
        print(f"    Phase 2/3  Supervisor     ({phase2_steps} steps)")
        t_p2      = time.time()
        log_p2    = max(1, phase2_steps // 5)
        p2_loss   = 0.0
        for step in range(1, phase2_steps + 1):
            xb = _batch()
            opt_S.zero_grad(set_to_none=True)
            with torch.no_grad():
                H = E(xb)
            loss = mse(H[:, 1:, :], S(H[:, :-1, :]))
            loss.backward()
            opt_S.step()
            p2_loss += loss.item()
            _pbar(step, phase2_steps, prefix="p2")
            if step % log_p2 == 0 or step == phase2_steps:
                print(f"    step {step:4d}/{phase2_steps} | "
                      f"sup_loss {p2_loss/log_p2:.4f} | "
                      f"{_eta(time.time()-t_p2, step, phase2_steps)}"
                      f"{_gpu_mem()}")
                p2_loss = 0.0

        # ── Phase 3: Joint training ───────────────────────────────────────────
        print(f"    Phase 3/3  Joint GAN      ({phase3_steps} steps)")
        log_every = max(1, phase3_steps // 5)
        t_p3      = time.time()

        for step in range(1, phase3_steps + 1):
            xb = _batch()
            B  = xb.size(0)

            # ---- Generator + Supervisor update ----
            opt_GS.zero_grad(set_to_none=True)
            E_hat = G(_noise(B))
            H_hat = S(E_hat)

            # Adversarial: fool discriminator
            G_adv = bce(D(H_hat), torch.ones(B, 1, device=DEVICE))

            # Step-consistency: synthetic latents should respect temporal order
            G_sup = mse(S(E_hat[:, :-1, :]), E_hat[:, 1:, :])

            # Moment matching: mean & variance of fake vs. real latents
            with torch.no_grad():
                H_real = E(xb)
            G_mu  = (H_hat.mean(dim=[0, 1]) - H_real.mean(dim=[0, 1])).abs().mean()
            G_var = (H_hat.var (dim=[0, 1]) - H_real.var (dim=[0, 1])).abs().mean()

            G_loss = G_adv + 100.0 * G_sup + 100.0 * (G_mu + G_var)
            G_loss.backward()
            nn.utils.clip_grad_norm_(
                list(G.parameters()) + list(S.parameters()), 1.0)
            opt_GS.step()

            # ---- Discriminator update (conditional) ----
            opt_D.zero_grad(set_to_none=True)
            with torch.no_grad():
                H_real_d = E(xb)
                H_fake_d = S(G(_noise(B)))
            D_real = bce(D(H_real_d), torch.ones (B, 1, device=DEVICE))
            D_fake = bce(D(H_fake_d), torch.zeros(B, 1, device=DEVICE))
            D_loss = D_real + D_fake
            # Skip discriminator update when it's already very strong —
            # prevents the generator from collapsing early
            if D_loss.item() > 0.15:
                D_loss.backward()
                opt_D.step()

            # ---- Embedder update (joint) ----
            opt_E2.zero_grad(set_to_none=True)
            H_e    = E(xb)
            E_loss = (10.0 * mse(R(H_e), xb)
                      + 0.1 * mse(H_e[:, 1:, :], S(H_e[:, :-1, :])))
            E_loss.backward()
            opt_E2.step()

            _pbar(step, phase3_steps, prefix="p3")
            if step % log_every == 0 or step == phase3_steps:
                print(f"    step {step:4d}/{phase3_steps} | "
                      f"G={G_loss.item():.4f}  "
                      f"D={D_loss.item():.4f}  "
                      f"E={E_loss.item():.4f} | "
                      f"{_eta(time.time()-t_p3, step, phase3_steps)}"
                      f"{_gpu_mem()}")

        # Save model weights for reproducibility / inspection
        torch.save(
            {"E": E.state_dict(), "R": R.state_dict(),
             "G": G.state_dict(), "S": S.state_dict(),
             "D": D.state_dict()},
            MODEL_DIR / f"timegan_{key}.pt",
        )

        # ── Generate synthetic sequences ──────────────────────────────────────
        n_syn = int(N * syn_multiplier)
        E.eval(); R.eval(); G.eval(); S.eval()

        chunks = []
        with torch.no_grad():
            for i in range(0, n_syn, batch_size):
                b     = min(batch_size, n_syn - i)
                E_hat = G(_noise(b))
                H_hat = S(E_hat)
                chunks.append(R(H_hat).cpu().numpy())

        X_syn_scaled = np.concatenate(chunks, axis=0)   # (n_syn, T, F), in [0,1]
        X_syn = scaler.inverse_transform(
            X_syn_scaled.reshape(-1, F)
        ).reshape(n_syn, T, F).astype(np.float32)
        X_syn[:, :, PM25_IDX] = X_syn[:, :, PM25_IDX].clip(0)
        y_syn = X_syn[:, -1, PM25_IDX].astype(np.float32)

        archive[f"{key}_X_real"] = X_raw
        archive[f"{key}_y_real"] = y_raw
        archive[f"{key}_X_syn"]  = X_syn
        archive[f"{key}_y_syn"]  = y_syn
        print(f"    Synthetic seqs : {n_syn:,}")

    out_path = OUTPUT_DIR / "CS2_timegan_aug.npz"
    np.savez_compressed(out_path, **archive)
    print(f"\n  Saved -> {out_path.name}  ({(time.time()-t0)/60:.1f} min)")
    return archive


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(
        description="Synthetic / augmented data generator for the PM2.5 transformer",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--pchip",   action="store_true",
                    help="PCHIP 30-min temporal upsampling")
    ap.add_argument("--cvae",    action="store_true",
                    help="Conditional VAE sequence augmentation")
    ap.add_argument("--timegan", action="store_true",
                    help="TimeGAN sequence augmentation")
    ap.add_argument("--all",     action="store_true",
                    help="Run all three methods")
    ap.add_argument("--syn-multiplier", type=float, default=1.0, metavar="M",
                    help="Ratio of synthetic to real sequences (CVAE/TimeGAN)")
    # Advanced overrides
    ap.add_argument("--cvae-epochs",    type=int,   default=150)
    ap.add_argument("--cvae-latent",    type=int,   default=64)
    ap.add_argument("--tgan-p1",        type=int,   default=600,
                    help="TimeGAN Phase 1 steps (autoencoder)")
    ap.add_argument("--tgan-p2",        type=int,   default=300,
                    help="TimeGAN Phase 2 steps (supervisor)")
    ap.add_argument("--tgan-p3",        type=int,   default=2000,
                    help="TimeGAN Phase 3 steps (joint GAN)")
    ap.add_argument("--tgan-hidden",    type=int,   default=64)
    ap.add_argument("--pchip-freq",     type=str,   default="30min",
                    help="Target frequency for PCHIP (e.g. '30min', '15min')")
    args = ap.parse_args()

    run_pchip   = args.all or args.pchip
    run_cvae    = args.all or args.cvae
    run_timegan = args.all or args.timegan

    if not (run_pchip or run_cvae or run_timegan):
        ap.print_help()
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    _device_info()
    print(f"  Input  : {INPUT_CSV}")
    df = _load_and_prep(INPUT_CSV)
    print(f"Loaded  : {len(df):,} rows | zones: {sorted(df['Zone'].unique())}")

    if run_pchip:
        generate_pchip(df, freq=args.pchip_freq)

    if run_cvae:
        generate_cvae(
            df,
            epochs=args.cvae_epochs,
            latent_dim=args.cvae_latent,
            syn_multiplier=args.syn_multiplier,
        )

    if run_timegan:
        generate_timegan(
            df,
            phase1_steps=args.tgan_p1,
            phase2_steps=args.tgan_p2,
            phase3_steps=args.tgan_p3,
            hidden_dim=args.tgan_hidden,
            syn_multiplier=args.syn_multiplier,
        )

    print("\nAll done.")
    print(f"Outputs written to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
