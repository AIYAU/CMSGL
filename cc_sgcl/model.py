"""Standalone CC-SGCL model definition.

This file contains only the CC-SGCL method code. It is independent from the
original HyLiOSR model implementation.
"""

from __future__ import annotations

import math
import os
import shutil
import sys
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


_WKV6_CUDA_MODULE = None
_WKV6_CUDA_LOAD_ERROR = None


def _find_vision_rwkv_root() -> Path | None:
    env_root = os.environ.get("COSGL_VISION_RWKV_ROOT")
    if env_root:
        candidate = Path(env_root).expanduser().resolve()
        if candidate.exists():
            return candidate
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "optional_deps" / "Vision-RWKV"
        if candidate.exists():
            return candidate
    return None


def _load_wkv6_cuda(head_size: int, token_limit: int = 256):
    """Load the official Vision-RWKV6 CUDA kernel on demand."""

    global _WKV6_CUDA_MODULE, _WKV6_CUDA_LOAD_ERROR
    if _WKV6_CUDA_MODULE is not None:
        return _WKV6_CUDA_MODULE
    if _WKV6_CUDA_LOAD_ERROR is not None:
        return None
    root = _find_vision_rwkv_root()
    if root is None:
        _WKV6_CUDA_LOAD_ERROR = "Vision-RWKV repository not found"
        return None
    cuda_dir = root / "classification" / "mmcls_custom" / "models" / "backbones" / "cuda_v6"
    sources = [cuda_dir / "wkv6_op.cpp", cuda_dir / "wkv6_cuda.cu"]
    if not all(path.exists() for path in sources):
        _WKV6_CUDA_LOAD_ERROR = f"Vision-RWKV6 CUDA sources not found under {cuda_dir}"
        return None
    try:
        import torch.utils.cpp_extension as cpp_extension

        temp_root = Path(os.environ.get("TEMP", ".")).resolve() / "cosgl_vrwkv6_cuda"
        source_dir = temp_root / "src"
        build_dir = temp_root / f"build_h{head_size}_t{token_limit}"
        source_dir.mkdir(parents=True, exist_ok=True)
        build_dir.mkdir(parents=True, exist_ok=True)
        local_sources = []
        for path in sources:
            target = source_dir / path.name
            shutil.copyfile(path, target)
            local_sources.append(target)

        scripts_dir = Path(sys.executable).resolve().parent / "Scripts"
        if (scripts_dir / "ninja.exe").exists():
            os.environ["PATH"] = str(scripts_dir) + os.pathsep + os.environ.get("PATH", "")
        if sys.platform.startswith("win"):
            cpp_extension.SUBPROCESS_DECODE_ARGS = ("utf-8", "ignore")
        _WKV6_CUDA_MODULE = cpp_extension.load(
            name=f"cosgl_wkv6_h{head_size}_t{token_limit}",
            sources=[str(path) for path in local_sources],
            build_directory=str(build_dir),
            verbose=False,
            extra_cuda_cflags=[
                "-res-usage",
                "--use_fast_math",
                "-O3",
                "-Xptxas",
                "-O3",
                "--extra-device-vectorization",
                f"-D_N_={head_size}",
                f"-D_T_={token_limit}",
            ],
        )
    except Exception as exc:  # pragma: no cover - depends on local CUDA toolchain.
        _WKV6_CUDA_LOAD_ERROR = str(exc)
        return None
    return _WKV6_CUDA_MODULE


class _OfficialWKV6Cuda(torch.autograd.Function):
    @staticmethod
    def forward(ctx, r: torch.Tensor, k: torch.Tensor, v: torch.Tensor, w: torch.Tensor, u: torch.Tensor, num_heads: int):
        batch_size, tokens, dim = r.shape
        head_size = dim // num_heads
        module = _load_wkv6_cuda(head_size)
        if module is None:
            raise RuntimeError(_WKV6_CUDA_LOAD_ERROR or "WKV6 CUDA extension is unavailable")
        r_f = r.float().contiguous()
        k_f = k.float().contiguous()
        v_f = v.float().contiguous()
        ew = (-torch.exp(w.float())).contiguous()
        u_f = u.float().contiguous()
        y = torch.empty((batch_size, tokens, dim), device=r.device, dtype=torch.float32)
        module.forward(batch_size, tokens, dim, num_heads, r_f, k_f, v_f, ew, u_f, y)
        ctx.save_for_backward(r_f, k_f, v_f, ew, u_f)
        ctx.shape = (batch_size, tokens, dim, num_heads)
        return y.to(dtype=r.dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        batch_size, tokens, dim, num_heads = ctx.shape
        head_size = dim // num_heads
        module = _load_wkv6_cuda(head_size)
        if module is None:
            raise RuntimeError(_WKV6_CUDA_LOAD_ERROR or "WKV6 CUDA extension is unavailable")
        r, k, v, ew, u = ctx.saved_tensors
        gy = grad_output.float().contiguous()
        gr = torch.empty((batch_size, tokens, dim), device=gy.device, dtype=torch.float32)
        gk = torch.empty_like(gr)
        gv = torch.empty_like(gr)
        gw = torch.empty_like(gr)
        gu = torch.empty((batch_size, dim), device=gy.device, dtype=torch.float32)
        module.backward(batch_size, tokens, dim, num_heads, r, k, v, ew, u, gy, gr, gk, gv, gw, gu)
        gu = torch.sum(gu, 0).view(num_heads, head_size)
        return gr, gk, gv, gw, gu, None


def variant_enabled(method_variant: str, feature: str) -> bool:
    """Return whether a method feature is active for a named CoSGL variant."""

    table = {
        "mp": {"mp"},
        "np": {"np"},
        "ds": {"ds"},
        "uf": {"uf"},
        "mp_np": {"mp", "np"},
        "mp_ds": {"mp", "ds"},
        "mp_uf": {"mp", "uf"},
        "np_ds": {"np", "ds"},
        "np_uf": {"np", "uf"},
        "all": {"np", "ds", "uf"},
        "manr": {"mp", "np", "ds", "uf"},
    }
    return feature in table.get(method_variant, set())


class HSIEncoder(nn.Module):
    """Lightweight spectral-spatial encoder for HSI patches."""

    def __init__(self, in_channels: int, feat_dim: int) -> None:
        super().__init__()
        hidden = max(feat_dim // 2, 32)
        self.spectral_3d = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=(7, 3, 3), padding=(3, 1, 1), bias=False),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
            nn.Conv3d(16, 32, kernel_size=(5, 3, 3), padding=(2, 1, 1), bias=False),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
        )
        self.spatial_2d = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, feat_dim),
        )

    def forward(self, x_h: torch.Tensor) -> torch.Tensor:
        assert x_h.ndim == 4, f"Expected HSI tensor [B, C_h, P, P], got {x_h.shape}"
        x_h = x_h.unsqueeze(1)
        feats_3d = self.spectral_3d(x_h)
        feats_2d = feats_3d.mean(dim=2)
        pooled = self.spatial_2d(feats_2d)
        return self.proj(pooled)


class SpectralRWKVBlock(nn.Module):
    """Lightweight RWKV-style time mixing along the spectral axis."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.time_mix_k = nn.Parameter(torch.rand(1, 1, dim))
        self.time_mix_v = nn.Parameter(torch.rand(1, 1, dim))
        self.time_mix_r = nn.Parameter(torch.rand(1, 1, dim))
        self.time_decay = nn.Parameter(torch.zeros(dim))
        self.key = nn.Linear(dim, dim, bias=False)
        self.value = nn.Linear(dim, dim, bias=False)
        self.receptance = nn.Linear(dim, dim, bias=False)
        self.output = nn.Linear(dim, dim, bias=False)
        self.channel = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x_norm = self.norm(x)
        shifted = torch.cat([x_norm[:, :1], x_norm[:, :-1]], dim=1)
        xk = x_norm * self.time_mix_k + shifted * (1.0 - self.time_mix_k)
        xv = x_norm * self.time_mix_v + shifted * (1.0 - self.time_mix_v)
        xr = x_norm * self.time_mix_r + shifted * (1.0 - self.time_mix_r)

        k = torch.tanh(self.key(xk))
        v = self.value(xv)
        r = torch.sigmoid(self.receptance(xr))
        decay = torch.sigmoid(self.time_decay).view(1, 1, -1)
        state = torch.zeros_like(v[:, 0])
        mixed = []
        for idx in range(v.shape[1]):
            state = decay.squeeze(1) * state + (1.0 - decay.squeeze(1)) * (k[:, idx] * v[:, idx])
            mixed.append((r[:, idx] * state).unsqueeze(1))
        y = self.output(torch.cat(mixed, dim=1))
        x = residual + y
        return x + self.channel(x)


class SpectralRWKVHSIEncoder(nn.Module):
    """HSI encoder with RWKV-like spectral token mixing and local spatial pooling."""

    def __init__(self, in_channels: int, feat_dim: int, num_layers: int = 2) -> None:
        super().__init__()
        hidden = max(feat_dim, 64)
        self.band_spatial = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, hidden, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.band_pos = nn.Parameter(torch.zeros(1, in_channels, hidden))
        self.blocks = nn.ModuleList([SpectralRWKVBlock(hidden) for _ in range(num_layers)])
        self.proj = nn.Sequential(
            nn.LayerNorm(hidden * 2),
            nn.Linear(hidden * 2, feat_dim),
        )

    def forward(self, x_h: torch.Tensor) -> torch.Tensor:
        assert x_h.ndim == 4, f"Expected HSI tensor [B, C_h, P, P], got {x_h.shape}"
        batch_size, channels, height, width = x_h.shape
        tokens = self.band_spatial(x_h.reshape(batch_size * channels, 1, height, width))
        tokens = tokens.flatten(1).reshape(batch_size, channels, -1)
        tokens = tokens + self.band_pos[:, :channels]
        for block in self.blocks:
            tokens = block(tokens)
        pooled = torch.cat([tokens.mean(dim=1), tokens[:, -1]], dim=-1)
        return self.proj(pooled)


def _spectral_shift_previous(x: torch.Tensor) -> torch.Tensor:
    """Shift spectral tokens by one band while preserving the first token."""

    return torch.cat([x[:, :1], x[:, :-1]], dim=1)


def _stable_rwkv_recurrence(
    w: torch.Tensor,
    u: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> torch.Tensor:
    """Pure PyTorch RWKV recurrence for short spectral sequences."""

    batch_size, tokens, dim = k.shape
    aa = torch.zeros(batch_size, dim, device=k.device, dtype=k.dtype)
    bb = torch.zeros_like(aa)
    pp = torch.full_like(aa, -1e30)
    outputs = []
    w = w.view(1, dim)
    u = u.view(1, dim)
    for idx in range(tokens):
        kk = k[:, idx]
        vv = v[:, idx]

        ww = u + kk
        p = torch.maximum(pp, ww)
        e1 = torch.exp(pp - p)
        e2 = torch.exp(ww - p)
        y = (e1 * aa + e2 * vv) / (e1 * bb + e2 + 1e-6)
        outputs.append(y.unsqueeze(1))

        ww = pp + w
        p = torch.maximum(ww, kk)
        e1 = torch.exp(ww - p)
        e2 = torch.exp(kk - p)
        aa = e1 * aa + e2 * vv
        bb = e1 * bb + e2
        pp = p
    return torch.cat(outputs, dim=1)


def _bidirectional_spectral_wkv(
    w: torch.Tensor,
    u: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> torch.Tensor:
    forward = _stable_rwkv_recurrence(w, u, k, v)
    backward = _stable_rwkv_recurrence(w, u, k.flip(1), v.flip(1)).flip(1)
    return 0.5 * (forward + backward)


def _official_bidirectional_spectral_wkv(
    w: torch.Tensor,
    u: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    chunk_size: int = 32,
) -> torch.Tensor:
    """Pure PyTorch form of Vision-RWKV bi-WKV for spectral tokens.

    The official CUDA kernel computes a channel-wise decayed mixture over all
    tokens, with a special first-token bias for the query token itself.  The
    chunked implementation keeps memory bounded for long hyperspectral bands.
    """

    batch_size, tokens, dim = k.shape
    del batch_size
    pos = torch.arange(tokens, device=k.device, dtype=k.dtype)
    w = w.view(1, 1, 1, dim)
    u = u.view(1, dim)
    outputs = []
    for start in range(0, tokens, chunk_size):
        end = min(start + chunk_size, tokens)
        rows = torch.arange(start, end, device=k.device)
        dist = (pos[start:end, None] - pos[None, :]).abs()
        scores = k[:, None, :, :] - dist.view(1, end - start, tokens, 1) * w
        local_rows = torch.arange(end - start, device=k.device)
        scores[:, local_rows, rows, :] = k[:, rows, :] + u
        weights = torch.softmax(scores.float(), dim=2).to(k.dtype)
        outputs.append((weights * v[:, None, :, :]).sum(dim=2))
    return torch.cat(outputs, dim=1)


def _spectral_wkv6_bidirectional(
    r: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    num_heads: int,
    use_cuda: bool = False,
) -> torch.Tensor:
    """Pure PyTorch VRWKV6 WKV adapted from the official bidirectional kernel.

    VRWKV6 keeps a per-head key-value state matrix.  HSI spectral sequences are
    short enough that this reference implementation is practical for pilots.
    """

    batch_size, tokens, dim = r.shape
    head_dim = dim // num_heads
    if use_cuda and r.is_cuda and head_dim == 64 and tokens <= 256:
        return _OfficialWKV6Cuda.apply(r, k, v, w, u, num_heads)

    r = r.float().view(batch_size, tokens, num_heads, head_dim)
    k = k.float().view(batch_size, tokens, num_heads, head_dim)
    v = v.float().view(batch_size, tokens, num_heads, head_dim)
    decay = torch.exp(-torch.exp(w.float().view(batch_size, tokens, num_heads, head_dim)))
    u = u.float().view(1, 1, num_heads, head_dim, 1)

    state_prev = torch.zeros(batch_size, num_heads, head_dim, head_dim, device=r.device, dtype=torch.float32)
    forward = []
    for idx in range(tokens):
        kt = k[:, idx]
        vt = v[:, idx]
        rt = r[:, idx]
        self_state = kt.unsqueeze(-1) * vt.unsqueeze(-2)
        y = torch.einsum("bhn,bhno->bho", rt, state_prev + u.squeeze(1) * self_state)
        forward.append(y.unsqueeze(1))
        state_prev = state_prev * decay[:, idx].unsqueeze(-1) + self_state

    state_next = torch.zeros_like(state_prev)
    backward = [None] * tokens
    for idx in range(tokens - 1, -1, -1):
        kt = k[:, idx]
        vt = v[:, idx]
        rt = r[:, idx]
        y = torch.einsum("bhn,bhno->bho", rt, state_next)
        backward[idx] = y.unsqueeze(1)
        state_next = state_next * decay[:, idx].unsqueeze(-1) + kt.unsqueeze(-1) * vt.unsqueeze(-2)

    y = torch.cat(forward, dim=1) + torch.cat(backward, dim=1)
    return y.reshape(batch_size, tokens, dim).to(v.dtype)


class SpectralVRWKVSpatialMix(nn.Module):
    """VRWKV-style spectral token mixer adapted from Vision-RWKV."""

    def __init__(
        self,
        dim: int,
        num_layers: int,
        layer_id: int,
        key_norm: bool = True,
        official_biwkv: bool = False,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.num_layers = max(num_layers, 1)
        self.layer_id = layer_id
        self.official_biwkv = official_biwkv
        self._init_vrwkv_parameters()
        self.key = nn.Linear(dim, dim, bias=False)
        self.value = nn.Linear(dim, dim, bias=False)
        self.receptance = nn.Linear(dim, dim, bias=False)
        self.output = nn.Linear(dim, dim, bias=False)
        self.key_norm = nn.LayerNorm(dim) if key_norm else None

    def _init_vrwkv_parameters(self) -> None:
        ratio_0_to_1 = self.layer_id / max(self.num_layers - 1, 1)
        ratio_1_to_almost0 = 1.0 - (self.layer_id / self.num_layers)

        decay_speed = torch.empty(self.dim)
        for idx in range(self.dim):
            denom = max(self.dim - 1, 1)
            decay_speed[idx] = -5 + 8 * (idx / denom) ** (0.7 + 1.3 * ratio_0_to_1)
        self.spectral_decay = nn.Parameter(decay_speed)

        zigzag = torch.tensor([((idx + 1) % 3 - 1) * 0.5 for idx in range(self.dim)])
        self.spectral_first = nn.Parameter(torch.ones(self.dim) * math.log(0.3) + zigzag)

        x = torch.empty(1, 1, self.dim)
        for idx in range(self.dim):
            x[0, 0, idx] = idx / self.dim
        self.spectral_mix_k = nn.Parameter(torch.pow(x, ratio_1_to_almost0))
        self.spectral_mix_v = nn.Parameter(torch.pow(x, ratio_1_to_almost0) + 0.3 * ratio_0_to_1)
        self.spectral_mix_r = nn.Parameter(torch.pow(x, 0.5 * ratio_1_to_almost0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shifted = _spectral_shift_previous(x)
        xk = x * self.spectral_mix_k + shifted * (1.0 - self.spectral_mix_k)
        xv = x * self.spectral_mix_v + shifted * (1.0 - self.spectral_mix_v)
        xr = x * self.spectral_mix_r + shifted * (1.0 - self.spectral_mix_r)

        k = self.key(xk)
        v = self.value(xv)
        r = torch.sigmoid(self.receptance(xr))
        tokens = max(x.shape[1], 1)
        u = self.spectral_first / tokens
        if self.official_biwkv:
            w = self.spectral_decay / tokens
            mixed = _official_bidirectional_spectral_wkv(w, u, k, v)
        else:
            w = -torch.exp(self.spectral_decay / tokens)
            mixed = _bidirectional_spectral_wkv(w, u, k, v)
        if self.key_norm is not None:
            mixed = self.key_norm(mixed)
        return self.output(r * mixed)


class SpectralVRWKVChannelMix(nn.Module):
    """Official VRWKV ChannelMix pattern with squared-ReLU expansion."""

    def __init__(
        self,
        dim: int,
        num_layers: int,
        layer_id: int,
        hidden_rate: int = 4,
        key_norm: bool = False,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.num_layers = max(num_layers, 1)
        self.layer_id = layer_id
        self._init_vrwkv_parameters()
        hidden = dim * hidden_rate
        self.key = nn.Linear(dim, hidden, bias=False)
        self.key_norm = nn.LayerNorm(hidden) if key_norm else None
        self.receptance = nn.Linear(dim, dim, bias=False)
        self.value = nn.Linear(hidden, dim, bias=False)

    def _init_vrwkv_parameters(self) -> None:
        ratio_1_to_almost0 = 1.0 - (self.layer_id / self.num_layers)
        x = torch.empty(1, 1, self.dim)
        for idx in range(self.dim):
            x[0, 0, idx] = idx / self.dim
        self.spectral_mix_k = nn.Parameter(torch.pow(x, ratio_1_to_almost0))
        self.spectral_mix_r = nn.Parameter(torch.pow(x, ratio_1_to_almost0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shifted = _spectral_shift_previous(x)
        xk = x * self.spectral_mix_k + shifted * (1.0 - self.spectral_mix_k)
        xr = x * self.spectral_mix_r + shifted * (1.0 - self.spectral_mix_r)
        k = torch.square(torch.relu(self.key(xk)))
        if self.key_norm is not None:
            k = self.key_norm(k)
        kv = self.value(k)
        return torch.sigmoid(self.receptance(xr)) * kv


class SpectralVRWKVBlock(nn.Module):
    """Vision-RWKV block adapted to HSI spectral tokens."""

    def __init__(
        self,
        dim: int,
        num_layers: int,
        layer_id: int,
        init_values: float = 0.1,
        official_biwkv: bool = False,
        channel_key_norm: bool = False,
    ) -> None:
        super().__init__()
        self.layer_id = layer_id
        self.ln0 = nn.LayerNorm(dim) if layer_id == 0 else None
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)
        self.att = SpectralVRWKVSpatialMix(dim, num_layers, layer_id, official_biwkv=official_biwkv)
        self.ffn = SpectralVRWKVChannelMix(dim, num_layers, layer_id, key_norm=channel_key_norm)
        self.gamma1 = nn.Parameter(init_values * torch.ones(dim))
        self.gamma2 = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.ln0 is not None:
            x = self.ln0(x)
        x = x + self.gamma1 * self.att(self.ln1(x))
        x = x + self.gamma2 * self.ffn(self.ln2(x))
        return x


class SpectralVRWKVHSIEncoder(nn.Module):
    """HSI encoder using a pure PyTorch spectral VRWKV backbone."""

    def __init__(
        self,
        in_channels: int,
        feat_dim: int,
        num_layers: int = 2,
        official_biwkv: bool = False,
        channel_key_norm: bool = False,
    ) -> None:
        super().__init__()
        hidden = max(feat_dim, 64)
        self.band_spatial = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, hidden, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.band_pos = nn.Parameter(torch.zeros(1, in_channels, hidden))
        self.blocks = nn.ModuleList(
            [
                SpectralVRWKVBlock(
                    hidden,
                    num_layers,
                    layer_id,
                    official_biwkv=official_biwkv,
                    channel_key_norm=channel_key_norm,
                )
                for layer_id in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(hidden)
        self.proj = nn.Sequential(
            nn.LayerNorm(hidden * 3),
            nn.Linear(hidden * 3, feat_dim),
        )

    def forward(self, x_h: torch.Tensor) -> torch.Tensor:
        assert x_h.ndim == 4, f"Expected HSI tensor [B, C_h, P, P], got {x_h.shape}"
        batch_size, channels, height, width = x_h.shape
        tokens = self.band_spatial(x_h.reshape(batch_size * channels, 1, height, width))
        tokens = tokens.flatten(1).reshape(batch_size, channels, -1)
        tokens = tokens + self.band_pos[:, :channels]
        for block in self.blocks:
            tokens = block(tokens)
        tokens = self.norm(tokens)
        pooled = torch.cat([tokens.mean(dim=1), tokens.amax(dim=1), tokens[:, -1]], dim=-1)
        return self.proj(pooled)


class SpectralVRWKV6SpatialMix(nn.Module):
    """VRWKV6 spectral mixer adapted from the official Vision-RWKV6 code."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_layers: int,
        layer_id: int,
        key_norm: bool = False,
        use_cuda_wkv: bool = False,
    ) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.num_layers = max(num_layers, 1)
        self.layer_id = layer_id
        self.use_cuda_wkv = use_cuda_wkv
        self._init_vrwkv6_parameters()

        self.key = nn.Linear(dim, dim, bias=False)
        self.value = nn.Linear(dim, dim, bias=False)
        self.receptance = nn.Linear(dim, dim, bias=False)
        self.gate = nn.Linear(dim, dim, bias=False)
        self.output = nn.Linear(dim, dim, bias=False)
        self.key_norm = nn.LayerNorm(dim) if key_norm else None
        self.ln_x = nn.GroupNorm(num_heads, dim, eps=1e-5)

    def _init_vrwkv6_parameters(self) -> None:
        ratio_0_to_1 = self.layer_id / max(self.num_layers - 1, 1)
        ratio_1_to_almost0 = 1.0 - (self.layer_id / self.num_layers)

        ddd = torch.empty(1, 1, self.dim)
        for idx in range(self.dim):
            ddd[0, 0, idx] = idx / self.dim
        self.time_maa_x = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))
        self.time_maa_w = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))
        self.time_maa_k = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))
        self.time_maa_v = nn.Parameter(1.0 - (torch.pow(ddd, ratio_1_to_almost0) + 0.3 * ratio_0_to_1))
        self.time_maa_r = nn.Parameter(1.0 - torch.pow(ddd, 0.5 * ratio_1_to_almost0))
        self.time_maa_g = nn.Parameter(1.0 - torch.pow(ddd, 0.5 * ratio_1_to_almost0))

        time_mix_extra_dim = 32
        self.time_maa_w1 = nn.Parameter(torch.zeros(self.dim, time_mix_extra_dim * 5).uniform_(-1e-4, 1e-4))
        self.time_maa_w2 = nn.Parameter(torch.zeros(5, time_mix_extra_dim, self.dim).uniform_(-1e-4, 1e-4))

        decay_speed = torch.empty(1, 1, self.dim)
        for idx in range(self.dim):
            denom = max(self.dim - 1, 1)
            decay_speed[0, 0, idx] = -6 + 5 * (idx / denom) ** (0.7 + 1.3 * ratio_0_to_1)
        self.time_decay = nn.Parameter(decay_speed)

        time_decay_extra_dim = 64
        self.time_decay_w1 = nn.Parameter(torch.zeros(self.dim, time_decay_extra_dim).uniform_(-1e-4, 1e-4))
        self.time_decay_w2 = nn.Parameter(torch.zeros(time_decay_extra_dim, self.dim).uniform_(-1e-4, 1e-4))

        first = torch.zeros(self.dim)
        for idx in range(self.dim):
            denom = max(self.dim - 1, 1)
            zigzag = ((idx + 1) % 3 - 1) * 0.1
            first[idx] = ratio_0_to_1 * (1 - (idx / denom)) + zigzag
        self.time_first = nn.Parameter(first.view(self.num_heads, self.head_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, tokens, _ = x.shape
        shifted_delta = _spectral_shift_previous(x) - x
        xxx = x + shifted_delta * self.time_maa_x
        dynamic_mix = torch.tanh(xxx @ self.time_maa_w1).view(batch_size * tokens, 5, -1).transpose(0, 1)
        dynamic_mix = torch.bmm(dynamic_mix, self.time_maa_w2).view(5, batch_size, tokens, self.dim)
        mw, mk, mv, mr, mg = dynamic_mix.unbind(dim=0)

        xw = x + shifted_delta * (self.time_maa_w + mw)
        xk = x + shifted_delta * (self.time_maa_k + mk)
        xv = x + shifted_delta * (self.time_maa_v + mv)
        xr = x + shifted_delta * (self.time_maa_r + mr)
        xg = x + shifted_delta * (self.time_maa_g + mg)

        r = self.receptance(xr)
        k = self.key(xk)
        v = self.value(xv)
        g = F.silu(self.gate(xg))
        w = self.time_decay + torch.tanh(xw @ self.time_decay_w1) @ self.time_decay_w2
        mixed = _spectral_wkv6_bidirectional(r, k, v, w, self.time_first, self.num_heads, use_cuda=self.use_cuda_wkv)
        if self.key_norm is not None:
            mixed = self.key_norm(mixed)
        mixed = self.ln_x(mixed.reshape(batch_size * tokens, self.dim)).view(batch_size, tokens, self.dim)
        return self.output(mixed * g)


class SpectralVRWKV6Block(nn.Module):
    """Official VRWKV6 block adapted to spectral tokens."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_layers: int,
        layer_id: int,
        init_values: float = 1e-5,
        key_norm: bool = False,
        use_cuda_wkv: bool = False,
    ) -> None:
        super().__init__()
        self.layer_id = layer_id
        self.ln0 = nn.LayerNorm(dim) if layer_id == 0 else None
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)
        self.att = SpectralVRWKV6SpatialMix(
            dim,
            num_heads,
            num_layers,
            layer_id,
            key_norm=key_norm,
            use_cuda_wkv=use_cuda_wkv,
        )
        self.ffn = SpectralVRWKVChannelMix(dim, num_layers, layer_id, key_norm=key_norm)
        self.gamma1 = nn.Parameter(init_values * torch.ones(dim))
        self.gamma2 = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.ln0 is not None:
            x = self.ln0(x)
        x = x + self.gamma1 * self.att(self.ln1(x))
        x = x + self.gamma2 * self.ffn(self.ln2(x))
        return x


class SpectralVRWKV6HSIEncoder(nn.Module):
    """HSI encoder using a pure PyTorch spectral VRWKV6 backbone."""

    def __init__(
        self,
        in_channels: int,
        feat_dim: int,
        num_layers: int = 2,
        num_heads: int = 2,
        key_norm: bool = True,
        use_cuda_wkv: bool = False,
    ) -> None:
        super().__init__()
        hidden = max(feat_dim, 64)
        if hidden % num_heads != 0:
            hidden = math.ceil(hidden / num_heads) * num_heads
        self.band_spatial = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, hidden, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.band_pos = nn.Parameter(torch.zeros(1, in_channels, hidden))
        self.blocks = nn.ModuleList(
            [
                SpectralVRWKV6Block(
                    hidden,
                    num_heads,
                    num_layers,
                    layer_id,
                    key_norm=key_norm,
                    use_cuda_wkv=use_cuda_wkv,
                )
                for layer_id in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(hidden)
        self.proj = nn.Sequential(
            nn.LayerNorm(hidden * 3),
            nn.Linear(hidden * 3, feat_dim),
        )

    def forward(self, x_h: torch.Tensor) -> torch.Tensor:
        assert x_h.ndim == 4, f"Expected HSI tensor [B, C_h, P, P], got {x_h.shape}"
        batch_size, channels, height, width = x_h.shape
        tokens = self.band_spatial(x_h.reshape(batch_size * channels, 1, height, width))
        tokens = tokens.flatten(1).reshape(batch_size, channels, -1)
        tokens = tokens + self.band_pos[:, :channels]
        for block in self.blocks:
            tokens = block(tokens)
        tokens = self.norm(tokens)
        pooled = torch.cat([tokens.mean(dim=1), tokens.amax(dim=1), tokens[:, -1]], dim=-1)
        return self.proj(pooled)


class LiDAREncoder(nn.Module):
    """Lightweight geometric-spatial encoder for LiDAR patches."""

    def __init__(self, in_channels: int, feat_dim: int) -> None:
        super().__init__()
        hidden = max(feat_dim // 2, 32)
        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, feat_dim),
        )

    def forward(self, x_l: torch.Tensor) -> torch.Tensor:
        assert x_l.ndim == 4, f"Expected LiDAR tensor [B, C_l, P, P], got {x_l.shape}"
        pooled = self.backbone(x_l)
        return self.proj(pooled)


class ConditionalMapper(nn.Module):
    """Class-conditional feature mapper G(z, e_k)."""

    def __init__(self, feat_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feat_dim * 2, feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, feat_dim),
        )

    def forward(self, z: torch.Tensor, e_k: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([z, e_k], dim=-1))


class ProbabilisticConditionalMapper(nn.Module):
    """Predicts a class-conditional Gaussian over LiDAR features."""

    def __init__(self, feat_dim: int) -> None:
        super().__init__()
        self.feat_dim = feat_dim
        self.net = nn.Sequential(
            nn.Linear(feat_dim * 2, feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, feat_dim * 2),
        )

    def forward(self, z: torch.Tensor, e_k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.net(torch.cat([z, e_k], dim=-1))
        mu, logvar = torch.split(out, self.feat_dim, dim=-1)
        return mu, logvar


class ClassConditionalCompatibilityEnergy(nn.Module):
    """Computes per-class energy terms for CC-SGCL and probabilistic S2G."""

    def __init__(
        self,
        num_classes: int,
        feat_dim: int,
        eta: float,
        method_variant: str = "cosgl",
        num_positive_prototypes: int = 3,
        num_negative_prototypes: int = 8,
        compat_mode: str = "bidirectional",
        compat_head_type: str = "deterministic",
        normalize_features: bool = True,
        energy_mode: str = "full",
        logvar_min: float = -5.0,
        logvar_max: float = 3.0,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.eta = eta
        self.method_variant = method_variant
        self.compat_mode = compat_mode
        self.compat_head_type = compat_head_type
        self.normalize_features = normalize_features
        self.energy_mode = energy_mode
        self.logvar_min = logvar_min
        self.logvar_max = logvar_max

        if self.compat_head_type not in {"deterministic", "probabilistic"}:
            raise ValueError(f"Unsupported compat_head_type: {self.compat_head_type}")
        if self.compat_head_type == "probabilistic" and self.compat_mode != "h2l_only":
            raise ValueError("Probabilistic compatibility currently supports compat_mode='h2l_only' only.")

        self.num_positive_prototypes = max(int(num_positive_prototypes), 1)
        self.use_multi_positive = variant_enabled(method_variant, "mp")
        if self.use_multi_positive:
            self.c_h = nn.Parameter(torch.randn(num_classes, self.num_positive_prototypes, feat_dim))
            self.c_l = nn.Parameter(torch.randn(num_classes, self.num_positive_prototypes, feat_dim))
        else:
            self.c_h = nn.Parameter(torch.randn(num_classes, feat_dim))
            self.c_l = nn.Parameter(torch.randn(num_classes, feat_dim))
        self.e_k = nn.Parameter(torch.randn(num_classes, feat_dim))
        self.negative_rel_prototypes = None
        if variant_enabled(method_variant, "np"):
            self.negative_rel_prototypes = nn.Parameter(torch.randn(num_negative_prototypes, feat_dim))
        if self.compat_head_type == "probabilistic":
            self.g_h2l = ProbabilisticConditionalMapper(feat_dim)
            self.g_l2h = None
        else:
            self.g_h2l = ConditionalMapper(feat_dim)
            self.g_l2h = ConditionalMapper(feat_dim)

    def _maybe_normalize(self, x: torch.Tensor) -> torch.Tensor:
        if not self.normalize_features:
            return x
        return F.normalize(x, dim=-1)

    def _gaussian_nll_diag(self, target: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        var = torch.exp(logvar)
        return 0.5 * (((target - mu) ** 2) / var + logvar).sum(dim=-1)

    def forward(self, z_h: torch.Tensor, z_l: torch.Tensor) -> Dict[str, torch.Tensor]:
        assert z_h.shape == z_l.shape, "HSI and LiDAR features must have the same shape"
        batch_size, feat_dim = z_h.shape
        assert feat_dim == self.feat_dim

        z_h = self._maybe_normalize(z_h)
        z_l = self._maybe_normalize(z_l)
        c_h = self._maybe_normalize(self.c_h)
        c_l = self._maybe_normalize(self.c_l)
        e_k = self._maybe_normalize(self.e_k)

        z_h_exp = z_h.unsqueeze(1).expand(batch_size, self.num_classes, feat_dim)
        z_l_exp = z_l.unsqueeze(1).expand(batch_size, self.num_classes, feat_dim)
        e_exp = e_k.unsqueeze(0).expand(batch_size, self.num_classes, feat_dim)
        if self.use_multi_positive:
            c_h_exp = c_h.unsqueeze(0).expand(batch_size, self.num_classes, self.num_positive_prototypes, feat_dim)
            c_l_exp = c_l.unsqueeze(0).expand(batch_size, self.num_classes, self.num_positive_prototypes, feat_dim)
            e_h_all = ((z_h_exp.unsqueeze(2) - c_h_exp) ** 2).sum(dim=-1)
            e_l_all = ((z_l_exp.unsqueeze(2) - c_l_exp) ** 2).sum(dim=-1)
            e_h = e_h_all.min(dim=2).values
            e_l = e_l_all.min(dim=2).values
        else:
            c_h_exp = c_h.unsqueeze(0).expand(batch_size, self.num_classes, feat_dim)
            c_l_exp = c_l.unsqueeze(0).expand(batch_size, self.num_classes, feat_dim)
            e_h = ((z_h_exp - c_h_exp) ** 2).sum(dim=-1)
            e_l = ((z_l_exp - c_l_exp) ** 2).sum(dim=-1)

        compat_outputs: Dict[str, torch.Tensor] = {}
        if self.compat_head_type == "probabilistic":
            mu_l, logvar_l = self.g_h2l(
                z_h_exp.reshape(-1, feat_dim), e_exp.reshape(-1, feat_dim)
            )
            mu_l = mu_l.reshape(batch_size, self.num_classes, feat_dim)
            logvar_l = torch.clamp(
                logvar_l.reshape(batch_size, self.num_classes, feat_dim),
                min=self.logvar_min,
                max=self.logvar_max,
            )
            mu_l = self._maybe_normalize(mu_l)
            e_h2l = self._gaussian_nll_diag(z_l_exp, mu_l, logvar_l)
            relation_residual = torch.abs(mu_l - z_l_exp)
            e_l2h = torch.zeros_like(e_h2l)
            e_compat = e_h2l
            compat_outputs = {
                'compat_mu_l': mu_l,
                'compat_logvar_l': logvar_l,
                'compat_logvar_mean': logvar_l.mean(dim=-1),
                'compat_logvar_min': logvar_l.min(dim=-1).values,
                'compat_logvar_max': logvar_l.max(dim=-1).values,
            }
        else:
            pred_l = self.g_h2l(
                z_h_exp.reshape(-1, feat_dim), e_exp.reshape(-1, feat_dim)
            ).reshape(batch_size, self.num_classes, feat_dim)
            pred_h = self.g_l2h(
                z_l_exp.reshape(-1, feat_dim), e_exp.reshape(-1, feat_dim)
            ).reshape(batch_size, self.num_classes, feat_dim)
            pred_l = self._maybe_normalize(pred_l)
            pred_h = self._maybe_normalize(pred_h)
            e_h2l = ((pred_l - z_l_exp) ** 2).sum(dim=-1)
            relation_residual = torch.abs(pred_l - z_l_exp)
            e_l2h = ((pred_h - z_h_exp) ** 2).sum(dim=-1)
            if self.compat_mode == "bidirectional":
                e_compat = e_h2l + e_l2h
            elif self.compat_mode == "h2l_only":
                e_compat = e_h2l
            else:
                raise ValueError(f"Unsupported compat_mode: {self.compat_mode}")

        if self.energy_mode == "full":
            energies = e_h + e_l + self.eta * e_compat
        elif self.energy_mode == "no_compat":
            energies = e_h + e_l
        elif self.energy_mode == "hsi_only":
            energies = e_h
        elif self.energy_mode == "lidar_only":
            energies = e_l
        else:
            raise ValueError(f"Unsupported energy_mode: {self.energy_mode}")

        negative_relation_score = torch.zeros(batch_size, device=z_h.device, dtype=z_h.dtype)
        negative_relation_distance = torch.zeros_like(negative_relation_score)
        if self.negative_rel_prototypes is not None:
            pred_idx = energies.argmin(dim=1)
            best_residual = relation_residual.gather(
                1,
                pred_idx.view(-1, 1, 1).expand(-1, 1, feat_dim),
            ).squeeze(1)
            best_residual = F.normalize(best_residual, dim=-1)
            neg_proto = F.normalize(self.negative_rel_prototypes, dim=-1)
            neg_dist = ((best_residual.unsqueeze(1) - neg_proto.unsqueeze(0)) ** 2).sum(dim=-1)
            negative_relation_distance = neg_dist.min(dim=1).values
            negative_relation_score = torch.exp(-negative_relation_distance)

        prototype_outputs = {
            "prototype_c_h": c_h,
            "prototype_c_l": c_l,
            "prototype_e_k": e_k,
            "relation_residual": relation_residual,
            "negative_relation_score": negative_relation_score,
            "negative_relation_distance": negative_relation_distance,
        }

        return {
            'energies': energies,
            'E_h': e_h,
            'E_l': e_l,
            'E_h2l': e_h2l,
            'E_l2h': e_l2h,
            'E_compat': e_compat,
            'E_total': energies,
            **compat_outputs,
            **prototype_outputs,
        }


class CCSGCLModel(nn.Module):
    """CC-SGCL model for class-conditional spectral-geometric compatibility."""

    def __init__(
        self,
        num_classes: int,
        hsi_channels: int,
        lidar_channels: int,
        feat_dim: int = 128,
        eta: float = 1.0,
        method_variant: str = "cosgl",
        num_positive_prototypes: int = 3,
        num_negative_prototypes: int = 8,
        np_score_weight: float = 1.0,
        margin_score_weight: float = 0.5,
        uf_score_weight: float = 1.0,
        compat_mode: str = "h2l_only",
        compat_head_type: str = "deterministic",
        normalize_features: bool = True,
        energy_mode: str = "full",
        hsi_encoder_type: str = "cnn",
        logvar_min: float = -5.0,
        logvar_max: float = 3.0,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.method_variant = method_variant
        self.np_score_weight = np_score_weight
        self.margin_score_weight = margin_score_weight
        self.uf_score_weight = uf_score_weight
        self.hsi_encoder_type = hsi_encoder_type
        if hsi_encoder_type == "cnn":
            self.hsi_encoder = HSIEncoder(hsi_channels, feat_dim)
        elif hsi_encoder_type == "spectral_rwkv":
            self.hsi_encoder = SpectralRWKVHSIEncoder(hsi_channels, feat_dim)
        elif hsi_encoder_type == "spectral_vrwkv":
            self.hsi_encoder = SpectralVRWKVHSIEncoder(hsi_channels, feat_dim)
        elif hsi_encoder_type == "spectral_vrwkv_official":
            self.hsi_encoder = SpectralVRWKVHSIEncoder(
                hsi_channels,
                feat_dim,
                num_layers=2,
                official_biwkv=True,
                channel_key_norm=True,
            )
        elif hsi_encoder_type == "spectral_vrwkv6":
            self.hsi_encoder = SpectralVRWKV6HSIEncoder(
                hsi_channels,
                feat_dim,
                num_layers=2,
                num_heads=2,
                key_norm=True,
            )
        elif hsi_encoder_type == "spectral_vrwkv6_cuda":
            self.hsi_encoder = SpectralVRWKV6HSIEncoder(
                hsi_channels,
                feat_dim,
                num_layers=2,
                num_heads=2,
                key_norm=True,
                use_cuda_wkv=True,
            )
        else:
            raise ValueError(f"Unsupported hsi_encoder_type: {hsi_encoder_type}")
        self.lidar_encoder = LiDAREncoder(lidar_channels, feat_dim)
        self.energy_head = ClassConditionalCompatibilityEnergy(
            num_classes=num_classes,
            feat_dim=feat_dim,
            eta=eta,
            method_variant=method_variant,
            num_positive_prototypes=num_positive_prototypes,
            num_negative_prototypes=num_negative_prototypes,
            compat_mode=compat_mode,
            compat_head_type=compat_head_type,
            normalize_features=normalize_features,
            energy_mode=energy_mode,
            logvar_min=logvar_min,
            logvar_max=logvar_max,
        )
        self.uf_head = None
        if variant_enabled(method_variant, "uf"):
            self.uf_head = nn.Sequential(
                nn.Linear(feat_dim * 3, feat_dim),
                nn.ReLU(inplace=True),
                nn.Linear(feat_dim, num_classes),
            )
        self.aux_classifier = nn.Linear(feat_dim * 3, num_classes, bias=False)

    def _prototype_similarity_score(
        self,
        z_h: torch.Tensor,
        z_l: torch.Tensor,
        energy_dict: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        z_h_norm = F.normalize(z_h, dim=-1)
        z_l_norm = F.normalize(z_l, dim=-1)
        c_h = energy_dict["prototype_c_h"]
        c_l = energy_dict["prototype_c_l"]

        if c_h.ndim == 3:
            sim_h = torch.einsum("bd,kpd->bkp", z_h_norm, c_h).max(dim=2).values
            sim_l = torch.einsum("bd,kpd->bkp", z_l_norm, c_l).max(dim=2).values
        else:
            sim_h = torch.einsum("bd,kd->bk", z_h_norm, c_h)
            sim_l = torch.einsum("bd,kd->bk", z_l_norm, c_l)

        class_similarity = 0.5 * (sim_h + sim_l)
        sorted_similarity = torch.sort(class_similarity, dim=1, descending=True).values
        top_similarity = sorted_similarity[:, 0]
        if class_similarity.shape[1] > 1:
            similarity_gap = sorted_similarity[:, 0] - sorted_similarity[:, 1]
        else:
            similarity_gap = torch.zeros_like(top_similarity)

        # High for samples that are far from all class directions or have ambiguous top classes.
        nc_unknown_score = (1.0 - top_similarity) + torch.exp(-torch.clamp(similarity_gap, min=0.0))
        feature_norm = 0.5 * (z_h.norm(dim=-1) + z_l.norm(dim=-1))
        return {
            "prototype_similarity": class_similarity,
            "prototype_similarity_gap": similarity_gap,
            "prototype_top_similarity": top_similarity,
            "feature_norm": feature_norm,
            "nc_unknown_score": nc_unknown_score,
        }

    def forward(self, x_h: torch.Tensor, x_l: torch.Tensor) -> Dict[str, torch.Tensor]:
        assert x_h.ndim == 4 and x_l.ndim == 4
        assert x_h.shape[0] == x_l.shape[0], "Batch sizes must match"
        assert x_h.shape[-2:] == x_l.shape[-2:], "Patch shapes must match"

        z_h = self.hsi_encoder(x_h)
        z_l = self.lidar_encoder(x_l)
        energy_dict = self.energy_head(z_h, z_l)
        nc_dict = self._prototype_similarity_score(z_h, z_l, energy_dict)
        energies = energy_dict['energies']
        logits = -energies
        sorted_energy = torch.sort(energies, dim=1).values
        min_energy = sorted_energy[:, 0]
        if energies.shape[1] > 1:
            energy_gap = sorted_energy[:, 1] - sorted_energy[:, 0]
        else:
            energy_gap = torch.zeros_like(min_energy)
        margin_uncertainty = torch.exp(-torch.clamp(energy_gap, min=0.0))
        open_score = min_energy

        if variant_enabled(self.method_variant, "np"):
            open_score = (
                open_score
                + self.np_score_weight * energy_dict["negative_relation_score"]
                + self.margin_score_weight * margin_uncertainty
            )

        uf_logits = None
        uf_probs = None
        uf_unknown_score = torch.zeros_like(open_score)
        z_h_norm = F.normalize(z_h, dim=-1)
        z_l_norm = F.normalize(z_l, dim=-1)
        fused = torch.cat([z_h_norm, z_l_norm, torch.abs(z_h_norm - z_l_norm)], dim=-1)
        aux_cosine = F.linear(
            F.normalize(fused, dim=-1),
            F.normalize(self.aux_classifier.weight, dim=-1),
        )
        if self.uf_head is not None:
            uf_logits = self.uf_head(fused)
            uf_probs = torch.sigmoid(uf_logits)
            uf_unknown_score = 1.0 - uf_probs.max(dim=1).values
            open_score = open_score + self.uf_score_weight * uf_unknown_score

        return {
            'z_h': z_h,
            'z_l': z_l,
            'energies': energies,
            'logits': logits,
            'energy_gap': energy_gap,
            'margin_uncertainty': margin_uncertainty,
            'open_score': open_score,
            'uf_logits': uf_logits,
            'uf_probs': uf_probs,
            'uf_unknown_score': uf_unknown_score,
            'aux_cosine': aux_cosine,
            **nc_dict,
            **energy_dict,
        }
