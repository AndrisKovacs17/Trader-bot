"""KLA-Mamba model definition (PyTorch)."""

from __future__ import annotations

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:  # pragma: no cover - optional dependency
    torch = None
    nn = None
    F = None


if nn is not None:
    class KLAMambaBlock(nn.Module):
        def __init__(self, d_model: int, heads: int, d_state: int, kernel_size: int = 4):
            super().__init__()
            self.H, self.D, self.E = heads, d_state, heads * d_state

            self.in_proj = nn.Linear(d_model, self.E * 2)
            self.conv1d = nn.Conv1d(
                self.E,
                self.E,
                kernel_size,
                groups=self.E,
                padding=kernel_size - 1,
            )
            self.out_proj = nn.Linear(self.E, d_model)
            self.res_proj = nn.Linear(d_model, d_model)
            self.proj_v = nn.Linear(self.E, self.E)

            self.Q_net = nn.Sequential(nn.Linear(self.E, 64), nn.SiLU(), nn.Linear(64, self.H))
            self.R_net = nn.Sequential(nn.Linear(self.E, 64), nn.SiLU(), nn.Linear(64, self.H))
            self.K_net = nn.Sequential(nn.Linear(self.E, 64), nn.SiLU(), nn.Linear(64, self.H))
            # R >> Q at init: softplus(-2.0)≈0.127, softplus(1.5)≈1.732
            # K_base = 0.127/1.859 ≈ 0.068 → A≈0.932 → memória ~14 lépés.
            # A modell a tanítás során a feladatnak megfelelő memóriára konvergál.
            self.q_scale = nn.Parameter(torch.tensor(-2.0))
            self.r_scale = nn.Parameter(torch.tensor( 1.5))

            self.mu_init = nn.Parameter(torch.zeros(1, self.H, self.D))
            self.res_gate_bias = nn.Parameter(torch.tensor(-1.0))
            self.v_norm = nn.LayerNorm(self.E)

        def parallel_scan(self, A: torch.Tensor, B_scan: torch.Tensor, mu_init: torch.Tensor) -> torch.Tensor:
            T, step = A.shape[1], 1
            while step < T:
                A_right   = A[:, step:]          # original A[t] BEFORE updating (critical!)
                A_shifted = A[:, :-step]         # A[t-step]
                B_shifted = B_scan[:, :-step]    # B[t-step]
                # A_new[t] = A[t] * A[t-step]
                A = torch.cat([A[:, :step], A_right * A_shifted], dim=1)
                # B_new[t] = B[t] + A_original[t] * B[t-step]  — must use A_right, NOT new A
                # (using new A would double-decay all past contributions)
                B_scan = torch.cat([B_scan[:, :step], B_scan[:, step:] + A_right * B_shifted], dim=1)
                step *= 2
            return B_scan + A * mu_init.unsqueeze(1)

        def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, float]]:
            batch_size, seq_len, _ = x.shape
            dtype = x.dtype

            x_cw, x_gate = self.in_proj(x).chunk(2, dim=-1)
            gate = F.silu(x_gate)

            x_core = self.conv1d(x_cw.transpose(1, 2))[:, :, :seq_len].transpose(1, 2)
            x_core = F.silu(x_core)

            v_seq = self.v_norm(self.proj_v(x_core)).view(batch_size, seq_len, self.H, self.D)

            Q = (F.softplus(self.Q_net(x_core)) * F.softplus(self.q_scale)).unsqueeze(-1)
            R = (F.softplus(self.R_net(x_core)) * F.softplus(self.r_scale)).unsqueeze(-1)

            K_base = Q / (Q + R + 1e-6)
            # Content-dependent K correction centered at 0 at init:
            # sigmoid(K_net≈0) = 0.5  →  K_delta = 0.3*(0.5-0.5) = 0  →  K_seq = K_base ≈ 0.07
            # Previously: 0.5*(K_net+1.0) init=0.5  →  K_seq = 0.7*0.07+0.3*0.5 = 0.199 (3× K_base!)
            # Also fixes: old formula could go negative when K_net<-1, killing gradients at clamp.
            K_delta = 0.3 * (torch.sigmoid(self.K_net(x_core).unsqueeze(-1).to(dtype)) - 0.5)
            K_seq = torch.clamp(K_base + K_delta, 1e-4, 0.999)

            # A = 1 - K: high gain → fast update (forgets past), low gain → slow update (long memory).
            # Old formula (1 - 0.5*K) with clamp 0.5 blocked fast updates AND capped memory depth.
            A_scan = torch.clamp(1.0 - K_seq, 0.01, 0.99)
            B_scan = K_seq * v_seq

            mu_all = self.parallel_scan(A_scan, B_scan, self.mu_init).reshape(batch_size, seq_len, self.E)
            y_p = self.out_proj(mu_all * gate)

            res_g = torch.sigmoid(self.res_gate_bias)
            out = y_p + res_g * (self.res_proj(x) - y_p)

            debug_stats = {
                "K_mean": K_seq.mean().item(),
                "K_std": K_seq.std().item(),
                "K_min": K_seq.min().item(),
                "K_max": K_seq.max().item(),
                "A_mean": A_scan.mean().item(),
                "R_mean": R.mean().item(),
                "res_gate_mean": res_g.item(),
            }
            return out, Q, R, debug_stats

else:
    class KLAMambaBlock:  # pragma: no cover - optional dependency fallback
        def __init__(self, *args, **kwargs):
            raise ImportError("KLAMambaBlock requires PyTorch. Install torch to use this module.")


if nn is not None:
    class KLAMambaStack(nn.Module):
        """Stacked KLA-Mamba encoder with multi-timescale cross-attention.

        Architecture:
          input_proj  : Linear(feature_dim → hidden_dim)
          input_norm  : LayerNorm(hidden_dim) — applied right after input_proj so all
                        blocks receive unit-variance inputs regardless of feature scale
          block_norms : N-1 intermediate LayerNorms between KLA blocks (pre-norm style)
          blocks      : N × KLAMambaBlock(hidden_dim)   [like real Mamba: 3-24 layers]
          norm        : LayerNorm(hidden_dim) — final output norm
          cross_attn  : last-token queries slow (every slow_stride-th token) context
                        → simulates a coarser timeframe (e.g. 5m bars, stride=12 ≈ 1h)
          output      : full sequence [batch, seq, hidden_dim]  (caller takes [:, -1, :])
        """

        def __init__(
            self,
            feature_dim: int,
            hidden_dim: int,
            num_layers: int,
            heads: int,
            d_state: int,
            slow_stride: int = 12,
        ) -> None:
            super().__init__()
            self.slow_stride = max(1, slow_stride)
            self.input_proj = nn.Linear(feature_dim, hidden_dim)
            # Normalise immediately after projection so blocks receive zero-mean,
            # unit-variance input regardless of how diverse/scaled the raw features are.
            self.input_norm = nn.LayerNorm(hidden_dim)
            self.blocks = nn.ModuleList(
                [KLAMambaBlock(d_model=hidden_dim, heads=heads, d_state=d_state) for _ in range(num_layers)]
            )
            # Inter-block LayerNorms (pre-norm between consecutive blocks).
            # Prevents activation scale drift across layers without adding skip paths.
            self.block_norms = nn.ModuleList(
                [nn.LayerNorm(hidden_dim) for _ in range(max(0, num_layers - 1))]
            )
            self.norm = nn.LayerNorm(hidden_dim)
            # num_heads for cross-attention: must divide hidden_dim evenly; cap at heads
            _ca_heads = min(heads, max(1, hidden_dim // 16))
            while hidden_dim % _ca_heads != 0 and _ca_heads > 1:
                _ca_heads -= 1
            self.cross_attn = nn.MultiheadAttention(
                embed_dim=hidden_dim, num_heads=_ca_heads, batch_first=True, dropout=0.0
            )

        def forward(
            self, x: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, float]]:
            # x: [batch, seq, feature_dim]
            h = self.input_norm(self.input_proj(x))  # project then normalise

            last_Q: torch.Tensor = torch.zeros(1, device=x.device)
            last_R: torch.Tensor = torch.zeros(1, device=x.device)
            last_debug: dict[str, float] = {}
            for i, block in enumerate(self.blocks):
                h, last_Q, last_R, last_debug = block(h)
                # Apply inter-block norm before feeding into the next block.
                if i < len(self.block_norms):
                    h = self.block_norms[i](h)

            h = self.norm(h)  # final norm before cross-attention

            # Cross-attention: last token (fast/current) attends to slow context
            slow = h[:, :: self.slow_stride, :]  # [batch, seq//stride, hidden_dim]
            if slow.shape[1] == 0:
                slow = h[:, :1, :]
            query = h[:, -1:, :]  # [batch, 1, hidden_dim]
            attn_out, _ = self.cross_attn(query, slow, slow)  # [batch, 1, hidden_dim]

            # Residual-inject cross-attention at last position only (gradient-safe)
            h_out = torch.cat([h[:, :-1, :], h[:, -1:, :] + attn_out], dim=1)
            return h_out, last_Q, last_R, last_debug

else:
    class KLAMambaStack:  # pragma: no cover
        def __init__(self, *args, **kwargs):
            raise ImportError("KLAMambaStack requires PyTorch.")


__all__ = ["KLAMambaBlock", "KLAMambaStack"]
