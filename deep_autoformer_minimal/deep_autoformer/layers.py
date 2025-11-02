
import torch
import torch.nn as nn
import torch.fft as fft

class MovingAverage(nn.Module):
    """
    Moving average for trend extraction.
    Uses 1D conv with fixed weights.
    """
    def __init__(self, kernel_size: int):
        super().__init__()
        self.kernel_size = kernel_size
        padding = (kernel_size - 1) // 2
        self.pad = nn.ConstantPad1d((padding, padding), 0.0)

    def forward(self, x):
        # x: [B, C, L]
        B, C, L = x.shape
        # Create weight for depthwise convolution: [C, 1, kernel_size]
        weight = torch.ones(C, 1, self.kernel_size, device=x.device, dtype=x.dtype) / self.kernel_size
        x = self.pad(x)
        return torch.conv1d(x, weight, groups=C)  # [B, C, L]


class SeriesDecomposition(nn.Module):
    """
    Decompose series into trend + seasonal (residual) using MA filter.
    """
    def __init__(self, kernel_size: int = 25):
        super().__init__()
        self.ma = MovingAverage(kernel_size)

    def forward(self, x):
        trend = self.ma(x)
        seasonal = x - trend
        return seasonal, trend


class PositionalEncoding(nn.Module):
    """
    Standard sinusoidal positional encoding.
    """
    def __init__(self, d_model: int, max_len: int = 10000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, L, D]
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x: [B, L, D]
        L = x.size(1)
        return x + self.pe[:, :L, :]


class AutoCorrelation(nn.Module):
    """
    A simplified Auto-Correlation mechanism inspired by Autoformer.
    It aggregates values across top-k seasonal lags determined by
    frequency-domain correlation between Q and K.

    NOTE: This is a pedagogical, compact implementation –
    not an exact replica of the original optimized kernel.
    """
    def __init__(self, d_model: int, n_heads: int, top_k: int = 4, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.top_k = top_k

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def _topk_lags(self, q, k):
        """
        Estimate top-k lags by maximizing cross-correlation via FFT.
        q, k: [B, H, L, Dh]
        Returns: indices of top-k positive lags, shape [B, H, K]
        """
        B, H, L, Dh = q.shape
        # mean center along time
        q0 = q - q.mean(dim=2, keepdim=True)
        k0 = k - k.mean(dim=2, keepdim=True)

        # FFT along time axis
        Qf = fft.rfft(q0, dim=2)
        Kf = fft.rfft(k0, dim=2)
        # cross-correlation via conj product
        corr_f = Qf * torch.conj(Kf)
        corr = fft.irfft(corr_f, n=L, dim=2).real  # [B, H, L, Dh]
        # aggregate over feature dimension Dh
        corr_sum = corr.mean(dim=3)  # [B, H, L]
        # consider only positive lags (1..L-1)
        corr_pos = corr_sum[:, :, 1:]
        vals, idx = torch.topk(corr_pos, k=min(self.top_k, corr_pos.size(-1)), dim=-1)
        # convert to lag indices (1..L-1)
        lags = idx + 1
        return lags  # [B, H, K]

    def _shift_gather(self, v, lags):
        """
        For each head and batch, gather V shifted by each lag and average.
        v: [B, H, L, Dh]
        lags: [B, H, K]
        returns: [B, H, L, Dh]
        """
        B, H, L, Dh = v.shape
        K = lags.size(-1)
        out = 0.0
        for i in range(K):
            lag = lags[..., i]  # [B, H]
            # build shifted indices per batch/head
            idx = torch.arange(L, device=v.device).view(1, 1, L).repeat(B, H, 1) - lag.unsqueeze(-1)
            idx = idx.clamp(min=0)  # pad front
            gathered = torch.gather(v, dim=2, index=idx.unsqueeze(-1).repeat(1, 1, 1, Dh))
            out = out + gathered
        out = out / K
        return out

    def forward(self, x_q, x_kv):
        """
        x_q: [B, Lq, D], x_kv: [B, Lk, D]
        """
        B, Lq, D = x_q.shape
        Lk = x_kv.size(1)

        q = self.q_proj(x_q).view(B, Lq, self.n_heads, self.d_head).permute(0, 2, 1, 3)  # [B,H,Lq,Dh]
        k = self.k_proj(x_kv).view(B, Lk, self.n_heads, self.d_head).permute(0, 2, 1, 3)
        v = self.v_proj(x_kv).view(B, Lk, self.n_heads, self.d_head).permute(0, 2, 1, 3)

        # pad or trim k,v to Lq for lagging convenience
        if Lk != Lq:
            if Lk < Lq:
                pad_len = Lq - Lk
                pad = torch.zeros(B, self.n_heads, pad_len, self.d_head, device=x_q.device, dtype=x_q.dtype)
                k = torch.cat([pad, k], dim=2)
                v = torch.cat([pad, v], dim=2)
            else:
                k = k[:, :, -Lq:, :]
                v = v[:, :, -Lq:, :]

        lags = self._topk_lags(q, k)  # [B,H,K]
        agg = self._shift_gather(v, lags)  # [B,H,Lq,Dh]

        out = agg.permute(0, 2, 1, 3).contiguous().view(B, Lq, D)
        out = self.o_proj(out)
        return self.dropout(out)


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, top_k: int, dropout: float, kernel_size: int):
        super().__init__()
        self.decomp = SeriesDecomposition(kernel_size)
        self.attn = AutoCorrelation(d_model, n_heads, top_k=top_k, dropout=dropout)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        # x: [B, L, D]
        seasonal, trend = self.decomp(x.transpose(1, 2))  # to [B,C,L]
        seasonal = seasonal.transpose(1, 2)
        trend = trend.transpose(1, 2)

        # auto-correlation block on seasonal part
        y = self.attn(self.norm1(seasonal), seasonal) + seasonal
        y = self.ff(self.norm2(y)) + y

        # recompose with trend (skip connection keeps trend)
        out = y + trend
        return out


class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, top_k: int, dropout: float, kernel_size: int):
        super().__init__()
        self.decomp = SeriesDecomposition(kernel_size)
        self.self_attn = AutoCorrelation(d_model, n_heads, top_k=top_k, dropout=dropout)
        self.cross_attn = AutoCorrelation(d_model, n_heads, top_k=top_k, dropout=dropout)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, x, memory):
        # x: [B, Ld, D]; memory: [B, Le, D]
        seasonal, trend = self.decomp(x.transpose(1, 2))
        seasonal = seasonal.transpose(1, 2)
        trend = trend.transpose(1, 2)

        y = self.self_attn(self.norm1(seasonal), seasonal) + seasonal
        y = self.cross_attn(self.norm2(y), memory) + y
        y = self.ff(self.norm3(y)) + y

        out = y + trend
        return out
