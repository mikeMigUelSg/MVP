
import torch
import torch.nn as nn
from .layers import PositionalEncoding, EncoderLayer, DecoderLayer

class DeepAutoformer(nn.Module):
    """
    Minimal Deep-Autoformer for sequence-to-sequence forecasting.
    Adds extra MLP ("deep") blocks around Autoformer-style encoder/decoder.
    """
    def __init__(
        self,
        enc_in: int,
        dec_in: int,
        d_model: int = 256,
        n_heads: int = 4,
        e_layers: int = 2,
        d_layers: int = 1,
        d_ff: int = 512,
        top_k: int = 4,
        kernel_size: int = 25,
        dropout: float = 0.1,
        pred_len: int = 24,
        label_len: int = 24,
        output_dim: int = 1,
        add_deep_mlp: bool = True,
    ):
        super().__init__()
        self.pred_len = pred_len
        self.label_len = label_len
        self.enc_embed = nn.Linear(enc_in, d_model)
        self.dec_embed = nn.Linear(dec_in, d_model)
        self.pos = PositionalEncoding(d_model)

        self.encoder = nn.ModuleList(
            [EncoderLayer(d_model, n_heads, d_ff, top_k, dropout, kernel_size) for _ in range(e_layers)]
        )
        self.decoder = nn.ModuleList(
            [DecoderLayer(d_model, n_heads, d_ff, top_k, dropout, kernel_size) for _ in range(d_layers)]
        )

        # Deep MLP blocks (this is the "deep" addition vs. vanilla Autoformer)
        self.add_deep_mlp = add_deep_mlp
        if add_deep_mlp:
            self.deep_enc = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, d_ff),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_ff, d_model),
            )
            self.deep_dec = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, d_ff),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_ff, d_model),
            )

        self.proj = nn.Linear(d_model, output_dim)

    def forward(self, enc_x, dec_x):
        """
        enc_x: [B, L_enc, enc_in]
        dec_x: [B, L_label + L_pred, dec_in] (first L_label are known previous targets)
        """
        en = self.pos(self.enc_embed(enc_x))   # [B, Le, D]
        de = self.pos(self.dec_embed(dec_x))   # [B, Ld, D]

        for layer in self.encoder:
            en = layer(en)
        if self.add_deep_mlp:
            en = en + self.deep_enc(en)

        mem = en
        for layer in self.decoder:
            de = layer(de, mem)
        if self.add_deep_mlp:
            de = de + self.deep_dec(de)

        out = self.proj(de)                   # [B, Ld, 1]
        # return only the last pred_len steps
        return out[:, -self.pred_len:, :]
