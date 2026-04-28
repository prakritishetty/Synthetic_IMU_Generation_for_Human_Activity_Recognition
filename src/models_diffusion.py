import math
import torch
import torch.nn as nn


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class AdaLayerNorm(nn.Module):
    """
    DiT-style Adaptive LayerNorm.
    Predicts scale and shift from a conditioning vector, which makes class
    and timestep information much more effective than simple addition.
    Reference: Peebles & Xie, "Scalable Diffusion Models with Transformers" (DiT).
    """
    def __init__(self, d_model, cond_dim):
        super().__init__()
        # elementwise_affine=False: we supply our own scale/shift from the condition
        self.norm = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        # Projects conditioning → (scale, shift) per feature
        self.proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, d_model * 2)
        )
        # Zero-init so the block starts as an identity transform
        nn.init.zeros_(self.proj[-1].weight)
        nn.init.zeros_(self.proj[-1].bias)

    def forward(self, x, cond):
        # cond: (B, cond_dim) → scale/shift: (B, 1, d_model)
        scale_shift = self.proj(cond).unsqueeze(1)
        scale, shift = scale_shift.chunk(2, dim=-1)
        return (1.0 + scale) * self.norm(x) + shift


class DiTBlock(nn.Module):
    """
    Transformer block using AdaLayerNorm for conditioning (DiT-style).
    The conditioning signal (time + class) modulates both the attention
    pre-norm and the FFN pre-norm independently.
    """
    def __init__(self, d_model, nhead, cond_dim, dim_feedforward=None, dropout=0.1):
        super().__init__()
        dim_feedforward = dim_feedforward or d_model * 4
        self.norm1 = AdaLayerNorm(d_model, cond_dim)
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm2 = AdaLayerNorm(d_model, cond_dim)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x, cond, key_padding_mask=None):
        # Self-attention with adaptive pre-norm
        normed = self.norm1(x, cond)
        attn_out, _ = self.attn(normed, normed, normed, key_padding_mask=key_padding_mask)
        x = x + self.drop(attn_out)
        # Feed-forward with adaptive pre-norm
        x = x + self.drop(self.ff(self.norm2(x, cond)))
        return x


class TokenTransformerDiffusion(nn.Module):
    """
    Transformer-based DDPM noise predictor with DiT-style AdaLayerNorm conditioning.
    Increased to 6 layers and wider condition MLP compared to previous version.
    """
    def __init__(self, seq_len=192, token_dim=64, num_classes=6,
                 d_model=128, nhead=4, num_layers=6):
        super().__init__()
        self.seq_len = seq_len
        self.token_dim = token_dim
        cond_dim = d_model * 2  # richer conditioning space

        # Timestep embedding: sinusoidal → wide MLP
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(d_model),
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, cond_dim),
        )
        # Class embedding
        self.class_emb = nn.Embedding(num_classes, cond_dim)

        # Input projection + learned positional embedding
        self.input_proj = nn.Linear(token_dim, d_model)
        self.pos_emb = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)

        # DiT blocks
        self.blocks = nn.ModuleList([
            DiTBlock(d_model, nhead, cond_dim, dim_feedforward=d_model * 4, dropout=0.1)
            for _ in range(num_layers)
        ])

        self.final_norm = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, token_dim)

        # Zero-init output projection (stabilises early training)
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(self, x, t, c, mask=None):
        B, L, _ = x.shape
        t_emb = self.time_mlp(t)           # (B, cond_dim)
        c_emb = self.class_emb(c)          # (B, cond_dim)
        cond = t_emb + c_emb               # (B, cond_dim)

        x = self.input_proj(x)
        x = x + self.pos_emb[:, :L, :]

        # Invert valid boolean mask for PyTorch MHA padding compatibility
        key_padding_mask = ~mask if mask is not None else None

        for block in self.blocks:
            x = block(x, cond, key_padding_mask=key_padding_mask)

        x = self.final_norm(x)
        return self.output_proj(x)


class IMUDecoder(nn.Module):
    """
    MLP decoder: 64-d token → normalised movement-element patch.
    Sigmoid output bounds predictions to (0, 1), matching the
    min-max normalised ME patches produced by the BioPM preprocessor.
    """
    def __init__(self, token_dim=64, patch_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(token_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, patch_dim),
            nn.Sigmoid(),  # bound output to (0, 1) — matches normalised ME patch targets
        )

    def forward(self, tokens):
        return self.net(tokens)

class WaveformDecoder(nn.Module):
    """
    Decodes the (N, L, 64) BioPM tokens directly back into the 
    raw physical (N, T, 3) acceleration windows.
    Since L=192 and T=300 (for 10s @ 30Hz), we use a 1D CNN with 
    temporal interpolation.
    """
    def __init__(self, token_dim=64, hidden_dim=128, out_channels=3, target_length=300):
        super().__init__()
        self.target_length = target_length
        
        self.conv1 = nn.Sequential(
            nn.Conv1d(token_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.GELU()
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.GELU()
        )
        
        self.out_conv = nn.Conv1d(hidden_dim, out_channels, kernel_size=3, padding=1)

    def forward(self, tokens):
        # tokens: (B, L, 64)
        x = tokens.transpose(1, 2)  # (B, 64, L)
        
        x = self.conv1(x)
        
        # Interpolate sequence length from L (192) to target_length (300)
        x = torch.nn.functional.interpolate(x, size=self.target_length, mode='linear', align_corners=False)
        
        x = self.conv2(x)
        x = self.out_conv(x)  # (B, 3, target_length)
        
        return x.transpose(1, 2)  # (B, target_length, 3)

