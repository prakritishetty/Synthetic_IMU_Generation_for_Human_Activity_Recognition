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

class TokenTransformerDiffusion(nn.Module):
    def __init__(self, seq_len=76, token_dim=64, num_classes=6, d_model=128, nhead=4, num_layers=4):
        super().__init__()
        self.seq_len = seq_len
        self.token_dim = token_dim
        
        # Maps diffusion time-step [0, 1000] to a dense embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(d_model),
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model)
        )
        # Condition on Activity Class ('Walking', 'Stairs', etc.)
        self.class_emb = nn.Embedding(num_classes, d_model)
        self.input_proj = nn.Linear(token_dim, d_model)
        
        # Learned position embedding for the continuous sequence
        self.pos_emb = nn.Parameter(torch.randn(1, seq_len, d_model))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model*4, 
            dropout=0.1, activation="gelu", batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(d_model, token_dim)
        
    def forward(self, x, t, c, mask=None):
        B, L, _ = x.shape
        t_emb = self.time_mlp(t)          
        c_emb = self.class_emb(c)         
        
        x = self.input_proj(x)            
        x = x + self.pos_emb[:, :L, :]    
        
        # Inject Time and Class into sequence uniformly
        cond_emb = (t_emb + c_emb).unsqueeze(1) 
        x = x + cond_emb
        
        # Invert valid boolean mask for PyTorch Transformer padding compatibility
        key_padding_mask = ~mask if mask is not None else None
        
        x = self.transformer(x, src_key_padding_mask=key_padding_mask)
        return self.output_proj(x)

class IMUDecoder(nn.Module):
    """ Lightweight MLP to decode abstract 64-d tokens back to physical Raw Patches """
    def __init__(self, token_dim=64, patch_dim=32*3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(token_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, patch_dim)
        )
    def forward(self, tokens):
        return self.net(tokens)
