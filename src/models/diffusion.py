import torch
import torch.nn as nn
import math

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class ConditionalUNet(nn.Module):
    """A minimal conditional UNet for diffusion."""
    def __init__(self, in_ch=6, out_ch=3, time_emb_dim=256):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.GELU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )
        self.conv1 = nn.Conv2d(in_ch, 64, 3, padding=1)
        self.out = nn.Conv2d(64, out_ch, 1)
        
    def forward(self, x, time):
        t = self.time_mlp(time)
        x = self.conv1(x)
        return self.out(x)

class DDPM(nn.Module):
    def __init__(self, unet, timesteps=1000):
        super().__init__()
        self.unet = unet
        self.timesteps = timesteps
        betas = torch.linspace(1e-4, 0.02, timesteps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        
    def q_sample(self, x_start, t, noise=None):
        if noise is None: noise = torch.randn_like(x_start)
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
        
    def p_losses(self, x_start, t, noise=None, condition=None):
        if noise is None: noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start, t, noise=noise)
        x_input = torch.cat([x_noisy, condition], dim=1) if condition is not None else x_noisy
        predicted_noise = self.unet(x_input, t)
        loss = nn.functional.mse_loss(predicted_noise, noise)
        return loss
