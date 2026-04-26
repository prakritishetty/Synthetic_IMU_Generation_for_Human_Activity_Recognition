import torch
from tqdm import tqdm

def sample_cfg_diffusion(model, shape, classes, device, null_class_id, cfg_scale=3.0, timesteps=1000):
    model.eval()
    alphas_cumprod = (1.0 - torch.linspace(1e-4, 0.02, timesteps)).cumprod(0).to(device)
    
    b, L, d = shape
    x = torch.randn(shape, device=device)
    null_classes = torch.full_like(classes, null_class_id)
    
    with torch.no_grad():
        for i in tqdm(reversed(range(timesteps)), desc="Evaluating CFG Matrix Denoising"):
            t_tensor = torch.full((b,), i, device=device, dtype=torch.long)
            
            ### CFG MATHEMATICS ###
            # 1. Ask the model for the specific Activity Noise
            cond_noise = model(x, t_tensor, classes, mask=torch.ones((b, L), device=device, dtype=torch.bool))
            # 2. Ask the model for the generic, unconditioned Void Noise
            uncond_noise = model(x, t_tensor, null_classes, mask=torch.ones((b, L), device=device, dtype=torch.bool))
            
            # 3. Algebra Extrapolation (The "Wow Factor")
            pred_noise = uncond_noise + cfg_scale * (cond_noise - uncond_noise)
            #######################
            
            alpha_t = alphas_cumprod[i]
            alpha_t_prev = alphas_cumprod[i-1] if i > 0 else torch.tensor(1.0, device=device)
            beta_t = 1 - (alpha_t / alpha_t_prev)
            
            x_mean = (1 / torch.sqrt(1 - beta_t)) * (x - (beta_t / torch.sqrt(1 - alpha_t)) * pred_noise)
            x = x_mean + (torch.sqrt(beta_t) * torch.randn_like(x) if i > 0 else 0)
    return x
