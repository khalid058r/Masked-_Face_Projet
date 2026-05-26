import torch
import torch.nn.functional as F

@torch.no_grad()
def psnr(pred, target, max_val=2.0):
    mse = F.mse_loss(pred, target)
    if mse < 1e-10:
        return torch.tensor(100.0, device=pred.device)
    return 10 * torch.log10(max_val**2 / mse)

def _gaussian_window(size=11, sigma=1.5):
    coords = torch.arange(size, dtype=torch.float32) - size // 2
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g = g / g.sum()
    return (g.unsqueeze(0) * g.unsqueeze(1))

@torch.no_grad()
def ssim(pred, target, window_size=11, max_val=2.0):
    window = _gaussian_window(window_size).to(pred.device)
    window = window.unsqueeze(0).unsqueeze(0).expand(pred.size(1), 1, -1, -1)
    pad = window_size // 2
    mu1 = F.conv2d(pred, window, padding=pad, groups=pred.size(1))
    mu2 = F.conv2d(target, window, padding=pad, groups=target.size(1))
    mu1_sq, mu2_sq, mu1_mu2 = mu1*mu1, mu2*mu2, mu1*mu2
    s1 = F.conv2d(pred*pred, window, padding=pad, groups=pred.size(1)) - mu1_sq
    s2 = F.conv2d(target*target, window, padding=pad, groups=target.size(1)) - mu2_sq
    s12 = F.conv2d(pred*target, window, padding=pad, groups=pred.size(1)) - mu1_mu2
    C1, C2 = (0.01*max_val)**2, (0.03*max_val)**2
    return (((2*mu1_mu2 + C1)*(2*s12 + C2)) / ((mu1_sq + mu2_sq + C1)*(s1 + s2 + C2))).mean()
