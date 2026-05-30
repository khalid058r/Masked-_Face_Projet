import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class VGGPerceptualLoss(nn.Module):
    """Loss L1 sur les features VGG16 relu3_3. Input attendu en [-1,1].
    Fallback automatique sur L1 seul si VGG16 indisponible (pas de réseau)."""
    def __init__(self, layer_idx=15):
        super().__init__()
        self.available = False
        try:
            vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features
            self.slice = nn.Sequential(*list(vgg.children())[:layer_idx+1]).eval()
            for p in self.slice.parameters():
                p.requires_grad = False
            self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1))
            self.register_buffer('std',  torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1))
            self.available = True
        except Exception as e:
            print(f'VGG16 indisponible ({type(e).__name__}) -> fallback L1 seul pour la partie perceptual')
            
    def _prep(self, x):
        return ((x + 1) / 2 - self.mean) / self.std
        
    def forward(self, pred, target):
        if not self.available:
            return F.l1_loss(pred, target)  # fallback
        return F.l1_loss(self.slice(self._prep(pred)), self.slice(self._prep(target)))

class CombinedLoss(nn.Module):
    def __init__(self, lambda_perc=0.1):
        super().__init__()
        self.perc = VGGPerceptualLoss()
        self.lambda_perc = lambda_perc
        
    def forward(self, pred, target):
        l1 = F.l1_loss(pred, target)
        lp = self.perc(pred, target)
        return l1 + self.lambda_perc * lp, {'l1': l1.item(), 'perc': lp.item()}

def compute_r1_penalty(real_preds, real_inputs):
    """Compute R1 gradient penalty for discriminator."""
    grad_real, = torch.autograd.grad(
        outputs=real_preds.sum(),
        inputs=real_inputs,
        create_graph=True,
        retain_graph=True
    )
    grad_penalty = grad_real.pow(2).reshape(grad_real.shape[0], -1).sum(1).mean()
    return grad_penalty
