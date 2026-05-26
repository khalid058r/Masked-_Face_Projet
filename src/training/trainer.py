import time
import torch
from torch.utils.data import DataLoader
from .losses import CombinedLoss
from ..utils.metrics import psnr, ssim
from ..data.dataset import MaskedFaceDataset, build_dataloaders
from ..models.unet import UNet

@torch.no_grad()
def evaluate(model, loader, criterion, device='cuda', max_batches=None):
    model.eval()
    tot_loss, tot_psnr, tot_ssim, n = 0.0, 0.0, 0.0, 0
    for i, (xb, yb) in enumerate(loader):
        if max_batches is not None and i >= max_batches: break
        xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
        yh = model(xb)
        loss, _ = criterion(yh, yb)
        tot_loss += loss.item() * xb.size(0)
        tot_psnr += psnr(yh, yb).item() * xb.size(0)
        tot_ssim += ssim(yh, yb).item() * xb.size(0)
        n += xb.size(0)
    return tot_loss/n, tot_psnr/n, tot_ssim/n

def train_model(task, cfg, index_df, device='cuda', verbose=True):
    loaders = build_dataloaders(index_df, task=task, cfg=cfg, device_type=device)
    scale = 2 if task == 'sr' else 1
    model = UNet(base=cfg['base_ch'], scale_factor=scale).to(device)
    criterion = CombinedLoss(cfg['lambda_perc']).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['lr'], betas=cfg['betas'])
    scaler = torch.amp.GradScaler('cuda', enabled=cfg['use_amp'])

    val_df = index_df[index_df['split'] == 'val'].head(cfg['val_subset'])
    quick_val_ds = MaskedFaceDataset(val_df, task=task, lr_size=cfg['lr_size'],
                                     hr_size=cfg['hr_size'], augment=False)
    quick_val_loader = DataLoader(quick_val_ds, batch_size=cfg['batch_size'],
                                  num_workers=cfg.get('num_workers', 2),
                                  pin_memory=(device=='cuda'))

    history = {'epoch': [], 'train_loss': [], 'val_loss': [], 'val_psnr': [], 'val_ssim': [], 'epoch_time': []}
    best_psnr, best_state = -float('inf'), None

    for epoch in range(1, cfg['epochs']+1):
        t0 = time.time()
        model.train()
        running = 0.0; nb = 0
        for xb, yb in loaders['train']:
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=cfg['use_amp']):
                yh = model(xb)
                loss, _ = criterion(yh, yb)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running += loss.item() * xb.size(0); nb += xb.size(0)
        train_loss = running / nb

        val_loss, val_psnr, val_ssim = evaluate(model, quick_val_loader, criterion, device=device)
        dt = time.time() - t0

        history['epoch'].append(epoch)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_psnr'].append(val_psnr)
        history['val_ssim'].append(val_ssim)
        history['epoch_time'].append(dt)

        is_best = val_psnr > best_psnr
        if is_best:
            best_psnr = val_psnr
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if verbose:
            flag = ' 🔥' if is_best else ''
            print(f"  Epoch {epoch:2d}/{cfg['epochs']}  "
                  f"train={train_loss:.4f}  val={val_loss:.4f}  "
                  f"PSNR={val_psnr:5.2f}dB  SSIM={val_ssim:.4f}  [{dt:5.1f}s]{flag}")

    model.load_state_dict(best_state)
    return model, history
