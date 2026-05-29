import time
import torch
from torch.utils.data import DataLoader
from .losses import CombinedLoss
from ..utils.metrics import psnr, ssim
from ..data.dataset import MaskedFaceDataset, build_dataloaders
from ..models.unet import UNet
from ..models.discriminator import PatchDiscriminator
import torch.nn.functional as F

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

def train_gan(task, cfg, index_df, device='cuda', verbose=True):
    loaders = build_dataloaders(index_df, task=task, cfg=cfg, device_type=device)
    scale = 2 if task == 'sr' else 1
    
    # 1. Models
    generator = UNet(base=cfg['base_ch'], scale_factor=scale).to(device)
    # PatchDiscriminator attends 6 canaux: image d'entrée masquée (3) + image générée/réelle (3)
    discriminator = PatchDiscriminator(in_ch=6, base_ch=64, use_spectral_norm=True).to(device)
    
    # 2. Losses
    criterion_g = CombinedLoss(cfg['lambda_perc']).to(device)
    criterion_gan = torch.nn.BCEWithLogitsLoss().to(device)
    
    # 3. Optimizers
    opt_g = torch.optim.Adam(generator.parameters(), lr=cfg['lr'], betas=cfg['betas'])
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=cfg['lr'], betas=cfg['betas'])
    scaler_g = torch.amp.GradScaler('cuda', enabled=cfg['use_amp'])
    scaler_d = torch.amp.GradScaler('cuda', enabled=cfg['use_amp'])

    val_df = index_df[index_df['split'] == 'val'].head(cfg['val_subset'])
    quick_val_ds = MaskedFaceDataset(val_df, task=task, lr_size=cfg['lr_size'], hr_size=cfg['hr_size'], augment=False)
    quick_val_loader = DataLoader(quick_val_ds, batch_size=cfg['batch_size'], num_workers=cfg.get('num_workers', 2), pin_memory=(device=='cuda'))

    history = {'epoch': [], 'train_loss_g': [], 'train_loss_d': [], 'val_psnr': [], 'val_ssim': [], 'epoch_time': []}
    best_psnr, best_state_g = -float('inf'), None

    for epoch in range(1, cfg['epochs']+1):
        t0 = time.time()
        generator.train()
        discriminator.train()
        running_g, running_d, nb = 0.0, 0.0, 0
        
        for xb, yb in loaders['train']:
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            
            # Redimensionner l'entrée pour qu'elle corresponde à la sortie (pour la concatenation)
            if task == 'sr':
                xb_resized = F.interpolate(xb, size=yb.shape[2:], mode='bilinear', align_corners=False)
            else:
                xb_resized = xb
                
            # ---------------------
            # Train Discriminator
            # ---------------------
            opt_d.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=cfg['use_amp']):
                with torch.no_grad():
                    fake_yb = generator(xb)
                
                fake_pred = discriminator(torch.cat([xb_resized, fake_yb], dim=1))
                real_pred = discriminator(torch.cat([xb_resized, yb], dim=1))
                
                loss_d_fake = criterion_gan(fake_pred, torch.zeros_like(fake_pred))
                loss_d_real = criterion_gan(real_pred, torch.ones_like(real_pred))
                loss_d = (loss_d_real + loss_d_fake) * 0.5
                
            scaler_d.scale(loss_d).backward()
            scaler_d.step(opt_d)
            scaler_d.update()
            
            # ---------------------
            # Train Generator
            # ---------------------
            opt_g.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=cfg['use_amp']):
                fake_yb = generator(xb)
                fake_pred = discriminator(torch.cat([xb_resized, fake_yb], dim=1))
                
                loss_g_adv = criterion_gan(fake_pred, torch.ones_like(fake_pred))
                loss_g_rec, _ = criterion_g(fake_yb, yb)
                
                # Le poids 0.1 pour loss_g_adv est un standard pour équilibrer texture et couleurs
                loss_g = loss_g_rec + 0.1 * loss_g_adv
                
            scaler_g.scale(loss_g).backward()
            scaler_g.step(opt_g)
            scaler_g.update()
            
            running_g += loss_g.item() * xb.size(0)
            running_d += loss_d.item() * xb.size(0)
            nb += xb.size(0)

        train_loss_g = running_g / nb
        train_loss_d = running_d / nb

        _, val_psnr, val_ssim = evaluate(generator, quick_val_loader, criterion_g, device=device)
        dt = time.time() - t0

        history['epoch'].append(epoch)
        history['train_loss_g'].append(train_loss_g)
        history['train_loss_d'].append(train_loss_d)
        history['val_psnr'].append(val_psnr)
        history['val_ssim'].append(val_ssim)
        history['epoch_time'].append(dt)

        is_best = val_psnr > best_psnr
        if is_best:
            best_psnr = val_psnr
            best_state_g = {k: v.detach().cpu().clone() for k, v in generator.state_dict().items()}

        if verbose:
            flag = ' 🔥' if is_best else ''
            print(f"  Epoch {epoch:2d}/{cfg['epochs']}  "
                  f"G={train_loss_g:.4f} D={train_loss_d:.4f}  "
                  f"PSNR={val_psnr:5.2f}dB  SSIM={val_ssim:.4f}  [{dt:5.1f}s]{flag}")

    generator.load_state_dict(best_state_g)
    return generator, discriminator, history
