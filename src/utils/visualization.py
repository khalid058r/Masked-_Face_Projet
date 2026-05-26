import matplotlib.pyplot as plt
import torch

def plot_history(history, task_name='Model'):
    """Plot training curves."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    
    # Loss
    axes[0].plot(history['epoch'], history['train_loss'], label='Train Loss', linestyle='--')
    axes[0].plot(history['epoch'], history['val_loss'], label='Val Loss')
    axes[0].set_title(f'{task_name} - Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # PSNR
    if 'val_psnr' in history:
        axes[1].plot(history['epoch'], history['val_psnr'], label='Val PSNR', marker='o')
        axes[1].set_title(f'{task_name} - PSNR (dB)')
        axes[1].set_xlabel('Epoch')
        axes[1].legend()
        axes[1].grid(alpha=0.3)
        
    # SSIM
    if 'val_ssim' in history:
        axes[2].plot(history['epoch'], history['val_ssim'], label='Val SSIM', marker='o')
        axes[2].set_title(f'{task_name} - SSIM')
        axes[2].set_xlabel('Epoch')
        axes[2].legend()
        axes[2].grid(alpha=0.3)
        
    plt.tight_layout()
    return fig

def display_samples(masked, unmasked, pred=None, n_samples=4):
    """Display side-by-side comparison of masked, unmasked, and optionally prediction."""
    n_samples = min(n_samples, masked.size(0))
    cols = 3 if pred is not None else 2
    fig, axes = plt.subplots(n_samples, cols, figsize=(cols*3, n_samples*3))
    
    for i in range(n_samples):
        # Move channel from 0 to 2 for matplotlib and denormalize if needed
        m = masked[i].cpu().permute(1, 2, 0)
        u = unmasked[i].cpu().permute(1, 2, 0)
        
        # Adjust for [-1, 1] normalization
        m = (m + 1) / 2
        u = (u + 1) / 2
        
        ax_m = axes[i, 0] if n_samples > 1 else axes[0]
        ax_u = axes[i, 1] if n_samples > 1 else axes[1]
        
        ax_m.imshow(m.clamp(0, 1))
        ax_m.set_title('Masked')
        ax_m.axis('off')
        
        ax_u.imshow(u.clamp(0, 1))
        ax_u.set_title('Target')
        ax_u.axis('off')
        
        if pred is not None:
            p = pred[i].cpu().permute(1, 2, 0)
            p = (p + 1) / 2
            ax_p = axes[i, 2] if n_samples > 1 else axes[2]
            ax_p.imshow(p.clamp(0, 1))
            ax_p.set_title('Prediction')
            ax_p.axis('off')
            
    plt.tight_layout()
    return fig
