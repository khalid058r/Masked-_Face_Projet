import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms.functional as TF
from PIL import Image

# Palette de couleurs pour simuler des masques de toutes couleurs
_MASK_COLORS = [
    (0,   0,   0),    # noir
    (255, 255, 255),  # blanc
    (80,  80,  80),   # gris foncé
    (200, 200, 200),  # gris clair
    (139, 0,   0),    # rouge foncé
    (0,   100, 0),    # vert foncé
    (75,  0,   130),  # violet
    (255, 140, 0),    # orange
    (165, 42,  42),   # marron
    (0,   0,   139),  # bleu foncé
    (255, 20,  147),  # rose
    (64,  64,  64),   # gris anthracite
]


def _recolor_mask(masked: Image.Image, unmasked: Image.Image) -> Image.Image:
    """
    Remplace les pixels du masque par une couleur aléatoire.
    La zone masquée est détectée en comparant l'image masquée et non-masquée.
    """
    m = np.array(masked,   dtype=np.float32)
    u = np.array(unmasked, dtype=np.float32)

    # Détection de la zone masquée : pixels avec grande différence
    diff = np.abs(m - u).mean(axis=2)          # (H, W)
    threshold = max(diff.max() * 0.25, 10.0)   # au moins 10/255 d'écart
    binary = (diff > threshold).astype(np.float32)  # (H, W)  1 = masque

    if binary.sum() < 50:   # pas assez de pixels détectés → pas de recoloriage
        return masked

    color = np.array(random.choice(_MASK_COLORS), dtype=np.float32)
    mask3d = binary[:, :, np.newaxis]          # (H, W, 1)
    result = m * (1 - mask3d) + color * mask3d
    return Image.fromarray(result.clip(0, 255).astype(np.uint8))


class MaskedFaceDataset(Dataset):
    MEAN = [0.5, 0.5, 0.5]
    STD  = [0.5, 0.5, 0.5]

    def __init__(self, df, task='inpainting', lr_size=64, hr_size=128,
                 augment=False, recolor_augment=False):
        assert task in ('inpainting', 'sr')
        self.df = df.reset_index(drop=True)
        self.task = task
        self.lr_size = lr_size
        self.hr_size = hr_size
        self.augment = augment
        self.recolor_augment = recolor_augment   # diversification couleur masque
        self._norm = transforms.Normalize(self.MEAN, self.STD)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        m = Image.open(row['path_masked']).convert('RGB')
        u = Image.open(row['path_unmasked']).convert('RGB')

        if m.size != (self.hr_size, self.hr_size):
            m = m.resize((self.hr_size,) * 2, Image.BICUBIC)
        if u.size != (self.hr_size, self.hr_size):
            u = u.resize((self.hr_size,) * 2, Image.BICUBIC)

        # Flip horizontal
        if self.augment and random.random() < 0.5:
            m = TF.hflip(m)
            u = TF.hflip(u)

        # Recoloriage aléatoire du masque (70 % du temps pendant l'entraînement)
        # Garde 30 % d'images bleues originales pour ne pas oublier la distribution initiale
        if self.recolor_augment and random.random() < 0.7:
            m = _recolor_mask(m, u)

        if self.task == 'sr':
            m = m.resize((self.lr_size,) * 2, Image.BICUBIC)

        return self._norm(TF.to_tensor(m)), self._norm(TF.to_tensor(u))

    @classmethod
    def denormalize(cls, t):
        t = t.clone().cpu()
        for c, (m, s) in enumerate(zip(cls.MEAN, cls.STD)):
            (t[c] if t.ndim == 3 else t[:, c]).mul_(s).add_(m)
        return t.clamp(0, 1)


def build_dataloaders(index_df, task, cfg, device_type='cuda'):
    loaders = {}
    recolor = cfg.get('recolor_augment', False)
    for split in ['train', 'val', 'test']:
        sub = index_df[index_df['split'] == split]
        if len(sub) == 0:
            continue
        ds = MaskedFaceDataset(
            sub, task=task,
            lr_size=cfg['lr_size'], hr_size=cfg['hr_size'],
            augment=(split == 'train'),
            recolor_augment=(recolor and split == 'train'),  # seulement en train
        )
        loaders[split] = DataLoader(
            ds, batch_size=cfg['batch_size'],
            shuffle=(split == 'train'),
            num_workers=cfg.get('num_workers', 2),
            pin_memory=(device_type == 'cuda'),
            drop_last=(split == 'train'),
        )
    return loaders
