import random
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms.functional as TF
from PIL import Image

class MaskedFaceDataset(Dataset):
    MEAN = [0.5, 0.5, 0.5]
    STD  = [0.5, 0.5, 0.5]
    
    def __init__(self, df, task='inpainting', lr_size=64, hr_size=128, augment=False):
        assert task in ('inpainting','sr')
        self.df = df.reset_index(drop=True)
        self.task = task
        self.lr_size = lr_size
        self.hr_size = hr_size
        self.augment = augment
        self._norm = transforms.Normalize(self.MEAN, self.STD)
        
    def __len__(self): 
        return len(self.df)
        
    def __getitem__(self, i):
        row = self.df.iloc[i]
        m = Image.open(row['path_masked']).convert('RGB')
        u = Image.open(row['path_unmasked']).convert('RGB')
        
        if m.size != (self.hr_size, self.hr_size): 
            m = m.resize((self.hr_size,)*2, Image.BICUBIC)
        if u.size != (self.hr_size, self.hr_size): 
            u = u.resize((self.hr_size,)*2, Image.BICUBIC)
            
        if self.augment and random.random() < 0.5:
            m = TF.hflip(m)
            u = TF.hflip(u)
            
        if self.task == 'sr':
            m = m.resize((self.lr_size,)*2, Image.BICUBIC)
            
        return self._norm(TF.to_tensor(m)), self._norm(TF.to_tensor(u))
        
    @classmethod
    def denormalize(cls, t):
        t = t.clone().cpu()
        for c, (m,s) in enumerate(zip(cls.MEAN, cls.STD)):
            (t[c] if t.ndim==3 else t[:,c]).mul_(s).add_(m)
        return t.clamp(0,1)

def build_dataloaders(index_df, task, cfg, device_type='cuda'):
    loaders = {}
    for split in ['train','val','test']:
        sub = index_df[index_df['split'] == split]
        if len(sub) == 0: continue
        ds = MaskedFaceDataset(sub, task=task, lr_size=cfg['lr_size'],
                               hr_size=cfg['hr_size'], augment=(split=='train'))
        loaders[split] = DataLoader(
            ds, batch_size=cfg['batch_size'],
            shuffle=(split=='train'),
            num_workers=cfg.get('num_workers', 2),
            pin_memory=(device_type=='cuda'),
            drop_last=(split=='train'))
    return loaders
