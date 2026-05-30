import torch
import torch.nn as nn

def conv_block(in_c, out_c):
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_c), nn.ReLU(inplace=True),
        nn.Conv2d(out_c, out_c, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_c), nn.ReLU(inplace=True),
    )

class UNet(nn.Module):
    """U-Net 4-niveaux, output tanh -> [-1,1].
    scale_factor > 1 active un pre-upsampling bicubic pour la tâche SR."""
    def __init__(self, in_ch=3, out_ch=3, base=32, scale_factor=1):
        super().__init__()
        self.pre = nn.Upsample(scale_factor=scale_factor, mode='bicubic', align_corners=False) \
                   if scale_factor > 1 else nn.Identity()
        c = base
        self.enc1 = conv_block(in_ch, c)
        self.enc2 = conv_block(c, c*2)
        self.enc3 = conv_block(c*2, c*4)
        self.enc4 = conv_block(c*4, c*8)
        self.bott = conv_block(c*8, c*16)
        self.pool = nn.MaxPool2d(2)
        self.up4 = nn.ConvTranspose2d(c*16, c*8, 2, 2)
        self.dec4 = conv_block(c*16, c*8)
        self.up3 = nn.ConvTranspose2d(c*8, c*4, 2, 2)
        self.dec3 = conv_block(c*8, c*4)
        self.up2 = nn.ConvTranspose2d(c*4, c*2, 2, 2)
        self.dec2 = conv_block(c*4, c*2)
        self.up1 = nn.ConvTranspose2d(c*2, c, 2, 2)
        self.dec1 = conv_block(c*2, c)
        self.final = nn.Conv2d(c, out_ch, 1)
        
    def forward(self, x):
        x = self.pre(x)
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b  = self.bott(self.pool(e4))
        d4 = self.dec4(torch.cat([self.up4(b),  e4], 1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], 1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], 1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], 1))
        return torch.tanh(self.final(d1))
