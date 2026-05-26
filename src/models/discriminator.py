import torch
import torch.nn as nn

from torch.nn.utils import spectral_norm

class PatchDiscriminator(nn.Module):
    """PatchGAN Discriminator (70x70) from Pix2Pix with optional Spectral Normalization."""
    def __init__(self, in_ch=6, base_ch=64, n_layers=3, use_spectral_norm=False):
        super().__init__()
        
        def conv_block(in_dim, out_dim, stride=2, norm=True):
            conv = nn.Conv2d(in_dim, out_dim, kernel_size=4, stride=stride, padding=1, bias=not norm)
            if use_spectral_norm:
                conv = spectral_norm(conv)
            layers = [conv]
            if norm and not use_spectral_norm:
                layers.append(nn.BatchNorm2d(out_dim))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        sequence = [*conv_block(in_ch, base_ch, norm=False)]
        
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += conv_block(base_ch * nf_mult_prev, base_ch * nf_mult)
            
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        
        conv = nn.Conv2d(base_ch * nf_mult_prev, base_ch * nf_mult, kernel_size=4, stride=1, padding=1, bias=False)
        if use_spectral_norm:
            conv = spectral_norm(conv)
            
        sequence += [conv]
        if not use_spectral_norm:
            sequence += [nn.BatchNorm2d(base_ch * nf_mult)]
            
        sequence += [nn.LeakyReLU(0.2, inplace=True)]
        
        final_conv = nn.Conv2d(base_ch * nf_mult, 1, kernel_size=4, stride=1, padding=1)
        if use_spectral_norm:
            final_conv = spectral_norm(final_conv)
        sequence += [final_conv]
        
        self.model = nn.Sequential(*sequence)

    def forward(self, x):
        return self.model(x)
