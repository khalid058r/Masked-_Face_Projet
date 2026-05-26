import torch
from src.models.unet import UNet
from src.models.discriminator import PatchDiscriminator

def test_unet_inpainting():
    model = UNet(base=8, scale_factor=1)
    x = torch.randn(2, 3, 128, 128)
    y = model(x)
    assert y.shape == (2, 3, 128, 128)

def test_unet_sr():
    model = UNet(base=8, scale_factor=2)
    x = torch.randn(2, 3, 64, 64)
    y = model(x)
    assert y.shape == (2, 3, 128, 128)

def test_patch_discriminator():
    model = PatchDiscriminator(in_ch=6, base_ch=16)
    x = torch.randn(2, 6, 128, 128)
    y = model(x)
    assert y.shape[0] == 2
    assert y.shape[1] == 1
