import pytest
import pandas as pd
import torch
from PIL import Image
import numpy as np
from unittest.mock import patch
from src.data.dataset import MaskedFaceDataset

@pytest.fixture
def dummy_index():
    data = {
        'filename': ['img1.png'],
        'path_masked': ['dummy_m1.png'],
        'path_unmasked': ['dummy_u1.png'],
        'part': ['part1'],
        'split': ['train']
    }
    return pd.DataFrame(data)

@patch('src.data.dataset.Image.open')
def test_dataset_getitem_inpainting(mock_open, dummy_index):
    # Mock image opening
    mock_img = Image.fromarray(np.zeros((128, 128, 3), dtype=np.uint8))
    mock_open.return_value = mock_img
    
    ds = MaskedFaceDataset(dummy_index, task='inpainting', hr_size=128)
    m, u = ds[0]
    
    assert m.shape == (3, 128, 128)
    assert u.shape == (3, 128, 128)

@patch('src.data.dataset.Image.open')
def test_dataset_getitem_sr(mock_open, dummy_index):
    # Mock image opening
    mock_img = Image.fromarray(np.zeros((128, 128, 3), dtype=np.uint8))
    mock_open.return_value = mock_img
    
    ds = MaskedFaceDataset(dummy_index, task='sr', lr_size=64, hr_size=128)
    m, u = ds[0]
    
    assert m.shape == (3, 64, 64)
    assert u.shape == (3, 128, 128)
