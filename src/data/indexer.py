import os
from pathlib import Path
import pandas as pd

def build_index(data_dir: Path, parts, splits=('train','val','test')) -> pd.DataFrame:
    rows = []
    for part in parts:
        for split in splits:
            m_dir = data_dir / part / split / 'masked'
            u_dir = data_dir / part / split / 'unmasked'
            if not (m_dir.is_dir() and u_dir.is_dir()): continue
            m = {p.name: p for p in m_dir.glob('*.png')}
            u = {p.name: p for p in u_dir.glob('*.png')}
            for name in sorted(set(m) & set(u)):
                rows.append({'filename': name, 'path_masked': str(m[name]),
                             'path_unmasked': str(u[name]), 'part': part, 'split': split})
    return pd.DataFrame(rows)
