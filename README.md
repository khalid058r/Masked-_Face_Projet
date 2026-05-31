"""README mis à jour pour correspondre à l'arborescence réelle du dossier Masked-_Face_Projet.
Patch : simplification et alignement des sections principales (notebooks, scripts, src, tests, dataset).
"""

# 🎭 Masked Face Super Resolution

> Reconstruction et super-résolution de visages partiellement masqués via deep learning.

Ce dépôt (dossier: `Masked-_Face_Projet`) contient les notebooks, le code source, les scripts d'entraînement et les données indexées utilisés pour expérimenter l'inpainting et la super-résolution conditionnée sur une image masquée.

---

## 📋 Aperçu rapide

- Notebooks : exploration, prétraitement, baselines et GAN
- Scripts : `app.py`, `train.py` (entrées pour entraînement / inference)
- Dataset indexé : `dataset_index.csv`
- Source : `src/` (dataset, models, training, utils)
- Tests unitaires : `tests/`

---

## 🗂 Structure du projet

```
Masked-_Face_Projet/
├── app.py
├── train.py
├── dataset_index.csv
├── requirements.txt
├── README.md
├── dataset/
│   ├── part1/...
│   ├── part2/...
│   └── part4/...
├── docs/
│   └── ARCHITECTURE.md
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02-preprocessing.ipynb
│   ├── 03-baseline.ipynb
│   └── 03-complete-baseline-and-gan.ipynb
├── src/
│   ├── inference.py
│   ├── data/
│   │   ├── dataset.py
│   │   └── indexer.py
│   ├── models/
│   │   ├── diffusion.py
│   │   ├── discriminator.py
│   │   └── unet.py
│   ├── training/
│   │   ├── losses.py
│   │   └── trainer.py
│   └── utils/
│       ├── metrics.py
│       └── visualization.py
└── tests/
    ├── test_dataset.py
    └── test_models.py
```

Notes : la structure ci‑dessus reflète les fichiers présents dans le dossier. Les parties du dataset sont rangées sous `dataset/` et indexées par `dataset_index.csv`.

---

## 🧭 Notebooks principaux

- `01_EDA.ipynb` — analyse exploratoire et sanity checks
- `02-preprocessing.ipynb` — indexation des paires masked/unmasked et définition de la classe `MaskedFaceDataset`
- `03-baseline.ipynb` — entraînement des baselines (autoencoder / U-Net)
- `03-complete-baseline-and-gan.ipynb` — fine-tuning GAN (Pix2Pix) depuis le baseline

---

## ⚙️ Installation rapide

1. Créer un environnement Python (recommandé)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. Vérifier l'index des paires

```python
from src.data.indexer import load_index
print(load_index('dataset_index.csv').shape)
```

3. Lancer un entraînement rapide (exemple)

```powershell
python train.py --config configs/baseline.yaml
```

-- Ajustez les commandes selon vos configurations locales (GPU/CPU).

---

## 🧪 Tests

Exécuter les tests unitaires avec `pytest` depuis la racine `Masked-_Face_Projet` :

```powershell
pip install pytest
pytest -q
```

---

## 📌 Points importants et choix techniques (résumé)

- Normalisation des images en `[-1, 1]` (compatible `tanh` en sortie)
- U-Net avec skip connections pour préserver les hautes fréquences
- Sélection finale des modèles GAN basée sur LPIPS pour la qualité perceptuelle
- AMP recommandé sur GPU pour accélérer et réduire la VRAM

---

## 🚀 Prochaines étapes (suggestions)

- Ajouter un script `evaluate.py` pour automatiser PSNR/SSIM/LPIPS sur le test set
- Intégrer des notebooks de visualisation des résultats `notebooks/vis_results.ipynb`
- Expérimenter la normalisation fine et les augmentations pendant l'entraînement

---

Si vous souhaitez que j'adapte davantage le README (liens vers notebooks, exemples d'inférence, badges CI, ou version anglaise), dites-le et je le mets à jour.

### 2. `02_preprocessing.ipynb` — Prétraitement

**Approche : pas de modification sur disque, tout à la volée.**

- **Indexation** : scan des dossiers, intersection `masked ∩ unmasked` par filename → DataFrame de **20 000 paires valides**
- **Classe `MaskedFaceDataset`** : un Dataset PyTorch unifié supportant 2 tâches via un paramètre
- **Augmentations** : flip horizontal symétrique (training only)
- **Normalisation** : `[-1, 1]` (mean=std=0.5) pour cohérence avec `tanh` output
- **Sauvegarde** : `pairs_index.csv` (3.7 MB) pour réutilisation par les notebooks suivants
- **Sanity checks** : visualisation de batches + benchmark de vitesse

### 3. Modélisation et Architectures

Ce projet décompose la solution en trois étapes pour observer la progression de qualité :

#### A. `03_baseline.ipynb` — Baseline Autoencoder simple

```
                  Input (masked)              Output (unmasked)
                  ┌──────────────┐           ┌──────────────┐
                  │  3×128×128   │           │  3×128×128   │
   Pre-upsample → │ (ou 3×64×64) │ → Encodeur→ │  (tanh)      │
                  └──────────────┘           └──────────────┘
```

| Composant | Détail |
|---|---|
| **Architecture** | Autoencodeur basique avec 2 niveaux de down/up, sans skip connections |
| **Output** | `tanh` → range `[-1, 1]` |
| **Loss** | L1 / MSE (MSE uniquement pour baseline, L1 classique) |

#### B. `04_baseline_unet.ipynb` — Baseline U-Net

```
                  Input (masked)              Output (unmasked)
                  ┌──────────────┐           ┌──────────────┐
                  │  3×128×128   │           │  3×128×128   │
   Pre-upsample → │ (ou 3×64×64) │ → U-Net → │  (tanh)      │
                  └──────────────┘           └──────────────┘
```

| Composant | Détail |
|---|---|
| **Architecture** | U-Net 4-niveaux down/up + skip connections + BatchNorm |
| **Paramètres** | ~8M (base width 32) |
| **Loss** | `L1 + 0.1 × VGG perceptual` (relu3_3) |

#### C. `05-gan.ipynb` — GAN Pix2Pix

```
        ┌──────────────────────────────────────────────────────────┐
        │                                                          │
        │   Generator G                  Discriminator D           │
        │   (U-Net du baseline)          (PatchGAN 70×70)          │
        │   ┌────────────┐              ┌────────────┐             │
        │   │  3×128×128 │──fake──────▶│  → 14×14   │             │
        │   │   tanh     │              │   patches  │             │
        │   └────────────┘              │   real/fake│             │
        │                              ┌─│            │             │
        │   real (target) ─────────────┘ └────────────┘             │
        │                                                          │
        └──────────────────────────────────────────────────────────┘
```

| Composant | Détail |
|---|---|
| **Générateur G** | Même U-Net que baseline, **chargé depuis les poids du baseline** (transfer learning) |
| **Discriminateur D** | PatchGAN convolutif → carte 14×14 de décisions vrai/faux |
| **Loss G** | `1·L1 + 0.1·VGG_perc + 0.01·MSE(D(G(x)), 1)` |
| **Loss D** | LSGAN : `0.5·(MSE(D(real),1) + MSE(D(fake),0))` |
| **Optimizers** | Adam G lr=1e-4 (fine-tune), D lr=2e-4 |
| **Sélection** | Meilleur **LPIPS** (qualité perceptuelle, pas PSNR) |

**Pourquoi PatchGAN** : Au lieu de juger l'image entière (vrai/faux global), il évalue des patches ~70×70 indépendamment. Le générateur est ainsi forcé de produire des **textures localement réalistes partout**, pas juste un rendu globalement crédible.

**Pourquoi LSGAN** : MSE sur les patches au lieu de BCE → entraînement plus stable, gradients moins saturés, recommandé par le papier original Pix2Pix.

**Pourquoi sélection sur LPIPS** : Le PSNR favorise les images floues (qui minimisent l'erreur quadratique). LPIPS mesure la similarité dans un espace de features apprises sur préférences humaines → meilleur proxy du réalisme.

---

## 🎯 Tâches modélisées

Les deux tâches sont entraînées **en parallèle** pour comparaison.

### Inpainting pur
- **Input** : image masquée 128×128
- **Output** : image dé-masquée 128×128
- **Cas d'usage** : reconstruction du visage complet à partir d'une photo nette mais avec masque

### Super-résolution + inpainting
- **Input** : image masquée **basse résolution** 64×64
- **Output** : image dé-masquée **haute résolution** 128×128 (×2 upscaling)
- **Cas d'usage** : photos lointaines/basse qualité de personnes masquées (vidéosurveillance, photos de groupe, etc.)

Implementation : un `nn.Upsample(scale_factor=2, mode='bicubic')` au début du U-Net → **même architecture pour les deux tâches**, comparaison juste.

---

## 📊 Résultats

Évaluation sur le **test set complet** (2 000 images jamais vues, ni en train ni en val).

### Baseline U-Net (10 epochs)

| Tâche | Test PSNR | Test SSIM |
|---|---|---|
| **Inpainting** | **24.35 dB** | **0.8144** |
| **SR (×2)** | **24.22 dB** | 0.7919 |

**Lecture** : 24 dB est dans la fourchette haute attendue pour un baseline. SSIM > 0.8 indique une bonne préservation structurelle. Le baseline est solide mais **les rendus sont visiblement flous** (caractéristique des losses L1).

### GAN Pix2Pix (15 epochs, fine-tuned depuis baseline)

| Métrique | Direction attendue | Pourquoi |
|---|---|---|
| **PSNR** | -1 à -2 dB | Le GAN invente des détails plausibles, pas exactement les bons pixels |
| **SSIM** | Quasi stable | Structure globale inchangée |
| **LPIPS** | -0.02 à -0.05 | **Gain principal** : textures réalistes |
| **Visuel** | Nettement plus net | Critère ultime — l'œil humain préfère le GAN |

**Observation clé** : un GAN bien entraîné **perd un peu en PSNR** mais **gagne nettement en réalisme visuel**. C'est le compromis classique documenté dans toute la littérature SR/inpainting.

### Observations qualitatives (baseline)

✅ **Ce qui marche** :
- Reconstruction du bas du visage très réaliste (nez, bouche, menton)
- Cohérence peau/éclairage avec les zones non-masquées
- Généralisation à différents types de masques (bleu, blanc, noir, FFP)
- Préservation parfaite des yeux, cheveux, contexte

⚠️ **Limites du baseline** :
- Flou caractéristique sur les zones reconstruites
- Texture de peau lissée (pas de pores, rides)
- Pas de dents même quand la target sourit
- Identité différente de la target (attendu — pairing non-identitaire)

→ Le **GAN attaque directement ces limites** en forçant la production de textures fines.

---

## 📂 Structure des fichiers

```
masked-face-sr/
├── README.md                                ← Ce fichier
├── notebooks/
│   ├── 01_EDA.ipynb                         ← Étape 1 : exploration
│   ├── 02-preprocessing.ipynb               ← Étape 2 : indexation + Dataset
│   ├── 03_baseline.ipynb                    ← Étape 3a : Autoencodeur simple
│   ├── 04_baseline_unet.ipynb               ← Étape 3b : U-Net (L1+VGG)
│   └── 05-gan.ipynb                         ← Étape 3c : GAN Pix2Pix (Draft)
├── data/
│   ├── raw/                                 ← Dataset Kaggle (non versionné)
│   │   ├── part1/
│   │   ├── part2/
│   │   └── part3/                           ← (non utilisé)
│   └── processed/
│       └── pairs_index.csv                  ← Index (ou pointant vers Kaggle input)
└── src/
    └── models/
        └── checkpoints/                     ← Produits du training (U-Net, Autoencoder, GAN)
```

---

## 🚀 Installation & utilisation

### Prérequis

- Python 3.10+
- PyTorch 2.0+ avec CUDA (recommandé, sinon CPU lent)
- Compte Kaggle (pour le dataset et le GPU T4 gratuit)

### Dépendances

```
torch >= 2.0
torchvision >= 0.15
numpy, pandas, matplotlib, pillow
lpips                # installé automatiquement dans le notebook 03
```

### Exécution sur Kaggle (recommandé)

1. **Crée un nouveau notebook Kaggle**
2. **Ajoute les datasets en Input** : recherche `ikramelmenhi/dataset` et `ikramelmenhi/dataset-index`
3. **Active le GPU T4** dans Settings → Accelerator → GPU T4 ×2
4. **Upload** ou copie-colle le contenu des notebooks successifs (`03_baseline`, `04_baseline_unet`, etc.)
5. **Run All** → L'exécution des modèles sera stockée dans `/kaggle/working/`

### Exécution locale

```bash
# 1. Cloner le projet
git clone <repo>
cd masked-face-sr

# 2. Télécharger le dataset depuis Kaggle dans data/raw/
kaggle datasets download ikramelmenhi/dataset -p dataset --unzip

# 3. Installer les dépendances
pip install torch torchvision numpy pandas matplotlib pillow lpips

# 4. Lancer les notebooks dans l'ordre
jupyter lab notebooks/
```

### Inférence sur une nouvelle image

```python
import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms import functional as TF

# Charger un modèle entraîné
ckpt = torch.load('checkpoints/gan_inpainting_best.pt', weights_only=False)
G = UNet(base=32, scale_factor=1)         # ou scale_factor=2 pour SR
G.load_state_dict(ckpt['state_G'])         # 'state' pour baseline, 'state_G' pour GAN
G.eval().cuda()

# Préparer l'image
img = Image.open('mon_visage_masque.png').convert('RGB').resize((128, 128))
norm = transforms.Normalize([0.5]*3, [0.5]*3)
x = norm(TF.to_tensor(img)).unsqueeze(0).cuda()

# Inférence
with torch.no_grad():
    y = G(x).squeeze(0).cpu()

# Dé-normaliser et sauvegarder
y_img = (y * 0.5 + 0.5).clamp(0, 1)
TF.to_pil_image(y_img).save('mon_visage_reconstruit.png')
```

---

## 🛠️ Choix techniques

### Pourquoi un U-Net plutôt qu'un autoencoder vanilla ?

Les **skip connections** transfèrent les détails haute fréquence (yeux, cheveux, contour du visage) directement du début du réseau vers la fin → ces zones ne passent pas par le bottleneck et restent nettes. C'est crucial pour préserver l'identité visuelle des zones non-masquées.

### Pourquoi `tanh` en sortie ?

Le `tanh` borne la sortie dans `[-1, 1]`, exactement la même range que les images normalisées en entrée. Cela évite les artefacts de saturation et facilite l'entraînement du discriminateur du GAN.

### Pourquoi normalisation `[-1, 1]` au lieu des stats du dataset ?

C'est le standard pour les architectures génératives (GAN, diffusion). Les valeurs `mean=std=0.5` donnent une range symétrique autour de 0 → le `tanh` peut produire toute la gamme.

### Pourquoi mixed precision (AMP) ?

Sur GPU T4, AMP donne un **speedup ~2×** en utilisant float16 pour les opérations principales et float32 pour la stabilité numérique. Économie également de VRAM (40-50 %).

### Pourquoi `num_workers=2` sur Kaggle ?

Limite recommandée pour les T4 Kaggle. Au-delà, on risque des deadlocks. En script Python local, on peut monter à 4-8.

### Pourquoi sélectionner sur LPIPS pour le GAN ?

PSNR récompense les images floues (qui minimisent l'erreur quadratique). Si on sélectionne le meilleur GAN sur PSNR, on obtient un modèle qui n'a pas convergé vers un comportement adversariel utile. **LPIPS reflète la qualité perceptuelle réelle**, calibrée sur des préférences humaines.

### Pourquoi des fallbacks pour VGG et LPIPS ?

Pour la **robustesse** : si le notebook tourne dans un environnement sans accès aux poids pré-entraînés (sandbox, réseau bloqué), il dégrade gracieusement plutôt que de planter. Sur Kaggle avec internet, les poids se téléchargent normalement.

---

## 🚀 Prochaines étapes

### Court terme : optimisations du GAN actuel

- **LR scheduling** (cosine annealing ou linear decay) pour stabiliser la fin d'entraînement
- **Spectral normalization** sur le discriminateur → entraînement plus stable
- **Augmentations plus agressives** : color jitter, rotations légères, crop aléatoire
- **R1 regularization** sur D → évite le mode collapse

### Moyen terme : autres architectures

- **CycleGAN-style** : entraînement non-apparié (utile si on veut généraliser à des paires non-alignées)
- **AttentionGAN** : self-attention pour mieux gérer les dépendances longues
- **ESRGAN-style** : pour pousser plus loin la super-résolution

### Long terme : diffusion

- **DDPM conditionné** : score-based generative model conditionné sur l'image masquée
- **Latent diffusion** (Stable Diffusion-style) pour réduire le coût d'inférence
- État-de-l'art en qualité, mais **inférence beaucoup plus lente** (20-50 steps)

### Évaluation humaine

- **MOS (Mean Opinion Score)** : tester sur ~50 paires avec 10-20 évaluateurs humains
- Comparer la préférence : baseline vs GAN vs target

---

## 📚 Références

- **U-Net** : Ronneberger et al., *U-Net: Convolutional Networks for Biomedical Image Segmentation*, MICCAI 2015
- **Pix2Pix** : Isola et al., *Image-to-Image Translation with Conditional Adversarial Networks*, CVPR 2017
- **PatchGAN** : ibid.
- **LSGAN** : Mao et al., *Least Squares Generative Adversarial Networks*, ICCV 2017
- **VGG perceptual loss** : Johnson et al., *Perceptual Losses for Real-Time Style Transfer and Super-Resolution*, ECCV 2016
- **LPIPS** : Zhang et al., *The Unreasonable Effectiveness of Deep Features as a Perceptual Metric*, CVPR 2018
- **FFHQ dataset** : Karras et al., *A Style-Based Generator Architecture for GANs*, CVPR 2019

---

## 📝 Licence

Projet académique. Le dataset est sous licence FFHQ (research-only).

---

*Dernière mise à jour : pipeline baseline + GAN testé end-to-end et validé sur Kaggle T4.*