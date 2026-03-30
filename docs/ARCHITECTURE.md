# Architecture du Projet - Masked Face Super Resolution

Ce document fournit une vue d'ensemble de l'architecture logicielle et système du projet de réparation et super-résolution pour les visages masqués. 

## 1. Flux de Traitement (Pipeline de Données)

Le système complet repose sur la transformation d'une image faciale basse résolution et masquée en une image haute résolution et reconstruite.

1. **Acquisition (Data Ingestion)** : 
   - Images sources : Dataset CelebA (ou similaire).
   - *Data Augmentation* : Dégradation intentionnelle (réduction de résolution, ajout de bruit) + Génération et application d'un masque synthétique pour cacher une partie du visage.
2. **Prétraitement (Preprocessing)** :
   - Redimensionnement standard.
   - Normalisation des valeurs de pixels (ex: [-1, 1]).
   - Conversion en tenseurs PyTorch.
3. **Inférence Modèle (Model Forward Pass)** :
   - Passage des images dégradées à travers notre architecture d'intelligence artificielle.
4. **Post-traitement (Postprocessing)** :
   - Reconstitution de l'image (détensorisation, dé-normalisation).
   - Évaluation via des métriques (PSNR, SSIM, LPIPS).

## 2. Arborescence du Code

La structure logicielle a été conçue pour être modulaire, permettant d'expérimenter facilement avec différentes architectures (Autoencodeur, GAN, Diffusion) :

```text
Projet_Tutor/
│
├── data/                  # Données locales (ignoré par Git)
│   ├── raw/               # Dataset original non modifié
│   └── processed/         # Dataset avec masques et basse résolution appliqués
│
├── docs/                  # Documentation pour les développeurs
│   └── ARCHITECTURE.md    # Le présent document
│
├── notebooks/             # Notebooks Jupyter (exploration, visualisation)
│
├── src/                   # Code source principal
│   ├── data/              # Datasets PyTorch, DataLoaders, et scripts de masquage
│   ├── models/            # Implémentations des réseaux de neurones (GAN, VAE, etc.)
│   ├── training/          # Boucles d'entraînement, loss functions (fonctions de perte)
│   └── utils/             # Utilitaires (métriques, visualisation, logging)
│
└── tests/                 # Scripts de tests unitaires pour le pipeline (TDD)
```

## 3. Techniques d'Investigation (Modèles IA)

Au lieu de se restreindre à une seule technique initiale, le framework implémentera les bases de trois approches différentes pour des tests d'ingénierie et de performance :

- **Autoencodeurs (ex: VAE, U-Net)** : L'approche de base. Modèle "Encodeur-Décodeur" simple testable rapidement. Fournit des résultats stables mais souvent flous.
- **Generative Adversarial Networks (GANs)** : (ex: SRGAN ou ESRGAN adaptés). Un générateur répare l'image tandis qu'un discriminateur critique le réalisme. Fournit d'excellents détails (texture de la peau) mais difficile à stabiliser.
- **Modèles de Diffusion** : L'état de l'art actuel. Retrait progressif du bruit pour retrouver l'image de base. L'inférence est plus lente mais la qualité générative et la justesse structurelle au niveau des masques sont excellentes.

---
**Note aux collaborateurs** : Veillez à placer les poids des modèles lourds dans un dossier `.checkpoints/` qui sera ignoré par Git. Ne poussez aucun fichier de données brut dans le repository distant.