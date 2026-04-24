# 📂 Échantillon du Dataset — Masked Face Super Resolution

Ce dossier contient un **échantillon représentatif** de l'ensemble du dataset utilisé pour le projet de super-résolution de visages masqués.

## 🎯 Objectif

Fournir un aperçu rapide de la structure et du contenu du dataset complet, sans avoir besoin de télécharger/décompresser les données volumineuses (~568 Mo pour part1 seule).

## 📊 Contenu de l'échantillon

**5 images** ont été sélectionnées aléatoirement dans chaque sous-catégorie :

| Part   | Split   | Masked | Unmasked | Total échantillon |
|--------|---------|--------|----------|-------------------|
| Part 1 | Train   | 5      | 5        | 10                |
| Part 1 | Test    | 5      | 5        | 10                |
| Part 1 | Val     | 5      | 5        | 10                |
| Part 2 | Train   | 5      | 5        | 10                |
| Part 2 | Test    | 5      | 5        | 10                |
| Part 2 | Val     | 5      | 5        | 10                |
| Part 3 | Train   | 5      | 5        | 10                |
| Part 3 | Test    | 5      | 5        | 10                |
| **Total** |      |        |          | **80 images**     |

## 📈 Statistiques du dataset complet

| Part   | Split   | Masked  | Unmasked | Total   |
|--------|---------|---------|----------|---------|
| Part 1 | Train   | 8 000   | 8 000    | 16 000  |
| Part 1 | Test    | 1 000   | 1 000    | 2 000   |
| Part 1 | Val     | 1 000   | 1 000    | 2 000   |
| Part 2 | Train   | 8 000   | 8 000    | 16 000  |
| Part 2 | Test    | 1 000   | 1 000    | 2 000   |
| Part 2 | Val     | 1 000   | 1 000    | 2 000   |
| Part 3 | Train   | 4 813   | 4 238    | 9 051   |
| Part 3 | Test    | 602     | 602      | 1 204   |
| Part 3 | Val     | —       | —        | —       |
| **Total** |      | **25 415** | **24 841** | **50 256** |

## 🗂️ Structure

```
sample/
├── part1/
│   ├── train/
│   │   ├── masked/      # 5 images de visages avec masque
│   │   └── unmasked/    # 5 images de visages sans masque (ground truth)
│   ├── test/
│   │   ├── masked/
│   │   └── unmasked/
│   └── val/
│       ├── masked/
│       └── unmasked/
├── part2/
│   ├── train/
│   │   ├── masked/
│   │   └── unmasked/
│   ├── test/
│   │   ├── masked/
│   │   └── unmasked/
│   └── val/
│       ├── masked/
│       └── unmasked/
├── part3/
│   ├── train/
│   │   ├── masked/
│   │   └── unmasked/
│   └── test/
│       ├── masked/
│       └── unmasked/
└── README.md
```

## 📝 Notes

- Les images `masked` et `unmasked` partagent le **même nom de fichier** (ex: `0000002.png`), formant des paires input/target pour l'entraînement.
- Format : **PNG**
- Les 3 parts proviennent de sources/splits différents du dataset original (CelebA + Mendeley Masked Faces).
- Part 3 ne contient pas de split de validation.
