# Masked Face Super Resolution

## Vision Globale
Ce projet a pour objectif de concevoir et implémenter un système intelligent capable de reconstruire et d'augmenter la résolution de visages recouverts par un masque (ou partiellement occultés). L'application s'inscrit dans un contexte d'amélioration d'images pour la restauration, la sécurité, ou encore la post-production photo/vidéo.

## Description Détaillée
L'entrée du système est une image d'un visage :
1. De basse résolution (basse qualité)
2. Dont une partie (nez, bouche, etc.) est cachée par un élément perturbateur (masque).

Le système IA "devine" la partie manquante en utilisant des informations contextuelles du visage et reconstitue la pleine résolution de l'image (Super-Resolution), de manière réaliste et proportionnelle.

## Choix Techniques (IA)
Pour ce projet, trois approches seront explorées, implémentées puis comparées :
- **Autoencodeurs (Baseline)** : Un réseau simple pour apprendre une représentation latente et décoder une version débruitée/réparée de l'image.
- **Réseaux Antagonistes Génératifs (GAN)** : Modèles optimisés pour la reconstruction de textures de peau très réalistes (haute fidélité perceptuelle).
- **Modèles de Diffusion** : Une méthode de pointe, produisant les transitions les plus harmonieuses et un visage très crédible mais nécessitant une puissance de calcul plus élevée pour l'inférence.

## Initialisation et Workflow

- Les données utilisées pour entraîner la preuve de concept (PoC) initial proviendront du dataset **CelebA** ainsi que du [dataset de visages masqués disponible sur Mendeley Data](https://data.mendeley.com/datasets/xyc9h3wjxf/2). Des masques seront appliqués de manière procédurale à des images haute définition pour générer le jeu d'entraînement.
- Le projet progresse de manière itérative :
  1. Préparation du pipeline de données (masquage/dégradation algorithmique).
  2. Création de la *Baseline* (Autoencodeurs).
  3. Implémentation itérative d'architectures plus complexes (GAN, Diffusion).

*Consultez docs/ARCHITECTURE.md pour explorer la structure du dépôt et le pipeline de code complet.*
