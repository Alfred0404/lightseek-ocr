<!-- Ancre vers le haut -->

<br />
<div align="center">
  <a href="https://github.com/alfred0404/lightseek-ocr">
    <img src="images/logo.svg" alt="Logo" width="300">
  </a>

  <h3 align="center">LightSeek-OCR</h3>

  <p align="center">
    Implémentation légère et exploration de l'architecture de DeepSeek-OCR.
    <br />
    <a href="https://github.com/alfred0404/lightseek-ocr"><strong>Parcourir le repo »</strong></a>
  </p>
</div>

[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![LinkedIn][linkedin-shield]][linkedin-url]

## À propos du projet

LightSeek-OCR est une ré-implémentation légère et expérimentale inspirée par DeepSeek-OCR : convertir du texte en représentations visuelles afin d'étendre la fenêtre de contexte effective des modèles de langage.

Le but de ce dépôt est de fournir une implémentation simple et lisible pour permettre aux chercheurs et développeurs d'expérimenter et d'évaluer si la compression texte $\rightarrow$ image peut réduire le nombre de tokens tout en conservant l'information.

<p align="center">
    <img src="images/DeepSeek-OCR_Architecture.png" alt="Fig 1. DeepSeek-OCR Architecture" style="border-radius:5px">
    <em>Fig 1. DeepSeek-OCR Architecture</em>
</p>

## Getting Started

Prérequis

- Python 3.8+
- pip

Cloner le dépôt :

```bash
git clone https://github.com/alfred0404/lightseek-ocr.git
cd lightseek-ocr
```

Installer les dépendances :

```bash
pip install -r requirements.txt
```

## Architecture

LightSeek-OCR implémente un pipeline d'encodage modulaire inspiré de l'architecture DeepSeek-OCR. Le projet est structuré en composants indépendants pour faciliter l'expérimentation et la réutilisation.

### Structure du Projet

```
lightseek-ocr/
├── src/
│   ├── DeepEncoder.py           # Pipeline principal d'encodage
│   ├── SAMFeatureExtractor.py   # Extraction de features SAM (locales)
│   ├── CLIPVisionProcessor.py   # Traitement CLIP (features globales)
│   └── Conv2DCompressor.py      # Compression par convolution
├── images/                      # Ressources visuelles et logo
├── ressources/                  # Documents et références
├── requirements.txt             # Dépendances Python
└── TODO.md                      # Suivi des tâches
```

### Pipeline d'Encodage (`DeepEncoder`)

Le pipeline transforme du texte en représentations visuelles multi-échelles :

1. **Rendu Texte → Image** : Conversion du texte en image via PIL
2. **Extraction SAM** : Features spatiales fines (256 canaux, 64×64)
3. **Compression** : Réduction dimensionnelle via convolution (1024 canaux, 16×16)
4. **Traitement CLIP** : Génération d'une séquence de tokens sémantiques (256 tokens, 768-dim)

**Sorties ("Visual Plugs")** :

- **Features locales** (SAM) : `(B, 256, 64, 64)` — détails spatiaux fins
- **Features globales** (CLIP) : `(B, 256, 768)` — séquence de tokens sémantiques

Ces deux tenseurs servent d'entrée au décodeur OCR (à implémenter).

### Composants

#### `SAMFeatureExtractor`

Utilise le modèle Segment Anything (SAM) de Meta pour extraire des features visuelles riches à partir d'images de texte rendu. Produit une carte de features dense (64×64) avec 256 canaux.

#### `Conv2DCompressor`

Compresse les features SAM via une convolution à stride 16 pour réduire la résolution spatiale tout en augmentant la dimensionnalité (256 → 768 canaux). Crée une grille compacte de 16×16 = 256 tokens.

#### `CLIPVisionProcessor`

Traite les features compressées à travers l'encodeur vision de CLIP en **contournant la couche d'embedding** :

- Ajoute des embeddings positionnels interpolés
- Pas de token [CLS] ni de pooling
- Retourne la séquence complète de tokens (pas d'agrégation)

### Utilisation

```python
from src.DeepEncoder import DeepEncoder

# Initialiser l'encodeur
encoder = DeepEncoder(verbose=True)

# Encoder du texte → features visuelles
results = encoder.encode("Votre texte ici")

# Accéder aux sorties
local_features = results['local_features']    # (B, 256, 64, 64) -> 25- channels et 64*64 = 4096 tokens
global_features = results['global_features']  # (B, 256, 768) -> 768 channels et 16*16 = 256 tokens
```

## Utilisation

- Flux de travail typique :
  1. Préparer un corpus de texte.
  2. Rendre des sections de texte en images.
  3. Exécuter l'OCR/encodeur et évaluer la performance.

## Contribuer

Les contributions sont bienvenues : signalez des bugs via des issues et proposez des pull requests pour les améliorations.

## Licence

Distribué sous la licence du projet. Voir `LICENSE.txt` pour les détails.

<p align="center">
	<a href="https://github.com/alfred0404/lightseek-ocr/LICENSE"><img src="https://img.shields.io/static/v1.svg?style=for-the-badge&label=License&message=MIT&logoColor=d9e0ee&colorA=363a4f&colorB=b7bdf8"/></a>
</p>

## Ressources

Même si les modèles ne seront pas recodés from scratch, il est primordial d'en comprendre le fonctionnement en profondeur pour mener à bien l'implémentation de l'architecture.

| Ressource                                                                                                                   | Description                                                          |
| :-------------------------------------------------------------------------------------------------------------------------- | :------------------------------------------------------------------- |
| [Segment Anything Model (SAM): Explained](https://medium.com/@utkarsh135/segment-anything-model-sam-explained-2900743cb61e) | Article Medium sur l'explication du modèle Segment Anything (SAM).             |
| [DeepSeek-OCR GitHub](https://github.com/deepseek-ai/DeepSeek-OCR/tree/main)                                                | Dépôt GitHub officiel du projet DeepSeek-OCR.                        |
| [SAM : Segment Anything – Meilleur Tutoriel](https://inside-machinelearning.com/sam-segment-anything/)                      | Tutoriel en français pour maîtriser SAM.                             |
| [Segment Anything GitHub](https://github.com/facebookresearch/segment-anything)                                             | Dépôt GitHub officiel du projet Segment Anything de Meta.            |
| [Sliding Window Attention](https://medium.com/@manojkumal/sliding-window-attention-565f963a1ffd)                            | Article Medium expliquant le mécanisme d'attention "Sliding Window". |
| [What makes deepseek-ocr so powerful ?](https://learnopencv.com/what-makes-deepseek-ocr-so-powerful/)                       | Analyse par LearnOpenCV des forces de l'architecture DeepSeek-OCR.   |
| [Documentation de SAM *(transformers)*](https://huggingface.co/docs/transformers/v4.57.1/en/model_doc/sam#transformers.SamImageProcessor)                                                   | Documentation du modèle SAM dans la librairie transformers  |

<p align="center">
	<img src="https://raw.githubusercontent.com/catppuccin/catppuccin/main/assets/footers/gray0_ctp_on_line.svg?sanitize=true" />
</p>

<!-- LINKS & IMAGES -->
<!-- Contributors -->

[contributors-shield]: https://img.shields.io/github/contributors/alfred0404/lightseek-ocr.svg?style=for-the-badge
[contributors-url]: https://github.com/alfred0404/lightseek-ocr/graphs/contributors

<!-- Forks -->

[forks-shield]: https://img.shields.io/github/forks/alfred0404/lightseek-ocr.svg?style=for-the-badge
[forks-url]: https://github.com/alfred0404/lightseek-ocr/network/members

<!-- Stars -->

[stars-shield]: https://img.shields.io/github/stars/alfred0404/lightseek-ocr.svg?style=for-the-badge
[stars-url]: https://github.com/alfred0404/lightseek-ocr/stargazers

<!-- Issues -->

[issues-shield]: https://img.shields.io/github/issues/alfred0404/lightseek-ocr.svg?style=for-the-badge
[issues-url]: https://github.com/alfred0404/lightseek-ocr/issues

<!-- License -->

[license-shield]: https://img.shields.io/github/license/alfred0404/lightseek-ocr.svg?style=for-the-badge
[license-url]: https://github.com/alfred0404/lightseek-ocr/blob/master/LICENSE.txt

<!-- Linkedin -->

[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/alfred-de-vulpian

<!-- Other links -->

[DeepSeek-OCR_GitHub]: https://github.com/deepseek-ai/DeepSeek-OCR/tree/main
[Python]: https://img.shields.io/badge/python-000000?style=for-the-badge&logo=python&
[Python-url]: https://www.python.org/

<!-- Logo Colors -->

[deepseek-blue]: #4d6bfe
[deepseek-blue-complementary]: #E0C570

<!-- Images -->

[DeepSeek-OCR_Architecture_path]: images/DeepSeek-OCR_Architecture.png
