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

Même si les modèles ne seront pas recodés from scratch, il me parait important d'en comprendre le fonctionnement en profondeur.

| Ressource | Description |
| :--- | :--- |
| [Segment Anything Model (SAM): Explained](https://medium.com/@utkarsh135/segment-anything-model-sam-explained-2900743cb61e) | Explication du modèle Segment Anything (SAM) sur Medium. |
| [DeepSeek-OCR GitHub](https://github.com/deepseek-ai/DeepSeek-OCR/tree/main) | Dépôt GitHub officiel du projet DeepSeek-OCR. |
| [SAM : Segment Anything – Meilleur Tutoriel](https://inside-machinelearning.com/sam-segment-anything/) | Tutoriel en français pour maîtriser SAM. |
| [Segment Anything GitHub](https://github.com/facebookresearch/segment-anything) | Dépôt GitHub officiel du projet Segment Anything de Meta. |
| [Sliding Window Attention](https://medium.com/@manojkumal/sliding-window-attention-565f963a1ffd) | Article Medium expliquant le mécanisme d'attention "Sliding Window". |
| [What makes deepseek-ocr so powerful ?](https://learnopencv.com/what-makes-deepseek-ocr-so-powerful/) | Analyse par LearnOpenCV des forces de l'architecture DeepSeek-OCR. |

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
