<!-- Back to top anchor -->

<a id="readme-top"></a>

[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![project_license][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url]

<br />
<div align="center">
  <a href="https://github.com/alfred0404/lightseek-ocr">
    <img src="images/logo.png" alt="Logo" width="80" height="80">
  </a>

  <h3 align="center">LightSeek-OCR</h3>

  <p align="center">
    Lightweight implementation and exploration of the DeepSeek-OCR concept.
    <br />
    <a href="https://github.com/alfred0404/lightseek-ocr"><strong>Explore the repo »</strong></a>
  </p>
</div>

## English

### About the project

LightSeek-OCR is a lightweight, experimental re-implementation inspired by the DeepSeek-OCR idea: encode textual context as visual representations to help extend effective context windows for language models.

This repository aims to provide a small, readable implementation so researchers and engineers can try the approach, reproduce experiments, and evaluate whether text-as-image compression can reduce token counts while preserving information.

Key goals:

- Provide a simple, well-documented implementation
- Offer example pipelines for converting text -> image -> OCR-style embeddings
- Enable reproducible experiments and easy extension

<p align="center"><a href="#readme-top">back to top</a></p>

### Getting started

Prerequisites

- Python 3.8+
- pip

Install dependencies (recommended inside a virtual environment):

```bash
pip install -r requirements.txt
```

Clone the repository:

```bash
git clone https://github.com/alfred0404/lightseek-ocr.git
cd lightseek-ocr
```

### Usage

- See the `examples/` folder for small demo scripts (text -> image -> evaluate).
- Typical workflow:
  1. Prepare a text corpus.
  2. Render text sections to images using the provided renderer.
  3. Run the OCR/encoder and evaluate compressed-context performance.

### Contributing

Contributions are welcome. Please open issues for bugs or feature requests and submit pull requests for improvements.

1. Fork the repository
2. Create a branch (`git checkout -b feature/my-feature`)
3. Commit your changes (`git commit -m "Add feature"`)
4. Push and open a PR

<p align="center"><a href="#readme-top">back to top</a></p>

## License

Distributed under the project license. See `LICENSE.txt` for details.

## Contact

Alfred de Vulpian - derfladv@gmail.com

Project: https://github.com/alfred0404/lightseek-ocr

<p align="center"><a href="#readme-top">back to top</a></p>

<!-- MARKDOWN LINKS & IMAGES -->

[contributors-shield]: https://img.shields.io/github/contributors/alfred0404/lightseek-ocr.svg?style=for-the-badge
[contributors-url]: https://github.com/alfred0404/lightseek-ocr/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/alfred0404/lightseek-ocr.svg?style=for-the-badge
[forks-url]: https://github.com/alfred0404/lightseek-ocr/network/members
[stars-shield]: https://img.shields.io/github/stars/alfred0404/lightseek-ocr.svg?style=for-the-badge
[stars-url]: https://github.com/alfred0404/lightseek-ocr/stargazers
[issues-shield]: https://img.shields.io/github/issues/alfred0404/lightseek-ocr.svg?style=for-the-badge
[issues-url]: https://github.com/alfred0404/lightseek-ocr/issues
[license-shield]: https://img.shields.io/github/license/alfred0404/lightseek-ocr.svg?style=for-the-badge
[license-url]: https://github.com/alfred0404/lightseek-ocr/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/alfred-de-vulpian
[DeepSeek-OCR]: https://github.com/deepseek-ai/DeepSeek-OCR/tree/main
[product-screenshot]: images/screenshot.png
[Python]: https://img.shields.io/badge/python-000000?style=for-the-badge&logo=python&
[Python-url]: https://www.python.org/
