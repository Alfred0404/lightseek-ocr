<!-- Back to top anchor -->

<br />
<div align="center">
  <a href="https://github.com/alfred0404/lightseek-ocr">
    <img src="images/logo.svg" alt="Logo" width="300">
  </a>

  <h3 align="center">LightSeek-OCR</h3>

  <p align="center">
    Lightweight implementation and exploration of the DeepSeek-OCR architecture.
    <br />
    <a href="https://github.com/alfred0404/lightseek-ocr"><strong>Explore the repo »</strong></a>
  </p>
</div>

[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![project_license][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url]

## About the project

LightSeek-OCR is a lightweight, experimental re-implementation inspired by the DeepSeek-OCR idea: encode textual context as visual representations to help extend effective context windows for language models.

This repository aims to provide a small, readable implementation so researchers and engineers can try the approach, reproduce experiments, and evaluate whether text-as-image compression can reduce token counts while preserving information.

<p align="center">
    <img src="images/DeepSeek-OCR_Architecture.png" alt="Fig 1. DeepSeek-OCR Architecture" style="border-radius:5px">
    <em>Fig 1. DeepSeek-OCR Architecture</em>
</p>

<p align="center"><a href="#readme-top">back to top</a></p>

## Getting started

Prerequisites

- Python 3.8+
- pip

Install dependencies (recommended inside a virtual environment):

Clone the repository:

```bash
git clone https://github.com/alfred0404/lightseek-ocr.git
cd lightseek-ocr
```

```bash
pip install -r requirements.txt
```

## Usage

- Typical workflow:
  1. Prepare a text corpus.
  2. Render text sections to images using the provided renderer.
  3. Run the OCR/encoder and evaluate compressed-context performance.

## Contributing

Contributions are welcome. Please open issues for bugs or feature requests and submit pull requests for improvements.

1. Fork the repository
2. Create a branch (`git checkout -b feature/my-feature`)
3. Commit your changes (`git commit -m "Add feature"`)
4. Push and open a PR

## License

Distributed under the project license. See `LICENSE.txt` for details.

<p align="center">
	<img src="https://raw.githubusercontent.com/catppuccin/catppuccin/main/assets/footers/gray0_ctp_on_line.svg?sanitize=true" />
</p>

<!-- <p align="center">
	<a href="https://github.com/alfred0404/lightseek-ocr/LICENSE"><img src="https://img.shields.io/static/v1.svg?style=for-the-badge&label=License&message=MIT&logoColor=d9e0ee&colorA=363a4f&colorB=b7bdf8"/></a>
</p> -->

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
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555

<!-- Linkedin -->

[linkedin-url]: https://linkedin.com/in/alfred-de-vulpian
[DeepSeek-OCR_GitHub]: https://github.com/deepseek-ai/DeepSeek-OCR/tree/main
[Python]: https://img.shields.io/badge/python-000000?style=for-the-badge&logo=python&
[Python-url]: https://www.python.org/

<!-- Logo Colors -->

[deepseek-blue]: #4d6bfe
[deepseek-blue-complementary]: #E0C570

<!-- Images -->

[DeepSeek-OCR_Architecture_path]: images/DeepSeek-OCR_Architecture.png
