# OpenSAE


## Installation

### Install via pip

This project is maintained with setuptools, so you can install it via pip directly. It requires `Python 3.12` (or higher version).

```bash
git clone git@github.com:THU-KEG/OpenSAE.git
cd OpenSAE
pip install -e .
```


### Install via Docker

Additionally, we provide a Docker image for the project. You can build the image by running the following command:

```bash
docker push transirius/sae:latest

docker run --gpus all \
    -it --rm -d \
    --name sae \
    -v {SAE CHECKPOINTS DIR}:/CHECKPOINTS \
    -v {TRAINING DATA DIR}:/DATA \
    -v {LLM CHECKPOINTS DIR}:/MODELS \
    sae:latest \
```

The Docker image is built on top of the `nvcr.io/nvidia/pytorch:24.02-py3` image. We inject a miniconda environment with the required dependencies into the image.


## Acknowledgements

This project draws inspiration from various third-party SAE (Sparse Autoencoder) tools. We would like to express our heartfelt gratitude to the following:

- [transformers](https://github.com/huggingface/transformers) by HuggingFace: The architecture design of OpenSAE is significantly influenced by their implementation.
- [sparse_autoencoder](https://github.com/openai/sparse_autoencoder) by OpenAI: We have adapted their kernel implementation and some of their training tricks.
- [sae](https://github.com/EleutherAI/sae) by EleutherAI: Our training pipeline is largely inspired by their work.
We deeply appreciate the contributions of these projects to the open-source community, which have been invaluable to the development of this project.