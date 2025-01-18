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


## Quick Start

### 1. Load the SAE

OpenSAE allows to load the SAE with only one line of code:

```python
from opensae import OpenSae
OpenSae.from_pretrained(/dir/to/sae)
```


An SAE model comprise two key components: An encoder, which maps the input hidden to the high dimensional space with sparse activation; and a decoder, which decodes the sparse activation to reconstruct the hidden.

In OpenSAE, we implement the following interfaces:

#### encode()

This method implement the encoder forward pass.

**input**

- hidden: `torch.Tensor`, required. Shape = (tokens, hidden_size). To process multiple sentences in a batch, this method requires to flatten the tokens in the batch.
- return_all_features: `bool`, optional, default to `False`. When set to `True`, by calling `encode()` will reture all the features before sparse activation in the output class.

**output**

- SaeEncoderOutput: `OrderedDict`. Fields include:
    - sparse_feature_activations: The activation value of the sparse features in SAE **after** sparse activation.
    - sparse_feature_indices: The indices of activated features.
    - all_features: All the features **before** the sparse activation, which means hidden_size $\times$ expansion_ratio features per token.
    - input_mean: The average of `hidden` for LayerNorm.
    - input_std: The standard deviationof `hidden` for LayerNorm.

#### decode()

This method implement the decoder forward pass.

**input**

- feature_indices: `torch.LongTensor`, required. Shape = (tokens, num of sparse features). The sparse feature activation. Usually from `SaeEncoderOutput.sparse_feature_indices`. When use TopK activation, the num of sparse features is K.
- feature_activation: `torch.FloatTensor`, required. Shape = (tokens, num of sparse features) The indices of the sparse feature activation. Usually from `SaeEncoderOutput.sparse_feature_activations`.
- input_mean: `torch.FloatTensor`, optional. Shape = (tokens,). The average of `hidden` for LayerNorm. This is required when the SAE model performs shift_back.
- input_std: `torch.FloatTensor`, optional. Shape = (tokens,). The standard deviation of `hidden` for LayerNorm. This is required when the SAE model performs shift_back.

**output**

- SaeDecoderOutput: `OrderedDict`. Fields includ:
    - sae_output: The reconstruction for the input hidden.

#### reconstruction_loss()



#### forward()



### 2. Bind the SAE with a LLM


### 3. LLM Intervention


### 4. SAE AutoInterpret


### 5. SAE Evaluation


### 6. SAE Training


## Acknowledgements

This project draws inspiration from various third-party SAE (Sparse Autoencoder) tools. We would like to express our heartfelt gratitude to the following:

- [transformers](https://github.com/huggingface/transformers) by HuggingFace: The architecture design of OpenSAE is significantly influenced by their implementation.
- [sparse_autoencoder](https://github.com/openai/sparse_autoencoder) by OpenAI: We have adapted their kernel implementation and some of their training tricks.
- [sae](https://github.com/EleutherAI/sae) by EleutherAI: Our training pipeline is largely inspired by their work.
We deeply appreciate the contributions of these projects to the open-source community, which have been invaluable to the development of this project.