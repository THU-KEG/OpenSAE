import os
import sys

import torch
from torch import Tensor

from dataclasses import dataclass
from abc import abstractmethod

from transformers.utils import logging
from transformers.utils import ModelOutput
from transformers.modeling_utils import PreTrainedModel

logging.set_verbosity_info()
logger = logging.get_logger("sae")

try:
    from .sparse_kernels import TritonDecoder
    TRITON_ENABLED = True
except ImportError:
    TRITON_ENABLED = False
    logger.warning("Import sparse decoder failed, cannot use Triton implementation of SAE decoder.")

@dataclass
class SaeEncoderOutput(ModelOutput):
    sparse_feature_activations: Tensor = None
    sparse_feature_indices: Tensor = None
    all_features: Tensor | None = None
    input_mean: Tensor | None = None
    input_std: Tensor | None = None


@dataclass
class SaeDecoderOutput(ModelOutput):
    sae_output: Tensor = None


@dataclass
class SaeForwardOutput(SaeEncoderOutput, SaeDecoderOutput):
    reconstruction_loss: Tensor = None
    auxk_loss: Tensor | None = None
    multi_topk_loss: Tensor | None = None
    l1_loss: Tensor | None = None
    l2_loss: Tensor | None = None
    loss: Tensor = None
    


class PreTrainedSae(PreTrainedModel):
    def __init__(self, config, **kwargs):
        self.config = config
        super().__init__(config, **kwargs)
        
    @abstractmethod
    def encode(self, **kwargs) -> SaeEncoderOutput:
        pass
    
    @abstractmethod
    def decode(self, **kwargs) -> SaeDecoderOutput:
        pass
    
    @abstractmethod
    def reconstruction_loss(
        self, 
        hidden: Tensor, 
        hidden_variance: Tensor, 
        sae_output: Tensor
    ) -> Tensor:
        pass
    
    @abstractmethod
    def forward(self, **kwargs) -> SaeForwardOutput:
        pass


# Fallback implementation of SAE decoder
def torch_decode(top_indices: Tensor, top_acts: Tensor, W_dec: Tensor):
    buf = top_acts.new_zeros(top_acts.shape[:-1] + (W_dec.shape[-1],))
    acts = buf.scatter_(dim=-1, index=top_indices, src=top_acts)
    return acts @ W_dec.mT


# Triton implementation of SAE decoder
def triton_decode(top_indices: Tensor, top_acts: Tensor, W_dec: Tensor):
    if TRITON_ENABLED:
        return TritonDecoder.apply(top_indices, top_acts, W_dec)
    else:
        raise ImportError("Triton not installed, cannot use Triton implementation of SAE decoder. Use `torch` implementation instead.")


def extend_encoder_output(original_encoder_output, new_encoder_output):
    for k in original_encoder_output:
        if k in new_encoder_output:
            original_encoder_output[k] = torch.concat(
                (original_encoder_output[k], new_encoder_output[k]),
                dim = 0
            )
    return original_encoder_output