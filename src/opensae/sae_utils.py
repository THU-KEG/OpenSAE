import os
import sys

import torch
from torch import Tensor

from dataclasses import dataclass
from abc import abstractmethod

from transformers.utils import logging
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import ModelOutput

try:
    from .sparse_kernels import TritonDecoder
    TRITON_ENABLED = True
except ImportError:
    TRITON_ENABLED = False

@dataclass
class SaeEncoderOutput(ModelOutput):
    feature_activation: Tensor
    feature_indices: Tensor


@dataclass
class SaeDecoderOutput(ModelOutput):
    sae_output: Tensor


@dataclass
class SaeForwardOutput(SaeEncoderOutput, SaeDecoderOutput):
    loss: Tensor
    reconstruction_loss: Tensor
    auxk_loss: Tensor | None
    multi_topk_loss: Tensor | None
    l1_loss: Tensor | None
    


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
    def forward(self, **kwargs) -> SaeForwardOutput:
        pass


# Fallback implementation of SAE decoder
def eager_decode(top_indices: Tensor, top_acts: Tensor, W_dec: Tensor):
    buf = top_acts.new_zeros(top_acts.shape[:-1] + (W_dec.shape[-1],))
    acts = buf.scatter_(dim=-1, index=top_indices, src=top_acts)
    return acts @ W_dec.mT


# Triton implementation of SAE decoder
def triton_decode(top_indices: Tensor, top_acts: Tensor, W_dec: Tensor):
    if TRITON_ENABLED:
        return TritonDecoder.apply(top_indices, top_acts, W_dec)
    else:
        raise ImportError("Triton not installed, cannot use Triton implementation of SAE decoder. Use `torch` implementation instead.")
