import os
import sys

import torch
from torch import Tensor

from dataclasses import dataclass
from abc import abstractmethod

from transformers.modeling_utils import PreTrainedModel
from transformers.utils import ModelOutput


@dataclass
class SaeEncoderOutput(ModelOutput):
    latent_indices: Tensor
    latent_acts: Tensor


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
