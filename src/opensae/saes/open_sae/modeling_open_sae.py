import os
import sys

import torch
import transformers

from ...sae_utils import PreTrainedSae, SaeEncoderOutput, SaeDecoderOutput, SaeForwardOutput
from .configuration_open_sae import PretrainedSaeConfig


class OpenSae(PreTrainedSae):
    def __init__(
        self, 
        config: PretrainedSaeConfig,
        device: str | torch.device = None,
        decoder: bool = True,
        **kwargs
    ):
        super().__init__(config, **kwargs)
        
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def encode(self, **kwargs):
        pass
    
    def decode(self, **kwargs):
        pass
    
    def forward(self, **kwargs):
        pass