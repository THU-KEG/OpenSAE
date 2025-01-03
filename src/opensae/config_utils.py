import os
import sys

from abc import abstractmethod

import torch

import transformers
from transformers.utils import logging
from transformers.configuration_utils import PretrainedConfig


class PretrainedSaeConfig(PretrainedConfig):
    def __init__(
        self,
        hidden_size: int,
        feature_size: int,
        input_normalize: bool,
        input_hookpoint: str,
        output_hookpoint: str,
        model_name: str,
        activation: str,
        dtype: torch.dtype | None,
        **kwargs
    ):
        self.hidden_size = hidden_size
        self.feature_size = feature_size
        
        self.input_normalize = input_normalize
        
        self.input_hookpoint = input_hookpoint
        self.output_hookpoint = output_hookpoint
        
        self.model_name = model_name
        
        self.activation = activation
        assert self.activation in [
            "topk",
            "jumprelu",
        ]
        
        self.dtype = dtype
        if self.dtype is None:
            self.dtype = torch.bfloat16
            logging.warning_advice(f"dtype is not provided, defaulting to {self.dtype}")
        
        super().__init__(**kwargs)


        
__all__ = [
    "PretrainedSaeConfig"
]