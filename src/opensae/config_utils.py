import os
import sys

from abc import abstractmethod

import torch

import transformers
from transformers.utils import logging
from transformers.configuration_utils import PretrainedConfig

logging.set_verbosity_info()
logger = logging.get_logger("sae")


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
        torch_dtype: torch.dtype | None,
        **kwargs
    ):
        super().__init__(**kwargs)
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

        if torch_dtype is None:
            self.torch_dtype = "bfloat16"
            logger.warning_advice(f"dtype is not provided, defaulting to {self.torch_dtype}")
        else:
            self.torch_dtype = torch_dtype


    def get_torch_dtype(self) -> torch.dtype:
        if not isinstance(self.torch_dtype, torch.dtype):
            dtype = getattr(torch, self.torch_dtype)
            assert isinstance(dtype, torch.dtype)
        
        return dtype

        
__all__ = [
    "PretrainedSaeConfig"
]