import os
import sys

import torch
from torch import Tensor
import einops

import transformers

from ...sae_utils import (
    PreTrainedSae, 
    SaeEncoderOutput, 
    SaeDecoderOutput, 
    SaeForwardOutput
)
from ...sparse_activation import (
    TopK,
    JumpReLU
)
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
        self.decoder = decoder

        self.encoder = torch.nn.Linear(
            in_features = self.config.hidden_size, 
            out_features = self.config.feature_size, 
            device = device, 
            dtype = self.config.dtype
        )
        self.encoder.bias.data.zero_()

        self.W_dec = torch.nn.Parameter(self.encoder.weight.data.clone()) if self.decoder else None
        if self.decoder and self.config.normalize_decoder:
            self.set_decoder_norm_to_unit_norm()
        self.b_dec = torch.nn.Parameter(
            torch.zeros(
                size = self.config.hidden_size,
                dtype = self.config.dtype, 
                device = device
            )
        )
        
        self.sparse_activation = None
        if self.config.activation == "topk":
            self.sparse_activation = TopK(k = self.config.k)
            if self.config.multi_topk:
                self.multi_topk = TopK(k = self.config.k * self.config.multi_topk)

        elif self.config.activation == "jumprelu":
            self.sparse_activation = JumpReLU(theta = self.config.jumprelu_theta)

    @torch.no_grad()
    def set_decoder_norm_to_unit_norm(self):
        assert self.W_dec is not None, "Decoder weight was not initialized."

        eps = torch.finfo(self.W_dec.dtype).eps
        norm = torch.norm(self.W_dec.data, dim=1, keepdim=True)
        self.W_dec.data /= norm + eps

    @torch.no_grad()
    def remove_gradient_parallel_to_decoder_directions(self):
        assert self.W_dec is not None, "Decoder weight was not initialized."
        assert self.W_dec.grad is not None  # keep pyright happy

        parallel_component = einops.einsum(
            self.W_dec.grad,
            self.W_dec.data,
            "d_sae d_in, d_sae d_in -> d_sae",
        )
        self.W_dec.grad -= einops.einsum(
            parallel_component,
            self.W_dec.data,
            "d_sae, d_sae d_in -> d_sae d_in",
        )
        
    def normalization(self, x: Tensor, eps: float = 1e-5) -> Tensor:
        mu = x.mean(dim=-1, keepdim=True)
        x = x - mu
        std = x.std(dim=-1, keepdim=True)
        x = x / (std + eps)
        return x, mu, std

    def encode(self, **kwargs) -> SaeEncoderOutput:
        pass
    
    def decode(self, **kwargs) -> SaeDecoderOutput:
        pass
    
    def forward(self, **kwargs) -> SaeForwardOutput:
        pass