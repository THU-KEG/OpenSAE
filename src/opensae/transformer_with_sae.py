import os
import sys
from pathlib import Path

import torch
import transformers

from transformers.utils import logging
from transformers import PreTrainedModel, PretrainedConfig

logging.set_verbosity_info()
logger = logging.get_logger("sae")

from .sae_utils import (
    PreTrainedSae,
    SaeEncoderOutput, 
    SaeDecoderOutput, 
    SaeForwardOutput
)
from .config_utils import PretrainedSaeConfig
from .saes.open_sae import OpenSaeConfig, OpenSae


class TransformerWithSae(torch.nn.Module):
    def __init__(
        self,
        transformer: transformers.PreTrainedModel | Path | str,
        sae: PreTrainedSae | Path | str,
        device: str | torch.device = "cpu"
    ):
        super().__init__()
        
        if isinstance(transformer, (Path, str)):
            self.transformer = transformers.AutoModelForCausalLM.from_pretrained(transformer)
        else:
            self.transformer = transformer
            
        if isinstance(sae, (Path, str)):
            self.sae = OpenSae.from_pretrained(sae)
        else:
            self.sae = sae
        self.config = self.sae.config
        
        self.device = device
        self.transformer.to(self.device)
        self.sae.to(self.device)
        
        self.token_indices  = None
        self.encoder_output = None
        
        self._register_input_hook()
        if self.config.output_hookpoint != self.config.input_hookpoint:
            self._register_output_hook()
            
        self.generate = self.transformer.generate


    def _input_hook_fn(
        self,
        module: torch.nn.Module,
        input: torch.Tensor | tuple[torch.Tensor],
        output: torch.Tensor | tuple[torch.Tensor]
    ):
        if isinstance(output, tuple):
            output_tensor = output[0]
        else:
            output_tensor = output
        _, _, hidden_size = output_tensor.size()
        
        # Extract By Indices
        if self.token_indices is None:
            sae_input = output_tensor   # Shape: (batch_size, seq_len, hidden_size)
        else:
            sae_input = output_tensor[self.token_indices]
        
        # We need to flatten the SAE input
        sae_input = sae_input.view(-1, hidden_size)
        
        # SAE Encoding
        self.encoder_output = self.sae.encode(sae_input)
        
        if self.config.output_hookpoint != self.config.input_hookpoint:
            return
        
        return self._output_hook_fn(module, input, output)


    def _output_hook_fn(
        self,
        module: torch.nn.Module,
        input: torch.Tensor | tuple[torch.Tensor],
        output: torch.Tensor | tuple[torch.Tensor]
    ):
        assert self.encoder_output is not None, "encoder_output is None"
        
        if isinstance(output, tuple):
            output_tensor = output[0]
        else:
            output_tensor = output
        
        output_tensor_dtype = output_tensor.dtype
        bsz, seq_len, hidden_size = output_tensor.size()
        
        sae_output = self.sae.decode(
            self.encoder_output.sparse_feature_indices,
            self.encoder_output.sparse_feature_activations,
            self.encoder_output.input_mean,
            self.encoder_output.input_std
        ).sae_output
        
        if self.token_indices is not None:
            # Insert the flatten SAE output back to the original output by indices
            reconstructed_output = output_tensor
            reconstructed_output[self.token_indices] = sae_output.view(-1, hidden_size)
        else:
            reconstructed_output = sae_output.view(bsz, seq_len, hidden_size)
        
        reconstructed_output = reconstructed_output.to(output_tensor_dtype)
        
        if isinstance(output, tuple):
            return_output_tuple = (reconstructed_output,) + output[1:]
            return return_output_tuple
        else:
            return reconstructed_output
            


    def _register_input_hook(self):
        input_hookpoint = self.config.input_hookpoint
        try:
            self.input_module = self.transformer.get_submodule(input_hookpoint)
        except:
            self.input_module = self.transformer.get_submodule("model." + input_hookpoint)
        self.input_module.register_forward_hook(self._input_hook_fn)


    def _register_output_hook(self):
        output_hookpoint = self.config.output_hookpoint
        try:
            self.output_module = self.transformer.get_submodule(output_hookpoint)
        except:
            self.output_module = self.transformer.get_submodule("model." + output_hookpoint)
        self.output_module.register_forward_hook(self._output_hook_fn)
        
    
    def forward(self, *inputs, **kwargs):
        return self.transformer(*inputs, **kwargs)