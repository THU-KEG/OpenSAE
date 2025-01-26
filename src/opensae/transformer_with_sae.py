import os
import sys
from pathlib import Path

import torch
import transformers

from transformers.utils import logging
from transformers import PreTrainedModel, PretrainedConfig, GenerationConfig

logging.set_verbosity_info()
logger = logging.get_logger("sae")

from .sae_utils import (
    PreTrainedSae,
    SaeEncoderOutput, 
    SaeDecoderOutput, 
    SaeForwardOutput
)
from .sae_utils import (
    extend_encoder_output,
    map_tokens_to_words
)
from .config_utils import PretrainedSaeConfig
from .saes.open_sae import OpenSaeConfig, OpenSae



class InterventionConfig(PretrainedConfig):
    def __init__(self, **kwargs):
        self.intervention = kwargs.pop("intervention", False)
        self.intervention_mode = kwargs.pop("intervention_mode", "set") # set, multiply, add
        assert self.intervention_mode in ["set", "multiply", "add"], "intervention_mode must be one of `set`, `multiply`, and `add`"

        self.intervention_indices = kwargs.pop("intervention_indices", None)
        if self.intervention:
            assert self.intervention_indices is not None, "intervention indices are not provided when set intervention to True"
        self.intervention_value = kwargs.pop("intervention_value", 0.0)

        self.prompt_only = kwargs.pop("prompt_only", False)



class TransformerWithSae(torch.nn.Module):
    def __init__(
        self,
        transformer: transformers.PreTrainedModel | Path | str,
        sae: PreTrainedSae | Path | str,
        device: str | torch.device = "cpu",
        intervention_config: InterventionConfig | None = None
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
        
        self.device = device
        self.transformer.to(self.device)
        self.sae.to(self.device)
        
        self.token_indices  = None
        self.encoder_output = None
        self.saved_features = None
        
        self.prefilling_stage = True
        
        self.forward_hook_handle = dict()
        self.backward_hook_handle = dict()
        self._register_input_hook()
        if self.config.output_hookpoint != self.config.input_hookpoint:
            self._register_output_hook()
        
        self.intervention_config = intervention_config
        if self.intervention_config is None:
            self.intervention_config = InterventionConfig()

    @property
    def config(self):
        return self.sae.config


    def clear_intermediates(self):
        self.token_indices  = None
        self.encoder_output = None
        self.saved_features = None
        self.prefilling_stage = True


    def _input_hook_fn(
        self,
        module: torch.nn.Module,
        input: torch.Tensor | tuple[torch.Tensor],
        output: torch.Tensor | tuple[torch.Tensor]
    ):
        if self.intervention_config.prompt_only and not self.prefilling_stage:
            return
        
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
        if self.saved_features is None:
            self.saved_features = self.encoder_output
        else:
            self.saved_features = extend_encoder_output(self.saved_features, self.encoder_output)
        
        if self.intervention_config.intervention:
            self._apply_intervention()
        
        if self.config.output_hookpoint != self.config.input_hookpoint:
            self.prefilling_stage = False
            return
        
        return self._output_hook_fn(module, input, output)


    def _output_hook_fn(
        self,
        module: torch.nn.Module,
        input: torch.Tensor | tuple[torch.Tensor],
        output: torch.Tensor | tuple[torch.Tensor]
    ):
        if self.intervention_config.prompt_only and not self.prefilling_stage:
            return

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
        
        self.prefilling_stage = False
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
        self.forward_hook_handle[input_hookpoint] = self.input_module.register_forward_hook(self._input_hook_fn)


    def _register_output_hook(self):
        output_hookpoint = self.config.output_hookpoint
        try:
            self.output_module = self.transformer.get_submodule(output_hookpoint)
        except:
            self.output_module = self.transformer.get_submodule("model." + output_hookpoint)
        self.backward_hook_handle[output_hookpoint] = self.output_module.register_forward_hook(self._output_hook_fn)


    def _apply_intervention(self):
        if self.intervention_config.intervention_mode == "multiply":
            self._apply_intervention_multiply()
        elif self.intervention_config.intervention_mode == "add" or self.intervention_config.intervention_mode == "set":
            self._apply_intervention_add_or_set()


    def _apply_intervention_multiply(self):
        for intervention_index in self.intervention_config.intervention_indices:
            mask = (self.encoder_output.sparse_feature_indices == intervention_index)
            self.encoder_output.sparse_feature_activations[mask] *= self.intervention_config.intervention_value


    def _apply_intervention_add_or_set(self):
        for intervention_index in self.intervention_config.intervention_indices:
            mask = (self.encoder_output.sparse_feature_indices == intervention_index)
            is_ind_activated = torch.any(mask, -1)
            if self.intervention_config.intervention_mode == "add":
                self.encoder_output.sparse_feature_activations[mask] += self.intervention_config.intervention_value
            elif self.intervention_config.intervention_mode == "set":
                self.encoder_output.sparse_feature_activations[mask] = self.intervention_config.intervention_value
            
            # In case that the index is not selected by the TopK
            # change the smallest value, which should be the least useful
            if not torch.all(is_ind_activated):
                min_val, min_ind = torch.min(self.encoder_output.sparse_feature_activations, -1)
                token_select = torch.arange(0, len(min_ind)).to(
                    dtype = torch.long,
                    device = self.encoder_output.sparse_feature_activations.device
                )
                
                set_val = min_val.clone()
                set_val[~is_ind_activated] = self.intervention_config.intervention_value
                
                set_ind = self.encoder_output.sparse_feature_indices[token_select, min_ind]
                set_ind[~is_ind_activated] = intervention_index
                
                self.encoder_output.sparse_feature_activations[token_select, min_ind] = set_val
                self.encoder_output.sparse_feature_indices[token_select, min_ind] = set_ind


    def update_intervention_config(self, intervention_config: InterventionConfig):
        self.intervention_config = intervention_config


    def forward(self, return_features = False, intervention_config = None, *inputs, **kwargs):
        self.clear_intermediates()
        if intervention_config is not None:
            self.update_intervention_config(intervention_config)
        forward_output = self.transformer(*inputs, **kwargs)
        if return_features:
            return (
                self.encoder_output,
                forward_output
            )
        else:
            return forward_output


    def generate(self, return_features = False, intervention_config = None, *inputs, **kwargs):
        self.clear_intermediates()
        if intervention_config is not None:
            self.update_intervention_config(intervention_config)
        
        generation = self.transformer.generate(*inputs, **kwargs)
        if return_features:
            return (
                self.saved_features,
                generation
            )
        else:
            return generation


    def visualize(self, *inputs, **kwargs):
        # TODO: To implement
        self.forward(*inputs, **kwargs)
        # visualize self.encoder_output


    def analyze(self, *inputs, **kwargs):
        # TODO: To implement
        self.forward(*inputs, **kwargs)
        # analyze self.encoder_output
