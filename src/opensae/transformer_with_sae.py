import sys
import numpy as np
import json
from pathlib import Path
from collections import Counter, defaultdict

import torch
import transformers
from transformers.utils import logging
from transformers import PreTrainedModel, PretrainedConfig, GenerationConfig

from .viz_utils import generate_html, save_html_file
from .extract_utils import refine_tokens

logger = logging.get_logger("sae")

from .sae_utils import (
    PreTrainedSae,
    SaeEncoderOutput, 
    SaeDecoderOutput, 
    SaeForwardOutput
)
from .config_utils import PretrainedSaeConfig
from .saes.open_sae import OpenSaeConfig, OpenSae


class InterventionConfig(PretrainedConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.intervention = kwargs.pop("intervention", False)
        self.intervention_mode = kwargs.pop("intervention_mode", "set") # set, multiply, add
        assert self.intervention_mode in ["set", "multiply", "add"], \
            "intervention_mode must be one of `set`, `multiply`, or `add`."

        self.intervention_indices = kwargs.pop("intervention_indices", None)
        if self.intervention:
            assert self.intervention_indices is not None, \
                "intervention indices are not provided when set intervention to True"
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

        self.config = self.sae.config
        self.device = device
        self.transformer.to(self.device)
        self.sae.to(self.device)
        
        self.token_indices = None
        self.encoder_output = None
        self.prefilling_stage = True
        
        self._register_input_hook()
        if self.config.output_hookpoint != self.config.input_hookpoint:
            self._register_output_hook()
        
        self.intervention_config = intervention_config
        if self.intervention_config is None:
            self.intervention_config = InterventionConfig()


    def clear_intermediates(self):
        self.token_indices = None
        self.encoder_output = None
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
        
        if self.token_indices is None:
            sae_input = output_tensor
        else:
            sae_input = output_tensor[self.token_indices]
        
        sae_input = sae_input.view(-1, hidden_size)
        
        self.encoder_output = self.sae.encode(sae_input)
        
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


    def _apply_intervention(self):
        mode = self.intervention_config.intervention_mode
        if mode == "multiply":
            self._apply_intervention_multiply()
        elif mode in ["add", "set"]:
            self._apply_intervention_add_or_set()


    def _apply_intervention_multiply(self):
        for intervention_index in self.intervention_config.intervention_indices:
            mask = (self.encoder_output.sparse_feature_indices == intervention_index)
            self.encoder_output.sparse_feature_activations[mask] *= self.intervention_config.intervention_value


    def _apply_intervention_add_or_set(self):
        for intervention_index in self.intervention_config.intervention_indices:
            mask = (self.encoder_output.sparse_feature_indices == intervention_index)
            is_ind_activated = torch.any(mask, dim=-1)

            if self.intervention_config.intervention_mode == "add":
                self.encoder_output.sparse_feature_activations[mask] += self.intervention_config.intervention_value
            elif self.intervention_config.intervention_mode == "set":
                self.encoder_output.sparse_feature_activations[mask] = self.intervention_config.intervention_value

            if not torch.all(is_ind_activated):
                min_val, min_ind = torch.min(self.encoder_output.sparse_feature_activations, dim=-1)
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


    def update_intervention_config(self, intervention_config: InterventionConfig):
        self.intervention_config = intervention_config


    def forward(self, return_features = False, *inputs, **kwargs):
        self.clear_intermediates()
        forward_output = self.transformer(*inputs, **kwargs)
        if return_features:
            return (
                self.encoder_output,
                forward_output
            )
        else:
            return forward_output


    def generate(self, *inputs, **kwargs):
        self.clear_intermediates()
        return self.transformer.generate(*inputs, **kwargs)

    def extract_data(self, encodings, tokenizer):
        input_ids = encodings["input_ids"].to(self.device)
        attention_mask = encodings["attention_mask"].to(self.device)

        self.forward(input_ids=input_ids, attention_mask=attention_mask)

        if self.encoder_output is None:
            print("[extract_data] self.encoder_output is None after forward.")
            return []

        all_indices = self.encoder_output.sparse_feature_indices  
        all_acts = self.encoder_output.sparse_feature_activations 

        batch_size, max_len = input_ids.shape
        lengths = attention_mask.sum(dim=1).tolist()

        sentences = []
        for i in range(batch_size):
            decoded_sentence = tokenizer.decode(
                input_ids[i], 
            )
            sentences.append(decoded_sentence)

        token_word_map_list = []
        for i in range(batch_size):
            sentence = sentences[i]
            map_result = refine_tokens(sentence, tokenizer)
            token_word_map_list.append(map_result)     
        filter_tokens = {"<|begin_of_text|>",}

        structured_data = []
        offset = 0

        for i in range(batch_size): 
            tokens_list = []

            offset_idx = 0
            for t_idx in range(max_len):
                if attention_mask[i, t_idx].item() == 0:
                    offset_idx += 1  
                    continue

                token_str = token_word_map_list[i]["tokens"][t_idx]
                if token_str in filter_tokens:
                    offset_idx += 1 
                    continue

                base_indices = all_indices[offset + offset_idx].tolist()
                base_acts = all_acts[offset + offset_idx].tolist()
                offset_idx += 1

                merged_activations = [
                    {
                        "base_vector": int(b_idx),
                        "activation": float(a_val),
                    }
                    for (b_idx, a_val) in zip(base_indices, base_acts)
                ]

                tokens_list.append({
                    "token": token_str,
                    "activations": merged_activations
                })

            offset += max_len


            structured_data.append({
                "sentence_id": i,
                "tokens": tokens_list
            })

        return structured_data


    def analyze_data(
        self,
        structured_data: list,
        mode: str = "frequency",
        top_k: int = 10,
        token_index: int = 0,
        noise_bases: set = None
    ):

        if noise_bases is None:
            noise_bases = set()

        if not structured_data:
            print("[analyze_data] structured_data is empty.")
            return {}

        if mode == "frequency":
            freq_counter = Counter()
            total_sentences = len(structured_data)

            for sent_info in structured_data:
                base_ids_in_sentence = set()
                for token_info in sent_info["tokens"]:
                    for act_dict in token_info["activations"]:
                        b_id = act_dict["base_vector"]
                        val = act_dict["activation"]
                        if b_id not in noise_bases and val != 0.0:
                            base_ids_in_sentence.add(b_id)
                for b_id in base_ids_in_sentence:
                    freq_counter[b_id] += 1

            sorted_freq = freq_counter.most_common()
            results = []
            for (b_id, freq) in sorted_freq:
                ratio = (freq / total_sentences) * 100
                results.append((b_id, freq, ratio))

            top_k_results = results[:top_k]
            return {
                "mode": "frequency",
                "total_sentences": total_sentences,
                "all_sorted": results,
                "top_k_results": top_k_results
            }

        elif mode == "position":
            freq_counter = Counter()
            total_sentences = len(structured_data)

            for sent_info in structured_data:
                tokens = sent_info["tokens"]
                if token_index < len(tokens):
                    token_info = tokens[token_index]
                    base_ids_in_token = set()
                    for act_dict in token_info["activations"]:
                        b_id = act_dict["base_vector"]
                        val = act_dict["activation"]
                        if b_id not in noise_bases and val != 0.0:
                            base_ids_in_token.add(b_id)
                    for b_id in base_ids_in_token:
                        freq_counter[b_id] += 1

            sorted_freq = freq_counter.most_common()
            results = []
            for (b_id, freq) in sorted_freq:
                ratio = (freq / total_sentences) * 100
                results.append((b_id, freq, ratio))

            top_k_results = results[:top_k]
            return {
                "mode": "position",
                "token_index": token_index,
                "total_sentences": total_sentences,
                "all_sorted": results,
                "top_k_results": top_k_results
            }

        elif mode == "activation":
            global_sum = defaultdict(float)
            global_count = defaultdict(int)

            for sent_info in structured_data:
                sentence_sum = defaultdict(float)
                sentence_count = defaultdict(int)

                for token_info in sent_info["tokens"]:
                    for act_dict in token_info["activations"]:
                        b_id = act_dict["base_vector"]
                        val = act_dict["activation"]
                        if b_id in noise_bases:
                            continue
                        if val != 0.0:
                            sentence_sum[b_id] += val
                            sentence_count[b_id] += 1

                for b_id, sum_val in sentence_sum.items():
                    cnt = sentence_count[b_id]
                    avg_val = sum_val / cnt
                    global_sum[b_id] += avg_val
                    global_count[b_id] += 1

            if not global_sum:
                return {
                    "mode": "activation",
                    "message": "No valid tokens or all noise?"
                }

            base_to_global_avg = {}
            for b_id, s_val in global_sum.items():
                c_val = global_count[b_id]
                final_avg = s_val / float(c_val)
                base_to_global_avg[b_id] = final_avg

            sorted_bases = sorted(
                base_to_global_avg.items(),
                key=lambda x: x[1],
                reverse=True
            )
            top_k_bases = sorted_bases[:top_k]

            return {
                "mode": "activation",
                "global_top_k": top_k_bases,
                "all_sorted": sorted_bases
            }

        else:
            print(f"[analyze_data] unknown mode: {mode}")
            return {}

    def visualize(self, activations, base_vector_indices, output_html: str):
        html_content = generate_html(activations, base_vector_indices)

        save_html_file(html_content, output_html)
        print(f"[visualize] Visualization saved to '{output_html}'")
