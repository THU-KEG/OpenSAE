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
    SaeForwardOutput,
    torch_decode,
    triton_decode
)
from ...sparse_activations import (
    TopK,
    JumpReLU
)
from .configuration_open_sae import OpenSaeConfig


class PreTrainedOpenSae(PreTrainedSae):
    """
    Interface class to set config class and weight initialization
    """
    is_parallelizable = False
    config_class = OpenSaeConfig
    
    def _init_weights(self, module: torch.nn.Module):
        """Initialize the weights."""
        return

class OpenSae(PreTrainedOpenSae):
    def __init__(
        self, 
        config: OpenSaeConfig,
        device: str | torch.device = None,
        decoder: bool = True,
        **kwargs
    ):
        super().__init__(config, **kwargs)
        
        device = torch.device("cpu") if device is None else torch.device(device)
        self.decoder = decoder
        
        self.encoder = torch.nn.Linear(
            in_features = self.config.hidden_size, 
            out_features = self.config.feature_size, 
            device = device, 
            dtype = self.config.get_torch_dtype()
        )
        self.encoder.bias.data.zero_()

        self.W_dec = torch.nn.Parameter(self.encoder.weight.data.clone()) if self.decoder else None
        if self.decoder and self.config.normalize_decoder:
            self.set_decoder_norm_to_unit_norm()
        self.b_dec = torch.nn.Parameter(
            torch.zeros(
                self.config.hidden_size,
                dtype = self.config.get_torch_dtype(), 
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

        if self.config.decoder_impl == "triton":
            self.decode_fn = triton_decode
        elif self.config.decoder_impl == "torch":
            self.decode_fn = torch_decode

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

    def pre_process(self, hidden: Tensor) -> Tensor:
        if self.config.input_normalize:
            hidden, mu, std = self.normalization(hidden, self.config.input_normalize_eps)
        
        if not self.config.normalize_shift_back:
            mu, std = None, None
        
        # Remove decoder bias as per Anthropic
        return hidden.to(self.b_dec.dtype) - self.b_dec, mu, std


    def encode(self, hidden: Tensor, return_all_features: bool = False) -> SaeEncoderOutput:
        sae_input, input_mean, input_std = self.pre_process(hidden)
        all_features = self.encoder(sae_input)
        # Remove negative features
        all_features = torch.nn.functional.relu(all_features)
        
        feature_activation, feature_indices = self.sparse_activation(all_features)
        
        return SaeEncoderOutput(
            sparse_feature_activations = feature_activation,
            sparse_feature_indices = feature_indices,
            all_features = all_features if return_all_features else None,
            input_mean = input_mean if self.config.input_normalize else None,
            input_std = input_std if self.config.input_normalize else None
        )


    def decode(
        self, 
        feature_indices: Tensor, 
        feature_activation: Tensor,
        input_mean: Tensor | None = None,
        input_std: Tensor | None = None
    ) -> SaeDecoderOutput:
        assert self.W_dec is not None, "Decoder weight was not initialized."
        if self.config.normalize_shift_back:
            assert input_mean is not None and input_std is not None, "Input mean and std must be provided for shift back normalization."            

        with torch.cuda.device(self.W_dec.device.index):
            reconstruction = self.decode_fn(
                feature_indices,
                feature_activation.to(torch.float32),
                self.W_dec.mT.to(torch.float32)
            )
        reconstruction = reconstruction + self.b_dec
        
        if self.config.normalize_shift_back:
            reconstruction = reconstruction * (input_std + self.config.input_normalize_eps) + input_mean

        return SaeDecoderOutput(sae_output = reconstruction)


    def reconstruction_loss(
        self,
        hidden: Tensor,
        hidden_variance: Tensor,
        sae_output: Tensor
    ) -> Tensor:
        reconstruction_error = sae_output - hidden
        dimensional_l2_loss = reconstruction_error.pow(2).sum(0)    # size = (hidden_size,), per-dimensional L2 loss
        normalized_l2_loss = dimensional_l2_loss / hidden_variance  # putting everything on a reasonable scale
        reconstruction_loss = torch.mean(normalized_l2_loss)
        l2_loss = dimensional_l2_loss.mean()
        
        return (
            reconstruction_error,
            l2_loss,
            reconstruction_loss,
        )
        
    
    def auxk_loss(
        self, 
        hidden: Tensor, 
        sae_output: Tensor,
        reconstruction_error: Tensor,
        hidden_variance: Tensor, 
        dead_mask: Tensor,
        all_features: Tensor,
        input_mean: Tensor | None = None,
        input_std: Tensor | None = None
    ) -> Tensor:
        assert dead_mask is not None, "Dead mask is not provided."
        num_dead = int(dead_mask.sum())
        if num_dead == 0:
            return sae_output.new_tensor(0.0)


        # Heuristic from Appendix B.1 in the paper
        k_aux = hidden.shape[-1] // 2

        # Reduce the scale of the loss if there are a small number of dead latents
        scale = min(num_dead / k_aux, 1.0)
        k_aux = min(k_aux, num_dead)

        # Don't include living latents in this loss
        auxk_all_features = torch.where(dead_mask[None], all_features, -torch.inf)

        # Top-k dead latents
        auxk_feature_activations, auxk_feature_indices = auxk_all_features.topk(k_aux, sorted=False)

        # Encourage the top ~50% of dead latents to predict the residual of the
        # top k living latents
        auxk_sae_decoder_output = self.decode(
            auxk_feature_indices, 
            auxk_feature_activations,
            input_mean, input_std
        ).sae_output
        auxk_loss = (auxk_sae_decoder_output - reconstruction_error).pow(2).sum(0)
        auxk_loss = scale * torch.mean(auxk_loss / hidden_variance)
        
        return auxk_loss


    def forward(
        self, 
        hidden: Tensor, 
        dead_mask: Tensor | None = None
    ) -> SaeForwardOutput:
        # 1. SAE computation: hidden --> [encode] --> features --> [decode] --> reconstruction
        sae_encoder_output = self.encode(hidden, return_all_features = self.config.multi_topk)
        sae_decoder_output = self.decode(
            sae_encoder_output.sparse_feature_indices, 
            sae_encoder_output.sparse_feature_activations,
            sae_encoder_output.input_mean,
            sae_encoder_output.input_std
        ).sae_output
        assert sae_decoder_output.shape == hidden.shape, f"Output shape {sae_decoder_output.shape} does not match input shape {hidden.shape}"
        
        
        # 2. Prepare per-dimensional variance to make the training-loss stable
        per_dimension_variance = (hidden - hidden.mean(0)).pow(2).sum(0)       # size = (hidden_size,)
        per_dimension_variance = torch.clamp(per_dimension_variance, min=1.0)  # clip to ensure total_variance < 5.0
        
        
        # 3. Compute losses
        # 3.1. Reconstruction loss
        reconstruction_error, l2_loss, reconstruction_loss = self.reconstruction_loss(
            hidden = hidden, 
            hidden_variance =  per_dimension_variance, 
            sae_output = sae_decoder_output
        )
        
        # 3.2. AuxK loss: help to reduce dead features.
        # INVOKE Extra decoder pass for AuxK loss
        if self.config.auxk_alpha > 1e-6 and dead_mask is not None and (num_dead := int(dead_mask.sum())) > 0:
            auxk_loss = self.auxk_loss(
                hidden = hidden,
                sae_output = sae_decoder_output,
                reconstruction_error = reconstruction_error,
                hidden_variance = per_dimension_variance,
                dead_mask = dead_mask,
                all_features = sae_encoder_output.all_features,
                input_mean = sae_encoder_output.input_mean,
                input_std = sae_encoder_output.input_std
            )
        else:
            auxk_loss = sae_decoder_output.new_tensor(0.0)

        # 3.3. Multi-TopK loss: help to reduce overfitting to k
        # INVOKE Extra decoder pass for multi-topk loss
        if self.config.multi_topk:
            multi_topk_feature_activations, multi_topk_feature_indices = self.multi_topk(sae_encoder_output.all_features)
            multi_topk_sae_decoder_output = self.decode(
                multi_topk_feature_indices, multi_topk_feature_activations,
                sae_encoder_output.input_mean, sae_encoder_output.input_std
            ).sae_output

            _, _, multi_topk_loss = self.reconstruction_loss(
                hidden = hidden, 
                hidden_variance = per_dimension_variance, 
                sae_output = multi_topk_sae_decoder_output
            )
        else:
            multi_topk_loss = sae_decoder_output.new_tensor(0.0)


        # 3.4. L1 loss
        l1_loss = torch.tensor(0.0, device=hidden.device)
        if self.config.l1_coef is not None and self.cfg.l1_coef > 1e-8:
            l1_loss = torch.norm(sae_encoder_output.all_features, p=1, dim=-1).mean() * self.cfg.l1_coef

        final_loss = reconstruction_loss + multi_topk_loss / 8 + auxk_loss * self.config.auxk_alpha
        if l1_loss > 1e-8:
            final_loss += l1_loss

        return SaeForwardOutput(
            # Encoder Outputs
            sparse_feature_activations = sae_encoder_output.sparse_feature_activations,
            sparse_feature_indices = sae_encoder_output.sparse_feature_indices,
            all_features = sae_encoder_output.all_features,
            # Decoder Outputs
            sae_output = sae_decoder_output,
            # Loss Outputs
            reconstruction_loss = reconstruction_loss,
            multi_topk_loss = multi_topk_loss,
            auxk_loss = auxk_loss,
            l1_loss = l1_loss,
            loss = final_loss,
        )