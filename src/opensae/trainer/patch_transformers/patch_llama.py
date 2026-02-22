import types

from flash_attn import flash_attn_func, flash_attn_varlen_func
from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input  # noqa

from typing import Callable, Optional, Union
import torch
import torch.nn.functional as F
from torch import nn
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache
from transformers.generation import GenerationMixin
from transformers.integrations import use_kernel_forward_from_hub
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_layers import GenericForQuestionAnswering, GenericForSequenceClassification, GenericForTokenClassification, GradientCheckpointingLayer
from transformers.modeling_outputs import MoeCausalLMOutputWithPast, MoeModelOutputWithPast
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs, auto_docstring, can_return_tuple
from transformers.utils.deprecation import deprecate_kwarg
from transformers.utils.generic import OutputRecorder, check_model_inputs
from transformers.models.qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs: Unpack[TransformersKwargs],
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights

def load_balancing_loss_func(
    gate_logits: Union[torch.Tensor, tuple[torch.Tensor], None],
    num_experts: Optional[int] = None,
    top_k=2,
    attention_mask: Optional[torch.Tensor] = None,
) -> Union[torch.Tensor, int]:
    r"""
    Computes auxiliary load balancing loss as in Switch Transformer - implemented in Pytorch.

    See Switch Transformer (https://huggingface.co/papers/2101.03961) for more details. This function implements the loss
    function presented in equations (4) - (6) of the paper. It aims at penalizing cases where the routing between
    experts is too unbalanced.

    Args:
        gate_logits:
            Logits from the `gate`, should be a tuple of model.config.num_hidden_layers tensors of
            shape [batch_size X sequence_length, num_experts].
        num_experts:
            Number of experts
        top_k:
            The number of experts to route per-token, can be also interpreted as the `top-k` routing
            parameter.
        attention_mask (`torch.Tensor`, *optional*):
            The attention_mask used in forward function
            shape [batch_size X sequence_length] if not None.

    Returns:
        The auxiliary loss.
    """
    if gate_logits is None or not isinstance(gate_logits, tuple):
        return 0

    if isinstance(gate_logits, tuple):
        compute_device = gate_logits[0].device
        concatenated_gate_logits = torch.cat([layer_gate.to(compute_device) for layer_gate in gate_logits], dim=0)

    routing_weights = torch.nn.functional.softmax(concatenated_gate_logits, dim=-1)

    _, selected_experts = torch.topk(routing_weights, top_k, dim=-1)

    expert_mask = torch.nn.functional.one_hot(selected_experts, num_experts)

    if attention_mask is None:
        # Compute the percentage of tokens routed to each experts
        tokens_per_expert = torch.mean(expert_mask.float(), dim=0)

        # Compute the average probability of routing to these experts
        router_prob_per_expert = torch.mean(routing_weights, dim=0)
    else:
        batch_size, sequence_length = attention_mask.shape
        num_hidden_layers = concatenated_gate_logits.shape[0] // (batch_size * sequence_length)

        # Compute the mask that masks all padding tokens as 0 with the same shape of expert_mask
        expert_attention_mask = (
            attention_mask[None, :, :, None, None]
            .expand((num_hidden_layers, batch_size, sequence_length, top_k, num_experts))
            .reshape(-1, top_k, num_experts)
            .to(compute_device)
        )

        # Compute the percentage of tokens routed to each experts
        tokens_per_expert = torch.sum(expert_mask.float() * expert_attention_mask, dim=0) / torch.sum(
            expert_attention_mask, dim=0
        )

        # Compute the mask that masks all padding tokens as 0 with the same shape of tokens_per_expert
        router_per_expert_attention_mask = (
            attention_mask[None, :, :, None]
            .expand((num_hidden_layers, batch_size, sequence_length, num_experts))
            .reshape(-1, num_experts)
            .to(compute_device)
        )

        # Compute the average probability of routing to these experts
        router_prob_per_expert = torch.sum(routing_weights * router_per_expert_attention_mask, dim=0) / torch.sum(
            router_per_expert_attention_mask, dim=0
        )

    overall_loss = torch.sum(tokens_per_expert * router_prob_per_expert.unsqueeze(0))
    return overall_loss * num_experts

# ----------------------- Modified LlamaAttention.forward -----------------------


def flash_attention_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: Optional[torch.Tensor],
    past_key_values: Optional[Cache] = None,
    cache_position: Optional[torch.LongTensor] = None,
    cu_seqlens: Optional[torch.LongTensor] = None,
    max_seqlens: Optional[torch.LongTensor] = None,
    **kwargs: Unpack[FlashAttentionKwargs],
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    #print("这里的隐藏层",hidden_states.shape)
    bsz, q_len, _ = hidden_states.size()
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_values is not None:
        # sin and cos are specific to RoPE models; cache_position needed for the static cache
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

    attention_interface: Callable = eager_attention_forward
    if self.config._attn_implementation != "eager":
        attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

    # TODO: These transpose are quite inefficient but Flash Attention requires the layout [batch_size, sequence_length, num_heads, head_dim]. We would need to refactor the KV cache
    # to be able to avoid many of these transpose/reshape/view.
    query_states = query_states.transpose(1, 2)
    key_states = key_states.transpose(1, 2)
    value_states = value_states.transpose(1, 2)
    attn_output = self._flash_attention_forward(
        query_states, key_states, value_states, attention_mask, q_len, 
        cu_seqlens=cu_seqlens, max_seqlens=max_seqlens, 
        dropout=0.0 if not self.training else self.attention_dropout
    )

    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, None




# ----------------------- Modified LlamaDecoderLayer.forward -----------------------


def decoder_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Cache] = None,
    cache_position: Optional[torch.LongTensor] = None,
    cu_seqlens: Optional[torch.LongTensor] = None,
    max_seqlens: Optional[torch.LongTensor] = None,
    **kwargs: Unpack[FlashAttentionKwargs],
) -> torch.FloatTensor:
    """
    Args:
        hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
        attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
            `(batch, sequence_length)` where padding elements are indicated by 0.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under
            returned tensors for more detail.
        output_router_logits (`bool`, *optional*):
            Whether or not to return the logits of all the routers. They are useful for computing the router loss,
            and should not be returned during inference.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
            (see `past_key_values`).
        past_key_values (`Cache`, *optional*): cached past key and value projection states
        cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
            Indices depicting the position of the input sequence tokens in the sequence.
        position_embeddings (`tuple[torch.FloatTensor, torch.FloatTensor]`, *optional*):
            Tuple containing the cosine and sine positional embeddings of shape `(batch_size, seq_len, head_dim)`,
            with `head_dim` being the embedding dimension of each attention head.
        kwargs (`dict`, *optional*):
            Arbitrary kwargs to be ignored, used for FSDP and other methods that injects code
            into the model
    """
    residual = hidden_states

    hidden_states = self.input_layernorm(hidden_states)

    # Self Attention
    hidden_states, _ = self.self_attn(
        hidden_states=hidden_states,
        position_embeddings=position_embeddings,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        cache_position=cache_position,
        cu_seqlens=cu_seqlens,
        max_seqlens=max_seqlens,
        **kwargs,
    )
    hidden_states = residual + hidden_states

    # Fully Connected
    residual = hidden_states
    hidden_states = self.post_attention_layernorm(hidden_states)
    hidden_states = self.mlp(hidden_states)
    # For the MoE layers, we need to unpack
    if isinstance(hidden_states, tuple):
        hidden_states, _ = hidden_states
    hidden_states = residual + hidden_states

    return hidden_states




# ----------------------- Modified LlamaModel.forward -----------------------



def model_forward(
    self,
    input_ids: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Cache] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
    cu_seqlens: Optional[torch.LongTensor] = None,
    max_seqlens: Optional[torch.LongTensor] = None,
    max_layer_num: Optional[int] = None,
    **kwargs: Unpack[TransformersKwargs],
) -> MoeModelOutputWithPast:
    if (input_ids is None) ^ (inputs_embeds is not None):
        raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

    if use_cache and past_key_values is None:
        past_key_values = DynamicCache(config=self.config)

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)

    if cache_position is None:
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        cache_position = torch.arange(
            past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
        )
    if position_ids is None:
        position_ids = cache_position.unsqueeze(0)

    mask_function = create_causal_mask if self.config.sliding_window is None else create_sliding_window_causal_mask
    causal_mask = mask_function(
        config=self.config,
        input_embeds=inputs_embeds,
        attention_mask=attention_mask,
        cache_position=cache_position,
        past_key_values=past_key_values,
        position_ids=position_ids,
    )

    hidden_states = inputs_embeds

    # create position embeddings to be shared across the decoder layers
    position_embeddings = self.rotary_emb(hidden_states, position_ids)

    return_legacy_cache = False
    layer_counter = 0
    for decoder_layer in self.layers[: self.config.num_hidden_layers]:
        if max_layer_num is not None and layer_counter >= max_layer_num + 1:
            break
        layer_counter += 1
        hidden_states = decoder_layer(
            hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=causal_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
                cu_seqlens=cu_seqlens,
                max_seqlens=max_seqlens,
            **kwargs,
        )

    hidden_states = self.norm(hidden_states)

    return MoeModelOutputWithPast(  # only diff with Mistral is the output type, we need MoE
        last_hidden_state=hidden_states,
        past_key_values=past_key_values,
    )


def _flash_attention_forward(
    self, query_states, key_states, value_states, attention_mask, query_length, 
    cu_seqlens = None, max_seqlens = None, dropout=0.0, softmax_scale=None
):
    # print("Patch LlamaFlashAttention2 _flash_attention_forward")
    # remove the "if" judgement logic
    causal = self.is_causal and query_length != 1

    # Contains at least one padding token in the sequence
    if attention_mask is not None:
        batch_size = query_states.shape[0]
        query_states, key_states, value_states, indices_q, cu_seq_lens, max_seq_lens = self._upad_input(
            query_states, key_states, value_states, attention_mask, query_length
        )

        cu_seqlens_q, cu_seqlens_k = cu_seq_lens
        max_seqlen_in_batch_q, max_seqlen_in_batch_k = max_seq_lens

        attn_output_unpad = flash_attn_varlen_func(
            query_states,
            key_states,
            value_states,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_in_batch_q,
            max_seqlen_k=max_seqlen_in_batch_k,
            dropout_p=dropout,
            softmax_scale=softmax_scale,
            causal=causal,
        )

        attn_output = pad_input(attn_output_unpad, indices_q, batch_size, query_length)
    # add here
    elif cu_seqlens is not None:
        assert query_states.size(0) == 1
        assert key_states.size(0) == 1
        assert value_states.size(0) == 1
        
        attn_output_unpad = flash_attn_varlen_func(
            query_states.squeeze(0),
            key_states.squeeze(0),
            value_states.squeeze(0),
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=max_seqlens,
            max_seqlen_k=max_seqlens,
            dropout_p=dropout,
            softmax_scale=softmax_scale,
            causal=causal,
        )
        attn_output = attn_output_unpad.unsqueeze(0)
    # end add here
    else:
        attn_output = flash_attn_func(
            query_states, key_states, value_states, dropout, softmax_scale=softmax_scale, causal=causal
        )

    # print(attn_output.size())
    return attn_output


def model_patch(model):

    model.forward = types.MethodType(model_forward, model)
    
    for l in model.layers:
        l.forward = types.MethodType(decoder_forward, l)
        l.self_attn.forward = types.MethodType(flash_attention_forward, l.self_attn)
        l.self_attn._flash_attention_forward = types.MethodType(_flash_attention_forward, l.self_attn)

    return model
