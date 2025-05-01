# -*- coding:utf-8 -*-

from typing import List, Optional, Tuple, Union

from torch import nn
import math
from transformers.models.llama.modeling_llama import rotate_half, repeat_kv
import torch
import transformers
from transformers.cache_utils import Cache
import pdb
import math
from transformers.modeling_outputs import BaseModelOutputWithPast, MoeCausalLMOutputWithPast
from transformers.modeling_outputs import CausalLMOutputWithPast
from flash_attn.losses.cross_entropy import CrossEntropyLoss
from transformers import LlamaConfig, PretrainedConfig
from flash_attn import flash_attn_with_kvcache, flash_attn_func, flash_attn_varlen_func
import flash_attn_2_cuda as flash_attn_cuda
from functools import partial
import math
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS




class RotaryEmbedding(nn.Module):
    def __init__(
        self,
        dim=None,
        max_position_embeddings=2048,
        base=10000,
        device=None,
        scaling_factor=1.0,
        rope_type="default",
        config: Optional[LlamaConfig] = None,
    ):
        super().__init__()
        # TODO (joao): remove the `if` below, only used for BC
        self.rope_kwargs = {}
        if config is None:
            self.rope_kwargs = {
                "rope_type": rope_type,
                "factor": scaling_factor,
                "dim": dim,
                "base": base,
                "max_position_embeddings": max_position_embeddings,
            }
            self.rope_type = rope_type
            self.max_seq_len_cached = max_position_embeddings
            self.original_max_seq_len = max_position_embeddings
        else:
            # BC: "rope_type" was originally "type"
            if config.rope_scaling is not None:
                self.rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
            else:
                self.rope_type = "default"
            self.max_seq_len_cached = config.max_position_embeddings
            self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device, **self.rope_kwargs)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    
    @torch.no_grad()
    def forward(self, x, position_ids):

        # Core RoPE block
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        # Force float32 (see https://github.com/huggingface/transformers/pull/29285)
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()

        # Advanced RoPE types (e.g. yarn) apply a post-processing scaling factor, equivalent to scaling attention
        cos = cos * self.attention_scaling
        sin = sin * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


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
    q_embed = (q * cos) + (rotate_half(q) * sin) if not q is None else None
    k_embed = (k * cos) + (rotate_half(k) * sin) if not k is None else None
    return q_embed, k_embed

def apply_rotary_pos_emb_by_heads(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
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
    q_embed = (q * cos) + (rotate_half(q) * sin) if not q is None else None
    k_embed = (k * cos) + (rotate_half(k) * sin) if not k is None else None
    return q_embed, k_embed


def atten_merge_forward(
    query_position,
    window_size,
    normal_query_states,
    normal_key_states,
    scale_query_states,
    scale_key_states,
    value_states,
    bsz,
    kv_seq_len,
    attn_dropout,
):
    def convert_lse_right_to_left(padded_lse, seq_lens):
        lse = torch.zeros_like(padded_lse)
        for idx in range(bsz):
            L = seq_lens[idx].item()
            if L > 0:
                lse[idx, :, -L:] = padded_lse[idx, :, :L]
        return lse

    if query_position.max() < window_size:
        return flash_attn_func(
            normal_query_states,
            normal_key_states,
            value_states,
            dropout_p=attn_dropout,
            causal=True,
            window_size=[-1, -1],
        )

    normal_attn_output, normal_lse_pad, _ = flash_attn_func(
        normal_query_states,
        normal_key_states,
        value_states,
        dropout_p=attn_dropout,
        causal=True,
        window_size=[window_size - 1, 0],
        return_attn_probs=True,
    )

    # Step 2: Scale attention (global)
    scale_len = kv_seq_len - window_size
    scale_q = scale_query_states[:, -scale_len:]
    scale_k = scale_key_states[:, :scale_len]
    scale_v = value_states[:, :scale_len]
    # scale_mask = attention_mask[:, :scale_len] if attention_mask is not None else None
    

    scale_attn_output, scale_lse_pad, _ = flash_attn_func(
        scale_q,
        scale_k,
        scale_v,
        dropout_p=attn_dropout,
        causal=True,
        window_size=[-1, -1],
        return_attn_probs=True,
    )

    # Step 3: Convert softmax LSE from left-align to right-align
    normal_len = torch.full((bsz, 1), kv_seq_len, dtype=torch.long)
    scale_len_tensor = torch.full((bsz, 1), scale_len, dtype=torch.long)

    normal_lse = convert_lse_right_to_left(normal_lse_pad, normal_len)
    scale_lse = convert_lse_right_to_left(scale_lse_pad, scale_len_tensor)

    # Step 4: Fuse outputs using LSE (log-sum-exp)
    normal_lse = normal_lse.transpose(1, 2).unsqueeze(-1)
    scale_lse = scale_lse.transpose(1, 2).unsqueeze(-1)

    lse_gap = scale_lse - normal_lse[:, -scale_lse.shape[1]:]
    normal_lse[:, -scale_lse.shape[1]:] = 1 / (1 + torch.exp(lse_gap))
    normal_lse[:, :-scale_lse.shape[1]] = 1.
    scale_lse = 1 / (1 + torch.exp(-lse_gap))

    normal_attn_output[:, -normal_lse.shape[1]:] *= normal_lse
    scale_attn_output[:, -scale_lse.shape[1]:] *= scale_lse

    attn_output = torch.empty_like(normal_attn_output).copy_(normal_attn_output)
    attn_output[:, window_size - kv_seq_len:] += scale_attn_output
    attn_output = torch.nan_to_num(attn_output, nan=0)
    return attn_output


def concat_tensors(tensor_list, dim_range, half_head_dim):
    segments = []
    # First half
    for i in range(len(tensor_list)):
        segments.append(tensor_list[i][:,:,dim_range[i]:dim_range[i+1]])
    # Second half
    for i in range(len(tensor_list)):
        segments.append(tensor_list[i][:,:,half_head_dim+dim_range[i]:half_head_dim+dim_range[i+1]])
    return torch.cat(segments, dim=-1)

def dpe_flash_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    """
        Require updating tansformers to >= 4.40.0, flash_attn >= 2.5.6
        a. Only support causal mask.
        b. Don't support atttention_mask.
        c. Only support batch size = 1.
        d. Only support q_len = 1 or q_len = seq_len.
    """

    bsz, q_len, _ = hidden_states.size()
    scale_factors = dpe_scale_factors
    window_size = dpe_local_window_size
    dim_groups_range = dpe_dim_groups_range
    select_topk_dim = dpe_select_topk_dim
    half_head_dim = self.head_dim // 2


    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    
    past_key_value = getattr(self, "past_key_value", past_key_value)
    if past_key_value is not None:
        # sin and cos are specific to RoPE models; position_ids needed for the static cache
        cache_kwargs = {"cache_position": cache_position}
        key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
    kv_seq_len = key_states.shape[-2]

    query_position = position_ids
    key_position = position_ids if q_len != 1 else torch.arange(kv_seq_len, dtype=position_ids.dtype).to(query_position.device).view(1, kv_seq_len) 
    attn_dropout = self.config.attention_dropout if self.training else 0.0
    if q_len == 1:
        normal_key_position = position_ids[:, -1] - key_position
        _re_window_size = 0 if position_ids.max() < window_size else window_size
        decode_k_cos_list, decode_k_sin_list = [], []
        for i in scale_factors:
            scale_key_position_i = position_ids[:, -1]//i - key_position//i + (_re_window_size - _re_window_size//i)
            decode_key_position_i = torch.cat([scale_key_position_i[:, :-window_size], normal_key_position[:,-window_size:]], dim=1)
            decode_k_cos_i, decode_k_sin_i = self.rotary_emb(value_states, decode_key_position_i)
            decode_k_cos_list.append(decode_k_cos_i)
            decode_k_sin_list.append(decode_k_sin_i)

        decode_k_cos = concat_tensors(decode_k_cos_list, dim_groups_range, half_head_dim)
        decode_k_sin = concat_tensors(decode_k_sin_list, dim_groups_range, half_head_dim)
        if select_topk_dim:
            dim_mask = dpe_dim_mask[self.layer_idx]
            scale_key_position_all = position_ids[:, -1] - key_position
            decode_key_position_all = torch.cat([scale_key_position_all[:, :-window_size], normal_key_position[:,-window_size:]], dim=1)
            decode_k_cos_all, decode_k_sin_all = self.rotary_emb(value_states, decode_key_position_all)#, seq_len=None)
            dim_mask = dim_mask.unsqueeze(2).expand(-1, -1, value_states.size(2), -1)
            decode_k_cos = torch.where(dim_mask, decode_k_cos, decode_k_cos_all)
            decode_k_sin = torch.where(dim_mask, decode_k_sin, decode_k_sin_all)
            decode_query_states = query_states.transpose(1,2).contiguous() # position 0: cos 0 = 1, sin 0 = 0
            key_states = repeat_kv(key_states, self.num_key_value_groups)
            decode_value_states = repeat_kv(value_states, self.num_key_value_groups).transpose(1, 2).contiguous()
            _, decode_key_states = apply_rotary_pos_emb_by_heads(None, key_states, decode_k_cos, -decode_k_sin)
            decode_key_states = decode_key_states.transpose(1, 2).contiguous()
        else: 
            decode_query_states = query_states.transpose(1,2).contiguous() # position 0: cos 0 = 1, sin 0 = 0
            _, decode_key_states = apply_rotary_pos_emb(None, key_states, decode_k_cos, -decode_k_sin) 

            decode_key_states = decode_key_states.transpose(1, 2).contiguous()
            decode_value_states = value_states.transpose(1, 2).contiguous()
        
        attn_output = flash_attn_func(decode_query_states,
                                      decode_key_states,
                                      decode_value_states,
                                      attn_dropout, 
                                      softmax_scale=None, 
                                      causal=True)
    elif q_len == kv_seq_len:
        # set correct position_ids & apply RoPE.
        normal_cos, normal_sin = self.rotary_emb(value_states, query_position)

        _re_window_size = 0 if query_position.max() < window_size else window_size # in case that, the smallest q position, g2-g2//g1 exceed the max position


        scale_query_position, scale_key_position = [], []

        for i in scale_factors:
            scale_query_position.append(query_position // i + _re_window_size - _re_window_size / i)
            scale_key_position.append(key_position // i)


        scale_q_cos_list, scale_q_sin_list = [], []
        scale_k_cos_list, scale_k_sin_list = [], []
        for i in range(len(scale_factors)):
            scale_q_cos_i, scale_q_sin_i = self.rotary_emb(value_states, scale_query_position[i])
            scale_q_cos_list.append(scale_q_cos_i)
            scale_q_sin_list.append(scale_q_sin_i)
            scale_k_cos_i, scale_k_sin_i = self.rotary_emb(value_states, scale_key_position[i])
            scale_k_cos_list.append(scale_k_cos_i)
            scale_k_sin_list.append(scale_k_sin_i)

        scale_q_cos = concat_tensors(scale_q_cos_list, dim_groups_range, half_head_dim)
        scale_q_sin = concat_tensors(scale_q_sin_list, dim_groups_range, half_head_dim)
        scale_k_cos = concat_tensors(scale_k_cos_list, dim_groups_range, half_head_dim)
        scale_k_sin = concat_tensors(scale_k_sin_list, dim_groups_range, half_head_dim)

        if select_topk_dim:
            dim_mask = dpe_dim_mask[self.layer_idx]
            dim_mask = dim_mask.unsqueeze(2).expand(-1, -1, value_states.size(2), -1)  # [len(scale_factors), 1, num_heads, seq_len, head_dim]
            scale_q_cos = torch.where(dim_mask, scale_q_cos, normal_cos)
            scale_q_sin = torch.where(dim_mask, scale_q_sin, normal_sin)
            scale_k_cos = torch.where(dim_mask, scale_k_cos, normal_cos)
            scale_k_sin = torch.where(dim_mask, scale_k_sin, normal_sin)
                
            normal_query_states, normal_key_states = apply_rotary_pos_emb(query_states, key_states, normal_cos, normal_sin, None)
            
            key_states = repeat_kv(key_states, self.num_key_value_groups)
            value_states = repeat_kv(value_states, self.num_key_value_groups)

            scale_query_states, _ = apply_rotary_pos_emb_by_heads(query_states, None, scale_q_cos, scale_q_sin, None)
            _, scale_key_states = apply_rotary_pos_emb_by_heads(None, key_states, scale_k_cos, scale_k_sin, None)

            normal_query_states = normal_query_states.transpose(1, 2).contiguous()
            normal_key_states = repeat_kv(normal_key_states, self.num_key_value_groups).transpose(1, 2).contiguous()
            scale_query_states = scale_query_states.transpose(1, 2).contiguous()
            scale_key_states = scale_key_states.transpose(1, 2).contiguous()
            value_states = value_states.transpose(1, 2).contiguous()
        else:
            normal_query_states, normal_key_states = apply_rotary_pos_emb(query_states, key_states, normal_cos, normal_sin, None)
            scale_query_states, _ = apply_rotary_pos_emb(query_states, None, scale_q_cos, scale_q_sin, None)
            _, scale_key_states = apply_rotary_pos_emb(None, key_states, scale_k_cos, scale_k_sin, None)
            

            normal_query_states = normal_query_states.transpose(1, 2).contiguous()
            normal_key_states = normal_key_states.transpose(1, 2).contiguous()
            scale_query_states = scale_query_states.transpose(1, 2).contiguous()
            scale_key_states = scale_key_states.transpose(1, 2).contiguous()
            value_states = value_states.transpose(1, 2).contiguous()

        attn_output = atten_merge_forward(
                                        query_position,
                                        window_size,
                                        normal_query_states,
                                        normal_key_states,
                                        scale_query_states,
                                        scale_key_states,
                                        value_states,
                                        bsz,
                                        kv_seq_len,
                                        attn_dropout,
                                    )
    else:
        raise ValueError("q_len should be 1 or seq_len.")
    
    attn_output = attn_output.contiguous()
    attn_output = attn_output.view(bsz, q_len, -1).contiguous()
    attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None
    return attn_output, attn_weights, past_key_value

def causal_forward(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
) -> Union[Tuple, CausalLMOutputWithPast]:
    r"""
    Args:
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

    Returns:

    Example:

    ```python
    >>> from transformers import AutoTokenizer, LlamaForCausalLM

    >>> model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
    >>> tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

    >>> prompt = "Hey, are you conscious? Can you talk to me?"
    >>> inputs = tokenizer(prompt, return_tensors="pt")

    >>> # Generate
    >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
    >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
    ```"""
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
    outputs = self.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
        # cache_position=cache_position,
    )

    hidden_states = outputs[0]
    full_logits_length = 32000

    if hidden_states.shape[-2] < full_logits_length:
        logits = self.lm_head(hidden_states)
        logits = logits.float()
        loss = None

        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)

            loss = loss_fct(shift_logits, shift_labels)
    else:
        res = 0
        div_len = full_logits_length // 2
        if labels is None:
            # only produce the last logits
            logits = self.lm_head(hidden_states[..., -1:, :])
            logits = logits.float()
            # logits = logits.expand(-1, hidden_states.shape[-2], -1)
            loss = None
        else:
            # calculate loss by chunk
            shift_hidden_states = hidden_states[..., :-1, :]
            shift_labels = labels[..., 1:].contiguous()

            for i in range(0, shift_hidden_states.shape[-2], div_len):
                st = i
                ed = min(i + div_len, shift_hidden_states.shape[-2])
                logits = self.lm_head(shift_hidden_states[..., st:ed, :])
                logits = logits.float()

                shift_logits = logits.contiguous()
                # Flatten the tokens
                loss_fct = CrossEntropyLoss()
                shift_logits = shift_logits.view(-1, self.config.vocab_size)
                shift_labels = shift_labels.view(-1)
                # Enable model parallelism
                shift_labels = shift_labels.to(shift_logits.device)

                res = res + loss_fct(shift_logits, shift_labels[st:ed]) * (ed - st)
            loss = res / (hidden_states.shape[-2] - 1)
            logits = None

    if not return_dict:
        output = (logits,) + outputs[1:]
        return (loss,) + output if loss is not None else output

    return CausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )

def dimension_selection(topk, selected_dim_path):
    selected_dim = torch.load(selected_dim_path).to("cuda")
    selected_dim = (torch.topk(selected_dim.mean(dim=0).mean(dim=0), dim=-1, k=topk)[1])
    L, H, K = selected_dim.shape  # layer, head, 48
    mask = torch.zeros((L, 1, H, 64), dtype=torch.bool, device=selected_dim.device)
    layer_idx = torch.arange(L, device=selected_dim.device).view(L, 1, 1).expand(L, H, K)   # [L, H, K]
    head_idx = torch.arange(H, device=selected_dim.device).view(1, H, 1).expand(L, H, K)    # [L, H, K]

    mask[layer_idx, 0, head_idx, selected_dim] = True
    dim_mask = torch.cat([mask, mask], dim=-1)
    return dim_mask

def replace_with_dpe(scale_factors_arg, local_window_size_arg, dim_groups_range_arg, select_topk_dim_arg=False, topk_dim_arg=None, selected_dim_path_arg=None):
    print("============== [DPE Config for Llama] ===============")
    print(f"Local Window Size: {local_window_size_arg}")
    print(f"Dimension Groups Numbers: {len(scale_factors_arg)}")
    print(f"Scale factors for different dimension groups: {scale_factors_arg}")
    ranges = [f"{start}-{end-1}" for start, end in zip(dim_groups_range_arg[:-1], dim_groups_range_arg[1:])]
    print(f"Dimension groups range: {', '.join(ranges)}")
    print(f"Select Topk Dimensions: {select_topk_dim_arg}")
    if select_topk_dim_arg:
        print(f"Only Scale Topk {topk_dim_arg} Dimensions")
        assert topk_dim_arg is not None, "topk_dim_arg should be set when select_topk_dim_arg is True"
        assert selected_dim_path_arg is not None, "selected_dim_path_arg should be set when select_topk_dim_arg is True"
    else:
        print("Scale all Dimensions")
    print("==============================================")
    if select_topk_dim_arg:
        dim_mask = dimension_selection(topk_dim_arg, selected_dim_path_arg)
    else:   
        dim_mask = None

    global dpe_scale_factors, dpe_local_window_size, dpe_dim_groups_range, dpe_select_topk_dim, dpe_dim_mask
    
    dpe_scale_factors = scale_factors_arg
    dpe_local_window_size = local_window_size_arg
    dpe_dim_groups_range = dim_groups_range_arg
    dpe_select_topk_dim = select_topk_dim_arg
    dpe_dim_mask = dim_mask

    
    transformers.models.mistral.modeling_mistral.MistralAttention.forward = dpe_flash_forward
    transformers.models.mistral.modeling_mistral.MistralFlashAttention2.forward = dpe_flash_forward
    transformers.models.llama.modeling_mistral.MistralForCausalLM.forward = causal_forward