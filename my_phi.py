import bert
import tensorflow as tf
import cache

import math
from typing import Optional, Union, Tuple, List

import masking_utils


# note: Instead of using register buffers, I am just using normal variables, but 
# this might cause issues when exporting models

# stuff to work on:
""" stuff to work on:
- testing everything
"""

def Flatten(x):
    elements = 1
    for d in x.shape:
        elements *= d
    return tf.reshape(x, (elements))

def _get_unpad_data(attention_mask):
    assert(attention_mask.dtype == tf.int32)
    seqlens_in_batch = tf.cast(tf.math.reduce_sum(attention_mask, axis=-1), dtype=tf.int32)
    zero = tf.constant(0, dtype=tf.int32)
    indices = Flatten(tf.where(tf.not_equal(Flatten(attention_mask), zero)))
    max_seqlen_in_batch = tf.math.reduce_max(seqlens_in_batch).numpy().item()
    cu_seqlens = tf.pad(tf.cast(tf.math.cumsum(seqlens_in_batch, axis=0), tf.int32), [[0,0],[1,1]])
    return (
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
    )

class PhiRotaryEmbedding(tf.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, name=None):
        super().__init__(name)
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.inv_freq = 1.0 / (self.base ** (tf.cast((tf.range(0, self.dim, 2)), dtype=tf.float32) / self.dim))
        self._set_cos_sin_cache(max_position_embeddings, dtype=tf.float16)
    def _set_cos_sin_cache(self, seq_len, dtype):
        self.max_seq_len_cached = seq_len
        t = tf.range(self.max_seq_len_cached, dtype=self.inv_freq.dtype)
        freqs = tf.experimental.numpy.outer(t, self.inv_freq)
        emb = tf.concat((freqs, freqs), axis=-1)
        self.cos_cached = tf.cast(tf.cos(emb), dtype=dtype)
        self.sin_cached = tf.cast(tf.sin(emb), dtype=dtype)
    def __call__(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, dtype=x.dtype)
        return (
            tf.cast(self.cos_cached[:seq_len], dtype=x.dtype),
            tf.cast(self.sin_cached[:seq_len], dtype=x.dtype),
        )

class PhiLinearScalingRotaryEmbedding(PhiRotaryEmbedding):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, scaling_factor=1.0):
        self.scaling_factor = scaling_factor
        super().__init__(dim, max_position_embeddings, base)

    def _set_cos_sin_cache(self, seq_len, dtype):
        self.max_seq_len_cached = seq_len
        t = tf.range(self.max_seq_len_cached, dtype=self.inv_freq.dtype)
        t = t / self.scaling_factor
        freqs = tf.outer(t, self.inv_freq)
        emb = tf.concat((freqs, freqs), axis=-1)
        self.cos_cached = tf.cast(tf.cos(emb), dtype=dtype)
        self.sin_cached = tf.cast(tf.sin(emb), dtype=dtype)

class PhiDynamicNTKScalingRotaryEmbedding(PhiRotaryEmbedding):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, scaling_factor=1.0):
        self.scaling_factor = scaling_factor
        super().__init__(dim, max_position_embeddings, base)

    def _set_cos_sin_cache(self, seq_len, dtype):
        self.max_seq_len_cached = seq_len
        if seq_len > self.max_position_embeddings:
            base = self.base * (
                (self.scaling_factor * seq_len / self.max_position_embeddings) - (self.scaling_factor - 1)
            ) ** (self.dim / (self.dim - 2))
            self.inv_freq = 1.0 / (base ** (tf.cast(tf.range(0, self.dim, 2), dtype=tf.float32) / self.dim))

        t = tf.range(self.max_seq_len_cached, dtype=self.inv_freq.dtype)
        freqs = tf.outer(t, self.inv_freq)
        emb = tf.concat((freqs, freqs), axis=-1)
        self.cos_cached = tf.cast(tf.cos(emb), dtype=dtype)
        self.sin_cached = tf.cast(tf.sin(emb), dtype=dtype)

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return tf.concat((-x2, x1), axis=-1)

def apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
    """See orginal source for better description of what code does"""
    cos = tf.expand_dims(tf.gather(cos, position_ids), axis=unsqueeze_dim)
    sin = tf.expand_dims(tf.gather(sin, position_ids), axis=unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class PhiMLP_Config():
    def __init__(self):
        self.hidden_act
        self.hidden_size
        self.intermediate_size

class NewGELU(tf.Module):
    def __init__(self, name=None):
        super().__init__(name)
    def __call__(self, input):
        return 0.5 * input * (1.0 + tf.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * tf.pow(input, 3.0))))

# need to come back to this
class PhiMLP(tf.Module):
    def __init__(self, config:PhiMLP_Config, params, name=None):
        super().__init__(name)
        self.config = config
        self.activation_fn = NewGELU() # check this matches the actual model
        self.fc1 = bert.Dense_v2(config.hidden_size, config.intermediate_size,
                                  params['mlp_fc1_weight'], params['mlp_fc1_bias'])
        self.fc2 = bert.Dense_v2(config.intermediate_size, config.hidden_size, 
                                 params['mlp_fc2_weight'], params['mlp_fc2_bias'])
    def __call__(self, hidden_states):
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states) # also the function is stateless (consider changing)
        hidden_states = self.fc2(hidden_states)
        return hidden_states

def repeat_kv(hidden_states, n_rep: int):
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states    
    hidden_states = tf.broadcast_to(hidden_states[:, :, None, :, :], (batch, num_key_value_heads, n_rep, slen, head_dim))
    return tf.reshape(hidden_states, (batch, num_key_value_heads * n_rep, slen, head_dim))


class PhiConfig():
    def __init__(self) -> None:
        self.attention_dropout
        self.hidden_size
        self.num_attention_heads
        self.num_key_value_heads
        self.max_position_embeddings
        self.rope_theta
        self.partial_rotary_factor
        self.layer_norm_eps
        self.rope_scaling # appears to be a dictionary of sort


class PhiAttention(tf.Module):
    def __init__(self, config: PhiConfig, params, layer_idx:int = None, name=None):
        super().__init__(name)
        self.config = config
        self.layer_idx = layer_idx
        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.partial_rotary_factor = config.partial_rotary_factor
        self.is_causal = True
        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.q_proj = bert.Dense_v2(self.hidden_size, self.num_heads * self.head_dim, params['q_proj_weight'], params['q_proj_bias'])
        self.k_proj =bert.Dense_v2(self.hidden_size, self.num_key_value_heads * self.head_dim, params['k_proj_weight'], params['k_proj_bias'])
        self.v_proj = bert.Dense_v2(self.hidden_size, self.num_key_value_heads * self.head_dim, params['v_proj_weight'], params['v_proj_bias'])
        self.dense = bert.Dense_v2(self.num_heads * self.head_dim, self.hidden_size, params['dense_weight'], params['dense_bias'])
        self.qk_layernorm = config.qk_layernorm
        # if self.qk_layernorm:
        #     self.q_layernorm = bert.LayerNorm(
        #         params['q_ln_weight'], params['q_ln_bias'], eps=config.layer_norm_eps)
        #     self.k_layernorm = bert.LayerNorm(
        #         params['k_ln_weight'], params['k_ln_bias'], eps=config.layer_norm_eps,)
        self._init_rope()
    def _init_rope(self):
        if self.config.rope_scaling is None:
            self.rotary_emb = PhiRotaryEmbedding(
                int(self.partial_rotary_factor * self.head_dim),
                max_position_embeddings=self.max_position_embeddings,
                base=self.rope_theta,
            )
        else:
            scaling_type = self.config.rope_scaling["type"]
            scaling_factor = self.config.rope_scaling["factor"]
            if scaling_type == "linear":
                self.rotary_emb = PhiLinearScalingRotaryEmbedding(
                    int(self.partial_rotary_factor * self.head_dim),
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            elif scaling_type == "dynamic":
                self.rotary_emb = PhiDynamicNTKScalingRotaryEmbedding(
                    int(self.partial_rotary_factor * self.head_dim),
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            else:
                raise ValueError(f"Unknown RoPE scaling type {scaling_type}")
    def __call__(
        self,
        hidden_states,
        attention_mask = None,
        position_ids = None,
        past_key_value = None,
        output_attentions: bool = False,
        use_cache: bool = False):
        bsz, q_len, _ = hidden_states.shape

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        if self.qk_layernorm:
            query_states = self.q_layernorm(query_states)
            key_states = self.k_layernorm(key_states)

        query_states =  tf.reshape(query_states,[bsz, q_len, self.num_heads,           self.head_dim])
        key_states =    tf.reshape(key_states,  [bsz, q_len, self.num_key_value_heads, self.head_dim])
        value_states =  tf.reshape(value_states,[bsz, q_len, self.num_key_value_heads, self.head_dim])
        query_states =  tf.transpose(query_states,  perm=[0, 2, 1, 3])
        key_states =    tf.transpose(key_states,    perm=[0, 2, 1, 3])
        value_states =  tf.transpose(value_states,  perm=[0, 2, 1, 3])

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            if self.layer_idx is None:
                raise ValueError(
                    f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                    "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                    "with a layer index."
                )
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)

        # Partial rotary embedding
        query_rot, query_pass = (
            query_states[..., : self.rotary_emb.dim],
            query_states[..., self.rotary_emb.dim :],
        )
        key_rot, key_pass = (
            key_states[..., : self.rotary_emb.dim],
            key_states[..., self.rotary_emb.dim :],
        )
        # [batch_size, seq_length, num_heads, head_dim // config.partial_rotary_factor]
        query_rot, key_rot = apply_rotary_pos_emb(query_rot, key_rot, cos, sin, position_ids)

        # [batch_size, seq_length, num_heads, head_dim]
        query_states = tf.concat((query_rot, query_pass), axis=-1)
        key_states = tf.concat((key_rot, key_pass), axis=-1)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "partial_rotation_size": self.rotary_emb.dim}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        # Queries and keys upcast to fp32 is required by Phi-2 to avoid overflow
        attn_weights = tf.matmul(
            tf.cast(query_states, dtype=tf.float32), 
            tf.transpose(tf.cast(key_states, dtype=tf.float32), perm=(0, 1, 3, 2))
        ) / math.sqrt(self.head_dim)

        if attn_weights.shape != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.shape != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + tf.cast(attention_mask, dtype=tf.float32)

        # upcast attention to fp32
        attn_weights = tf.cast(tf.nn.softmax(tf.cast(attn_weights, dtype=tf.float32), axis=-1), dtype=value_states.dtype)

        attn_output = tf.matmul(attn_weights, value_states)

        if attn_output.shape != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = tf.transpose(attn_output, perm=(0, 2, 1, 3))
        attn_output = tf.reshape(attn_output, (bsz, q_len, self.hidden_size))
        attn_output = self.dense(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

# for now, lets try without flash attention

PHI_ATTENTION_CLASSES = {
    "eager": PhiAttention,
    # "flash_attention_2": PhiFlashAttention2,
}

class PhiDecoderLayer(tf.Module):
    def __init__(self, config: PhiConfig, params, layer_idx: int, name=None):
        super().__init__(name)
        self.self_attn = PHI_ATTENTION_CLASSES[config._attn_implementation](config, params ,layer_idx=layer_idx)
        self.mlp = PhiMLP(config, params)
        self.input_layernorm = bert.LayerNorm(params['layernorm_weight'],
                                              params['layernorm_bias'],
                                              eps=config.layer_norm_eps)

    def __call__(
        self,
        hidden_states,
        attention_mask = None,
        position_ids = None,
        output_attentions = False,
        use_cache = False,
        past_key_value = None,
    ):
        """
        see orginal code for description
        """
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        # Self Attention
        attn_outputs, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache)
        feed_forward_hidden_states = self.mlp(hidden_states)
        hidden_states = attn_outputs + feed_forward_hidden_states + residual
        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)
        if use_cache:
            outputs += (present_key_value,)
        return outputs

class BaseModelOutputWithPast():
    def __init__(self, last_hidden_state, past_key_values, hidden_states, attentions):
        self.last_hidden_state= last_hidden_state
        self.past_key_values = past_key_values
        self.hidden_states = hidden_states
        self.attentions = attentions

class PhiModel(tf.Module):
    def __init__(self, config: PhiConfig, params, name=None):
        super().__init__(name)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = bert.Embedding(config.vocab_size, config.hidden_size, params['embed_tokens'])
        self.layers = [PhiDecoderLayer(config, params['decoder_layers'][layer_idx], layer_idx) 
                       for layer_idx in range(config.num_hidden_layers)]
        self.final_layernorm = bert.LayerNorm(params['final_layernorm_weight'], params['final_layernorm_bias'],
                                               eps=config.layer_norm_eps)
        self._use_flash_attention_2 = config._attn_implementation == "flash_attention_2" # should be False

    def __call__(
        self,
        input_ids = None,
        attention_mask = None,
        position_ids = None,
        past_key_values = None,
        inputs_embeds = None,
        use_cache: bool = False, # for, as they are using a transformers construct
        output_attentions: bool = None,
        output_hidden_states: bool = None,
        return_dict: bool = None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape[:2]
        elif inputs_embeds is not None:
            batch_size, seq_length = inputs_embeds.shape[:2]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        past_key_values_length = 0

        if use_cache:
            use_legacy_cache = not isinstance(past_key_values, cache.Cache)
            if use_legacy_cache:
                past_key_values = cache.DynamicCache.from_legacy_cache(past_key_values)
            past_key_values_length = past_key_values.get_usable_length(seq_length)

        if position_ids is None:
            position_ids = tf.range(
                past_key_values_length, seq_length + past_key_values_length, dtype=tf.int32)
            position_ids = tf.expand_dims(position_ids, axis=0)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        # Attention mask.
        # if self._use_flash_attention_2: # should be False
        #     # 2d mask is passed through the layers
        #     attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
        # else
            
        # 4d mask is passed through the layers
        attention_mask = masking_utils._prepare_4d_causal_attention_mask(
            tf.cast(attention_mask, dtype=inputs_embeds.dtype), (batch_size, seq_length), 
            inputs_embeds, past_key_values_length
        )

        hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.final_layernorm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = None
        if use_cache:
            next_cache = next_decoder_cache.to_legacy_cache() if use_legacy_cache else next_decoder_cache
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

class CausalLMOutputWithPast():
    def __init__(self, logits, past_key_values, hidden_states, attentions):
        self.logits = logits
        self.past_key_values = past_key_values
        self.hidden_states = hidden_states
        self.attentions = attentions


class PhiForCausalLM(tf.Module):
    _tied_weights_keys = ["lm_head.weight"]
    def __init__(self, config, params ,name=None):
        super().__init__(name)
        self.model = PhiModel(config, params)
        self.vocab_size = config.vocab_size
        self.lm_head = bert.Dense_v2(config.hidden_size, config.vocab_size,
                                      params['lm_head_weight'], params['lm_head_bias'])
    
    def __call__(
        self,
        input_ids = None,
        attention_mask = None,
        position_ids = None,
        past_key_values: Optional[List] = None,
        inputs_embeds = None,
        labels = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
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
        )

        # hidden_states = outputs[0]
        hidden_states = outputs.last_hidden_state
        logits = self.lm_head(hidden_states)
        logits = tf.cast(logits, dtype=tf.float32) # maybe should be 64

        if not return_dict:
            return (logits,) + outputs[1:]

        return CausalLMOutputWithPast(
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
    # create the prepare_inputs_for_generation() 
    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values is not None:
            if isinstance(past_key_values, cache.Cache):
                cache_length = past_key_values.get_seq_length()
                past_length = past_key_values.seen_tokens
                max_cache_length = past_key_values.get_max_length()
            else:
                cache_length = past_length = past_key_values[0][0].shape[2]
                max_cache_length = None

            # Keep only the unprocessed tokens:
            # 1 - If the length of the attention_mask exceeds the length of input_ids, then we are in a setting where
            # some of the inputs are exclusively passed as part of the cache (e.g. when passing input_embeds as
            # input)
            if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
                input_ids = input_ids[:, -(attention_mask.shape[1] - past_length) :]
            # 2 - If the past_length is smaller than input_ids', then input_ids holds all input tokens. We can discard
            # input_ids based on the past_length.
            elif past_length < input_ids.shape[1]:
                input_ids = input_ids[:, past_length:]
            # 3 - Otherwise (past_length >= input_ids.shape[1]), let's assume input_ids only has unprocessed tokens.

            # If we are about to go beyond the maximum cache length, we need to crop the input attention mask.
            if (
                max_cache_length is not None
                and attention_mask is not None
                and cache_length + input_ids.shape[1] > max_cache_length
            ):
                attention_mask = attention_mask[:, -max_cache_length:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = tf.cumsum(tf.cast(attention_mask, dtype=tf.int32), axis=-1) - 1
            position_ids = masking_utils.mask_fill(position_ids, tf.cast(attention_mask == 0, tf.int32), 1)

            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs