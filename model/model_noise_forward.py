import torch
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions

def bert_noise_forward(
    self,
    hidden_states,
    attention_mask=None,
    head_mask=None,
    encoder_hidden_states=None,
    encoder_attention_mask=None,
    past_key_values=None,
    use_cache=None,
    output_attentions=False,
    output_hidden_states=False,
    return_dict=True,
):
    all_hidden_states = () if output_hidden_states else None
    all_self_attentions = () if output_attentions else None
    all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None

    next_decoder_cache = () if use_cache else None

    for i, layer_module in enumerate(self.layer):
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        layer_head_mask = head_mask[i] if head_mask is not None else None
        past_key_value = past_key_values[i] if past_key_values is not None else None

        if getattr(self.config, "gradient_checkpointing", False) and self.training:

            if use_cache:
                logger.warning(
                    "`use_cache=True` is incompatible with `config.gradient_checkpointing=True`. Setting "
                    "`use_cache=False`..."
                )
                use_cache = False

            def create_custom_forward(module):
                def custom_forward(*inputs):
                    return module(*inputs, past_key_value, output_attentions)

                return custom_forward

            layer_outputs = torch.utils.checkpoint.checkpoint(
                create_custom_forward(layer_module),
                hidden_states,
                attention_mask,
                layer_head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
            )
        else:
            layer_outputs_ = layer_module(
                hidden_states,
                attention_mask,
                layer_head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                past_key_value,
                output_attentions,
            )
            if i%self.config.nth_layers==0 and self.config.single_layer==False:
                #print(f"Multi Layer: {i}-layer")
                randn_noise = torch.randn_like(layer_outputs_[0])*self.config.noise_eps

                temp  = layer_outputs_[0]+randn_noise

                if output_attentions:
                    layer_outputs = (temp, layer_outputs_[1])
                else:
                    layer_outputs = tuple(temp[None,:])

            else:
                layer_outputs = layer_outputs_

        hidden_states = layer_outputs[0]
        if use_cache:
            next_decoder_cache += (layer_outputs[-1],)
        if output_attentions:
            all_self_attentions = all_self_attentions + (layer_outputs[1],)
            if self.config.add_cross_attention:
                all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

    if output_hidden_states:
        all_hidden_states = all_hidden_states + (hidden_states,)

    if not return_dict:
        return tuple(
            v
            for v in [
                hidden_states,
                next_decoder_cache,
                all_hidden_states,
                all_self_attentions,
                all_cross_attentions,
            ]
            if v is not None
        )
    return BaseModelOutputWithPastAndCrossAttentions(
        last_hidden_state=hidden_states,
        past_key_values=next_decoder_cache,
        hidden_states=all_hidden_states,
        attentions=all_self_attentions,
        cross_attentions=all_cross_attentions,
    )
def roberta_noise_forward(
    self,
    hidden_states,
    attention_mask=None,
    head_mask=None,
    encoder_hidden_states=None,
    encoder_attention_mask=None,
    past_key_values=None,
    use_cache=None,
    output_attentions=False,
    output_hidden_states=False,
    return_dict=True,
):
    """
    - RoBERTa encoder model's forward function without an word embedding layer(nn.Embedding)
    """
    all_hidden_states = () if output_hidden_states else None
    all_self_attentions = () if output_attentions else None
    all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None

    next_decoder_cache = () if use_cache else None


    for i, layer_module in enumerate(self.layer):
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        layer_head_mask = head_mask[i] if head_mask is not None else None
        past_key_value = past_key_values[i] if past_key_values is not None else None

        if getattr(self.config, "gradient_checkpointing", False) and self.training:

            if use_cache:
                logger.warning(
                    "`use_cache=True` is incompatible with `config.gradient_checkpointing=True`. Setting "
                    "`use_cache=False`..."
                )
                use_cache = False

            def create_custom_forward(module):
                def custom_forward(*inputs):
                    return module(*inputs, past_key_value, output_attentions)

                return custom_forward

            layer_outputs = torch.utils.checkpoint.checkpoint(
                create_custom_forward(layer_module),
                hidden_states,
                attention_mask,
                layer_head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
            )
        else:
            layer_outputs_ = layer_module(
                hidden_states, # torch.Size([16, 80, 768])
                attention_mask, # torch.Size([16, 1, 1, 80])
                layer_head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                past_key_value,
                output_attentions,
            )

            if i%self.config.nth_layers==0 and self.config.single_layer==False:
                #print(f"Multi Layer: {i}-layer")
                randn_noise = torch.randn_like(layer_outputs_[0])*self.config.noise_eps

                temp  = layer_outputs_[0]+randn_noise

                if output_attentions:
                    layer_outputs = (temp, layer_outputs_[1])
                else:
                    layer_outputs = tuple(temp[None,:])

            else:
                #print(f"FF>0: {i}-layer")
                layer_outputs = layer_outputs_

        hidden_states = layer_outputs[0]

        if use_cache:
            next_decoder_cache += (layer_outputs[-1],)
        if output_attentions:
            all_self_attentions = all_self_attentions + (layer_outputs[1],)
            if self.config.add_cross_attention:
                all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

    if output_hidden_states:
        all_hidden_states = all_hidden_states + (hidden_states,)

    if not return_dict:
        return tuple(
            v
            for v in [
                hidden_states,
                next_decoder_cache,
                all_hidden_states,
                all_self_attentions,
                all_cross_attentions,
            ]
            if v is not None
        )
    return BaseModelOutputWithPastAndCrossAttentions(
        last_hidden_state=hidden_states,
        past_key_values=next_decoder_cache,
        hidden_states=all_hidden_states,
        attentions=all_self_attentions,
        cross_attentions=all_cross_attentions,
    )

