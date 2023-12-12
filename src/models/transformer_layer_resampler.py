import torch.nn as nn
from .layer_norm import LayerNorm, T5LayerNorm
from .position_ffn import PositionwiseFeedForward, GatedFeedForward
from .multi_headed_attn import MultiHeadedAttention, MultiHeadedAttention_resampler
from .relative_position_embedding import RelativePositionEmbedding

class TransformerLayer_resampler(nn.Module):
    def __init__(self, args, hidden_size):
        super(TransformerLayer_resampler, self).__init__()

        self.layernorm_positioning = args.layernorm_positioning

        if hasattr(args, "attention_head_size"):
            attention_head_size = args.attention_head_size
        else:
            attention_head_size = hidden_size // args.heads_num

        has_bias = bool(1 - args.remove_transformer_bias)
        with_scale = bool(1 - args.remove_attention_scale)

        self.self_attn = MultiHeadedAttention_resampler(
            hidden_size, args.heads_num, attention_head_size, args.dropout, has_bias=has_bias, with_scale = with_scale
        )
        self.dropout_1 = nn.Dropout(args.dropout)

        if args.feed_forward == "gated":
            self.feed_forward = GatedFeedForward(
                hidden_size, args.feedforward_size, args.hidden_act, has_bias
            )
        else:
            self.feed_forward = PositionwiseFeedForward(
                hidden_size, args.feedforward_size, args.hidden_act, has_bias
            )
        self.dropout_2 = nn.Dropout(args.dropout)

        if args.layernorm == "t5":
            self.layer_norm_1 = T5LayerNorm(hidden_size)
            self.layer_norm_2 = T5LayerNorm(hidden_size)
        else:
            self.layer_norm_1 = LayerNorm(hidden_size)
            self.layer_norm_2 = LayerNorm(hidden_size)

    def forward(self, k_v, query, mask, position_bias = None, has_residual_attention=False, prev_attn=None):
        if self.layernorm_positioning == "post":
            inter, prev_attn_out, _ = self.self_attn(k_v, k_v, query, mask, position_bias, has_residual_attention, prev_attn)
            inter = self.dropout_1(inter)
            inter = self.layer_norm_1(inter + query)
            output = self.dropout_2(self.feed_forward(inter))
            output = self.layer_norm_2(output + inter)
        else:
            k_v = self.layer_norm_1(k_v)
            query = self.layer_norm_1(query)
            inter, prev_attn_out, _ = self.self_attn(k_v, k_v, query, mask, position_bias, has_residual_attention, prev_attn)
            inter = self.dropout_1(inter)
            inter = query + inter
            output = self.layer_norm_2(inter)
            output = self.dropout_2(self.feed_forward(output)) + inter
        return output, prev_attn_out