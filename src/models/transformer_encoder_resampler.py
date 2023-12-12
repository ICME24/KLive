import torch
import torch.nn as nn
from .transformer_layer_resampler import TransformerLayer_resampler
from .layers.layer_norm import LayerNorm, T5LayerNorm
from .layers.relative_position_embedding import RelativePositionEmbedding

class TransformerEncoder_resampler(nn.Module):
    def __init__(self, args):
        super(TransformerEncoder_resampler, self).__init__()
        self.mask = args.mask
        self.layers_num = args.encode_layers_num
        self.parameter_sharing = args.parameter_sharing
        self.factorized_embedding_parameterization = args.factorized_embedding_parameterization
        self.layernorm_positioning = args.layernorm_positioning
        self.relative_position_embedding = args.relative_position_embedding
        self.has_residual_attention = args.has_residual_attention

        has_bias = bool(1 - args.remove_transformer_bias)

        if self.factorized_embedding_parameterization:
            try:
                self.linear = nn.Linear(args.emb_size, args.encoder_hidden_size)
            except:
                self.linear = nn.Linear(args.emb_size, args.hidden_size)

        if self.parameter_sharing:
            self.transformer = TransformerLayer_resampler(args, args.encoder_hidden_size)
        else:
            self.transformer_first = TransformerLayer_resampler(args, args.encoder_hidden_size)
            self.transformer = nn.ModuleList(
                [TransformerLayer_resampler(args, args.encoder_hidden_size) for _ in range(self.layers_num - 1)]
            )
        if self.layernorm_positioning == "pre":
            if args.layernorm == "t5":
                try:
                    self.layer_norm = T5LayerNorm(args.encoder_hidden_size)
                except:
                    self.layer_norm = T5LayerNorm(args.hidden_size)
            else:
                try:
                    self.layer_norm = LayerNorm(args.encoder_hidden_size)
                except:
                    self.layer_norm = LayerNorm(args.hidden_size)

        if self.relative_position_embedding:
            self.relative_pos_emb = RelativePositionEmbedding(bidirectional=True, heads_num=args.heads_num,
                                                              num_buckets=args.relative_attention_buckets_num)


    def forward(self, emb,author_emb, seg):
        if self.factorized_embedding_parameterization:
            emb = self.linear(emb)

        batch_size, seq_length, _ = emb.size()
        if self.mask == "fully_visible":
            mask = (seg > 0). \
                unsqueeze(1). \
                repeat(1, seq_length, 1). \
                unsqueeze(1)
            mask = mask.float()
            mask = (1.0 - mask) * -10000.0

        elif self.mask == "causal":
            mask = torch.ones(seq_length, seq_length, device=emb.device)
            mask = torch.tril(mask)
            mask = (1.0 - mask) * -10000
            mask = mask.repeat(batch_size, 1, 1, 1)
        else:
            mask_a = (seg == 1). \
                unsqueeze(1). \
                repeat(1, seq_length, 1). \
                unsqueeze(1).float()

            mask_b = (seg > 0). \
                unsqueeze(1). \
                repeat(1, seq_length, 1). \
                unsqueeze(1).float()

            mask_tril = torch.ones(seq_length, seq_length, device=emb.device)
            mask_tril = torch.tril(mask_tril)
            mask_tril = mask_tril.repeat(batch_size, 1, 1, 1)

            mask = (mask_a + mask_b + mask_tril >= 2).float()
            mask = (1.0 - mask) * -10000.0

        hidden = emb
        
        position_bias = None

        prev_attn = None
        for i in range(self.layers_num):
            if self.parameter_sharing:
                k_v_input = torch.cat((emb, author_emb), dim = -2)
                query = author_emb
                query, prev_attn = self.transformer(k_v_input, query, mask, position_bias=position_bias,
                                                     has_residual_attention=self.has_residual_attention,
                                                     prev_attn=prev_attn)
            else:
                if i == 0 :
                    k_v_input = torch.cat((emb, author_emb), dim = -2)
                    query = author_emb
                    query, prev_attn = self.transformer_first(k_v_input, query, mask, position_bias=position_bias,
                                                     has_residual_attention=self.has_residual_attention,
                                                     prev_attn=prev_attn)
                else:
                    k_v_input = torch.cat((emb, query), dim = -2)
                    query, prev_attn = self.transformer[i - 1](k_v_input,query, mask, position_bias=position_bias,
                                                            has_residual_attention=self.has_residual_attention,
                                                            prev_attn=prev_attn)

        if self.layernorm_positioning == "pre":
            return self.layer_norm(query)
        else:
            return query
