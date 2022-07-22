import math
import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from models import register_model

@register_model("transformer")
class EncoderTransformer(nn.Module):
    def __init__(self, args):
        super().__init__()
        
        self.args = args
        encoder_layers = TransformerEncoderLayer(
            args.pred_dim,
            args.n_heads,   # default=8
            args.pred_dim*4,
            args.dropout,
            batch_first=True
            )
        
        self.transformer_encoder = TransformerEncoder(encoder_layers, args.n_layers)
        self.pos_encoder = PositionalEncoding(
            args.pred_dim, args.dropout, args.max_seq_len
            ) if '_' in args.input2emb_model else None

        self.layer_norm = nn.LayerNorm(args.embed_dim, eps=1e-12)

    @classmethod
    def build_model(cls, args):
        """Build a new model instance."""
        return cls(args)

    def forward(self, x, input_ids, **kwargs):
        # input_ids: (B, S) (B x S, W ) -> (Bx s, W) -> (B, s, W)
                      
        B, S = input_ids.shape[0], input_ids.shape[1]
        if '_' in self.args.input2emb_model: 
            src_pad_mask = input_ids[:, :, 1].eq(0).to(x.device)
        else:
            src_pad_mask = input_ids.eq(0).to(x.device) 
        
        src_mask= None
        
        '''
        Note: [src/tgt/memory]_mask ensures that position i is allowed to attend the unmasked positions. 
                    If a ByteTensor is provided, 
                        the non-zero positions are not allowed to attend
                        while the zero positions will be unchanged. 

                    If a BoolTensor is provided, 
                        positions with ``True`` are not allowed to attend 
                        while ``False`` values will be unchanged. 

                    If a FloatTensor is provided, it will be added to the attention weight. 
                    https://pytorch.org/docs/sfeature/generated/torch.nn.Transformer.html

        '''
        if self.pos_encoder is not None:
            x = self.layer_norm(self.pos_encoder(x))
        encoder_output = self.transformer_encoder(x, mask=src_mask, src_key_padding_mask=src_pad_mask)

        return encoder_output

  

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout, max_len):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x): 
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """

        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)