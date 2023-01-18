import torch
import torch.nn as nn
import math

from torch.nn import TransformerEncoder, TransformerEncoderLayer
from models import register_model

from models.utils import PositionalEncoding

@register_model("eventaggregator")
class EventAggregator(nn.Module):
    def __init__(self, args):
        super().__init__()
        
        self.args = args

        self.pos_encoder = None
        if self.args.time_embed == 'aggregator':
            self.dropout = nn.Dropout(p=args.dropout)
        elif args.structure == 'hi':
            self.pos_encoder = PositionalEncoding(
                args.pred_dim, args.dropout, args.max_seq_len
            )

        self.layer_norm = nn.LayerNorm(args.embed_dim, eps=1e-12)

        encoder_layers = TransformerEncoderLayer(
            args.pred_dim,
            args.n_heads,   # default=8
            args.pred_dim*4,
            args.dropout,
            batch_first=True
        )
        
        self.transformer_encoder = TransformerEncoder(encoder_layers, args.n_layers)

    @classmethod
    def build_model(cls, args):
        """Build a new model instance."""
        return cls(args)

    def forward(self, x, input_ids, times, **kwargs):
        # input_ids: (B, S) (B x S, W ) -> (Bx s, W) -> (B, s, W)
                      
        B, S = input_ids.shape[0], input_ids.shape[1]
        if self.args.structure=='hi': 
            # True for padded events (B, S)
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
            https://pytorch.org/docs/stable/generated/torch.nn.Transformer.html

        '''
        #TODO: (argument - time_embed: encoder / aggregator)
        if self.args.time_embed == 'aggregator':
            times = times.unsqueeze(-1) # (B, S, 1)
            div_term = torch.exp(torch.arange(0, self.args.pred_dim, 2) * (-math.log(10000.0) / self.args.pred_dim)).to(x.device) # (pred_dim/2, )
            pe = torch.zeros(B, S, self.args.pred_dim) # (B, S, pred_dim)
            pe[:, :, 0::2] = torch.sin(times * div_term)
            pe[:, :, 1::2] = torch.cos(times * div_term)
            x = self.layer_norm(self.dropout(x + pe.to(x.device)))

        elif self.pos_encoder is not None:
            x = self.layer_norm(self.pos_encoder(x))
        # x: (B, S, E)
        # For each event, attend to all other non-pad events in the same icustay
        encoder_output = self.transformer_encoder(x, mask=src_mask, src_key_padding_mask=src_pad_mask)

        return encoder_output
