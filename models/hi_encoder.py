import logging
import torch
import torch.nn as nn
from models import register_model, MODEL_REGISTRY
from torch.nn import TransformerEncoder, TransformerEncoderLayer

logger = logging.getLogger(__name__)

@register_model("bert_enc")
class EventEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.pred_dim = args.pred_dim

        self.enc_model = self._enc_model.build_model(args)

        encoder_layers = TransformerEncoderLayer(
            args.pred_dim,
            args.n_heads,   # default=8
            args.pred_dim*4,
            args.dropout,
            batch_first=True
            )
        
        self.transformer_encoder = TransformerEncoder(encoder_layers, args.n_layers)
        self.layer_norm = nn.LayerNorm(args.embed_dim, eps=1e-12)

        self.post_encode_proj = (
            nn.Linear(args.embed_dim, self.pred_dim)
        )
   
        self.mlm_proj = (
            nn.Linear(args.embed_dim, 28996)
            if args.pretrain_task == "text_encoder_mlm" else None
        )

    @classmethod
    def build_model(cls, args):
        """Build a new model instance."""
        return cls(args)

    @property
    def _enc_model(self):
        return MODEL_REGISTRY[self.args.input2emb_model.split('_')[0]]

    def forward(self, input_ids, **kwargs):
      
        B, S, _= input_ids.size()
        x = self.enc_model(input_ids, **kwargs) # (B*S, W, E)
     
        src_pad_mask = input_ids.view(B*S, -1).eq(0).to(x.device) # (B, S, W) -> (B*S, W)
        encoder_output = self.transformer_encoder(x, src_key_padding_mask=src_pad_mask)

        if (
            (self.args.train_task == 'pretrain' and self.args.pretrain_task == 'w2v')
            or (self.args.pretrained_load == 'w2v')
            or (self.args.apply_mean)
        ):
            x = encoder_output
            x[src_pad_mask] = 0
            x = torch.div(x.sum(dim=1), (x!=0).sum(dim=1))
            net_output = self.post_encode_proj(x).view(B, -1, self.pred_dim)
        else:
            net_output = (
                self.post_encode_proj(
                    encoder_output[:, 0, :]
                ).view(B, -1, self.pred_dim) 
            )

        if self.mlm_proj:
            mlm_output = self.mlm_proj(bert_outputs[0]) # (B x S, W, H) -> (B x S, W, Bert-vocab)
            return mlm_output

        return net_output