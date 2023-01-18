import logging

import torch
import torch.nn as nn

from models import register_model, MODEL_REGISTRY

logger = logging.getLogger(__name__)

@register_model("ehr_model")
class UniHPF(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        self.emb_type_model = self._emb_type_model.build_model(args)
        self.pred_model = self._pred_model.build_model(args)
        self.emb2out_model = self._emb2out_model.build_model(args)
        
    #TODO: group emb_type, pred_model, emb2out to parent model
    @property
    def _emb_type_model(self):
        if self.args.structure=='hi':
            return MODEL_REGISTRY['eventencoder']
        else:
            return MODEL_REGISTRY['descemb']
    
    @property
    def _pred_model(self):
        return MODEL_REGISTRY[self.args.pred_model]
    
    @property
    def _emb2out_model(self):
        return MODEL_REGISTRY['predout']
    
    @classmethod
    def build_model(cls, args):
        """Build a new model instance."""
        return cls(args)

    @classmethod
    def from_pretrained(cls, args, checkpoint=None, state_dict=None):
        model = cls(args)

        if state_dict is None: 
            state_dict = torch.load(checkpoint, map_location='cpu')['model']     
        
        #Transfer learning codebase emb
        if args.train_task =='finetune' and args.emb_type=='codebase':
            state_dict = {
                    k: v for k,v in state_dict.items() if (
                        ('input2emb' not in k) and ('pos_enc' not in k)
                    )
                }
        
        model.load_state_dict(state_dict)
    
        return model 


    def forward(self, **kwargs):

        all_codes_embs = self.emb_type_model(**kwargs)  # (B, S, E)
        x = self.pred_model(all_codes_embs, **kwargs)
        net_output = self.emb2out_model(x, **kwargs)

        output = {
            'logits': net_output,
        }

        return output


    def get_outputs(self, net_output):

        logits = net_output['logits']['pred_output']
        
        return logits


    def get_targets(self, sample):

        return sample['labels']
