import logging
import torch.nn as nn
from models import register_model, MODEL_REGISTRY

logger = logging.getLogger(__name__)

@register_model("base_model")
class BaseModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        self.input2emb_model = self._input2emb_model.build_model(args)
        self.pred_model = self._pred_model.build_model(args)
        self.emb2out_model = self._emb2out_model.build_model(args)

    @property
    def _input2emb_model(self):
        if '_' in self.args.input2emb_model:
            return MODEL_REGISTRY[self.args.input2emb_model.split('_')[1]+'_enc']
        else:
            return MODEL_REGISTRY[self.args.input2emb_model]
    
    @property
    def _pred_model(self):
        return MODEL_REGISTRY[self.args.pred_model]
    
    @property
    def _emb2out_model(self):
        if self.args.train_task =='predict':
            return MODEL_REGISTRY['predout']
        elif self.args.train_task=='pretrain':
            return MODEL_REGISTRY['mlmout']
    
    @classmethod
    def build_model(cls, args):
        """Build a new model instance."""
        return cls(args)

    def get_logits(self, net_output):
        return net_output.float()

    def get_targets(self, sample):
        if self.args.train_task=='predict':
            return sample['label'].float()
        
        elif self.args.train_task=='pretrain':
            return{
              victim+'_label' :sample['net_input'][victim+'_label']
                for victim in self.args.mask_list
            }
                
            

    def forward(self, **kwargs):
        all_codes_embs = self.input2emb_model(**kwargs)  # (B, S, E)
        
        x = self.pred_model(all_codes_embs, **kwargs)
        net_output = self.emb2out_model(x, **kwargs)

        return net_output

