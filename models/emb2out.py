import torch.nn as nn
from models import register_model

@register_model("predout")
class PredOutPutLayer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        self.multi_label_dict = {
            'mimic3':{
                'dx':18,
                'im_disch':17,
                'fi_ac':18},
            'eicu':{
                'dx':18,
                'im_disch':8,
                'fi_ac':9},
             'mimic4':{
                 'dx':18,
                 'im_disch':17,
                 'fi_ac':18},
            'mimic3_eicu':{
                  'dx':18,
                   },
             'mimic3_mimic4':{
                   'dx':18,
                   'im_disch':17,
                    'fi_ac':18},
             'mimic3_mimic4_eicu':{
                     'dx':18,
                     },
              }

        self.final_proj = nn.Linear(
            args.pred_dim,
            self.multi_label_dict[args.src_data][args.pred_target] 
            if args.pred_target in ['dx', 'fi_ac', 'im_disch'] else 1 
        ) 
   
    @classmethod
    def build_model(cls, args):
        """Build a new model instance."""
        return cls(args)


    def forward(self, x, input_ids, **kwargs):
        B, S = input_ids.size(0),  input_ids.size(1) 
        if self.args.pred_pooling =='cls':
            x = x[:, 0, :]
        elif self.args.pred_pooling =='mean':
            if '_' in self.args.input2emb_model: 
                mask = ~input_ids[:, :, 1].eq(102)
            else:
                mask = (input_ids!=0)
            mask = mask.unsqueeze(dim=2).to(x.device).expand(B, S, self.args.pred_dim)
            x = (x*mask).sum(dim=1)/mask.sum(dim=1)
            #x = x.mean(dim=1)
        output = self.final_proj(x) # B, E -> B, 1
        output = output.squeeze()
        return {'pred_output': output}


@register_model("mlmout")
class MLMOutPutLayer(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.args = args
        if self.args.input2emb_model.startswith('codeemb'):
            input_index_size_dict = {
                'mimic3' : {#6543
                    'select' : {'cc': 5372, 'ct':6532},
                    'whole' :{'cc': 9017, 'ct':10389}
                },
                'eicu' : {
                    'select' : {'cc': 3971, 'ct':4151},
                    'whole' :{'cc': 5462, 'ct':6305}
                },
                'mimic4' : {
                    'select' : {'cc': 5869, 'ct':5581},
                    'whole' :{'cc': 10241, 'ct':9568}
                },
                
                # pooled
                'mimic3_eicu':{
                    'select' : {'cc': 5869},
                    'whole' :{'cc': 10241},
                },
                
                'mimic3_mimic4':{
                    'select' : {'cc': 7771},
                    'whole' :{'cc': 15356}
                },
                'mimic4_eicu':{
                    'select' : {'cc': 9813},
                    'whole' :{'cc': 15676},
                },
                'mimic3_mimic4_eicu':{
                    'select' : {'cc': 11716},
                    'whole' :{'cc': 20792},
                }
            }

            type_index_size_dict = {
                'mimic3' : {
                    'select' : {'cc': 17, 'ct':9},
                    'whole' :{'cc': 48, 'ct':10}
                },
                'eicu' : {
                    'select' : {'cc':16, 'ct':10},
                    'whole' :{'cc': 28, 'ct':10}
                },
                'mimic4' : {
                    'select' : {'cc': 16, 'ct':10},
                    'whole' :{'cc': 42, 'ct':9}
                },
                
                # pooled
                'mimic3_eicu':{
                        'select' : {'cc': 26},
                        'whole' :{'cc': 69},
                    },
                
                'mimic3_mimic4':{
                    'select' : {'cc': 16},
                    'whole' :{'cc': 53}
                },
                'mimic4_eicu':{
                    'select' : {'cc': 25},
                    'whole' :{'cc': 63},
                },
                'mimic3_mimic4_eicu':{
                    'select' : {'cc': 26},
                    'whole' :{'cc': 75},
                }
            }
            input_index_size = input_index_size_dict[args.src_data][args.feature][args.column_embed]
            type_index_size = type_index_size_dict[args.src_data][args.feature][args.column_embed]
        else:
            input_index_size = 28996
            type_index_size = 14
            dpe_index_size = 25
        
        self.input_ids_out = nn.Linear(
            args.embed_dim,
            input_index_size
        ) 
        
        self.type_ids_out = nn.Linear(
            args.embed_dim,
            type_index_size 
        ) if 'type' in self.args.mask_list else None

        self.dpe_ids_out = nn.Linear(
            args.embed_dim,
            dpe_index_size
        ) if 'dpe' in self.args.mask_list else None
    
    @classmethod
    def build_model(cls, args):
        """Build a new model instance."""
        return cls(args)

    def forward(self, x, **kwargs):
        input_ids = self.input_ids_out(x)
        type_ids = self.type_ids_out(x)  if self.type_ids_out else None
        dpe_ids = self.dpe_ids_out(x) if self.dpe_ids_out else None

        return {
            'input_ids' : input_ids,
            'type_ids' : type_ids,
            'dpe_ids' : dpe_ids
        }
