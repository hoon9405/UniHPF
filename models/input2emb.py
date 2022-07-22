import torch.nn as nn
from models import register_model
from models.transformer import PositionalEncoding

@register_model("codeemb")
class CodeEmb(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        input_index_size_dict = {
                'mimic3' : {
                    'select' : 6532,
                    'whole' :10389
                },
                'eicu' : {
                    'select' : 4151,
                    'whole' :6305
                },
                'mimic4' : {
                    'select' : 5581,
                    'whole' :9568
                },
                
                # pooled
                'mimic3_eicu':{
                    'select' :  9316,
                    'whole' : 14452,
                },
                
                'mimic3_mimic4':{
                    'select' :  7771,
                    'whole' : 15356
                },
                'mimic4_eicu':{
                    'select' :  9813,
                    'whole' : 15676,
                },
                'mimic3_mimic4_eicu':{
                    'select' : 11716,
                    'whole' : 20792,
                }
            }

        type_index_size_dict = {
            'mimic3' : {
                'select' : 9,
                'whole' : 10
            },
            'eicu' : {
                'select' : 10,
                'whole' : 10
            },
            'mimic4' : {
                'select' : 10,
                'whole' : 9
            },
            
            # pooled
            'mimic3_eicu':{
                    'select' : 26,
                    'whole' : 69,
                },
            
            'mimic3_mimic4':{
                'select' : 16,
                'whole' : 53
            },
            'mimic4_eicu':{
                'select' :  25,
                'whole' : 63,
            },
            'mimic3_mimic4_eicu':{
                'select' :  26,
                'whole' : 75,
            }
        }
        
        if args.train_type=='transfer':
            data = args.eval_data[0]

        self.input_index_size = input_index_size_dict[data][args.feature][args.column_embed]
        self.type_index_size = type_index_size_dict[data][args.feature][args.column_embed]

        self.input_ids_embedding =nn.Embedding(self.input_index_size, args.embed_dim, padding_idx=0)
        self.type_ids_embedding =nn.Embedding(self.type_index_size, args.embed_dim, padding_idx=0)
        

        max_len = args.max_word_len if '_' in args.input2emb_model else args.max_seq_len
        
        self.pos_encoder = PositionalEncoding(  
            args.embed_dim, args.dropout, max_len
            ) if self.args.pos_enc else None

        self.layer_norm = nn.LayerNorm(args.embed_dim, eps=1e-12)
        self.dropout = nn.Dropout(args.dropout)

    
    @classmethod
    def build_model(cls, args):
        """Build a new model instance."""
        return cls(args)
    

    def forward(self, input_ids, type_ids, **kwargs):
        B, S= input_ids.shape[0], input_ids.shape[1]
    
        x = self.input_ids_embedding(input_ids)

        if self.args.mapping:
            x = self.mapping_matrix(x)
        
        if self.type_ids_embedding: 
            x += self.type_ids_embedding(type_ids) 
        
        if '_' in self.args.input2emb_model:
            x = x.view(B*S, -1, self.args.embed_dim) 
            
        if self.pos_encoder:   
            x = self.pos_encoder(x) # (B, S, W, E) -> (B*S, W, E)
        x = self.dropout(self.layer_norm(x))
        return x



@register_model("descemb")
class DescEmb(nn.Module):
    def __init__(self, args, embed_dim=None):
        super().__init__()
        
        self.args = args
        
        self.input_index_size = 28119 #28996 # bio clinical bert vocab
        self.type_index_size = 14 # mimic3 + eicu + mimic4
        self.dpe_index_size = 25

        self.dpe = args.dpe
        self.token_type = args.type_token
        self.pos_enc = args.pos_enc

        if embed_dim:
            self.args.embed_dim = embed_dim        

        self.input_ids_embedding = nn.Embedding(
                self.input_index_size, self.args.embed_dim, padding_idx=0
        )

        self.type_ids_embedding =nn.Embedding(
                self.type_index_size, self.args.embed_dim, padding_idx=0
        ) if self.args.type_token else None

        self.dpe_ids_embedding =nn.Embedding(
            self.dpe_index_size, self.args.embed_dim, padding_idx=0
        ) if self.args.dpe else None

        max_len = args.max_word_len if '_' in args.input2emb_model else args.max_seq_len

        self.pos_encoder = PositionalEncoding(  
            args.embed_dim, args.dropout, max_len
            ) if self.pos_enc else None
        
        self.layer_norm = nn.LayerNorm(args.embed_dim, eps=1e-12)

    @classmethod
    def build_model(cls, args):
        """Build a new model instance."""
        return cls(args)

    def load_pretrained_embed_weight(self, load_dict):
        for victim, weight in load_dict:
            getattr(self, victim+'_embedding').from_pretrained(weight) 

    def forward(self, input_ids, type_ids, dpe_ids, only_features=False, **kwargs):
        B, S = input_ids.shape[0], input_ids.shape[1]
        
        x = self.input_ids_embedding(input_ids)

        if self.type_ids_embedding: # column description mean 
            x += self.type_ids_embedding(type_ids) 

        if self.dpe_ids_embedding:
            x += self.dpe_ids_embedding(dpe_ids)
        
        if self.args.structure=='hi':
            x = x.view(B*S, -1, self.args.embed_dim) 
            
        if self.pos_encoder:   
            x = self.pos_encoder(x) # (B, S, W, E) -> (B*S, W, E)
        x = self.layer_norm(x)
        return x

