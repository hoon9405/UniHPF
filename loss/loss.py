from tkinter.messagebox import NO
from tkinter import N
from builtins import NotImplemented
import torch.nn.functional as F
import torch.nn as nn



class BaseLoss():
    def __init__(self, args):
        self.args = args
        self.BCE_loss = nn.BCEWithLogitsLoss()
        self.CE_loss = nn.CrossEntropyLoss()

    def __call__(self):
        return None


class PredLoss(BaseLoss):
    def __init__(self, args):
        super().__init__(args=args)
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
            }
    def __call__(self, output, target):
        if self.args.pred_target in ['mort', 'los3', 'los7', 'readm']:
            return self.binary_class(output, target)

        elif self.args.pred_target in ['im_disch', 'fi_ac']:
            return self.multi_class(output, target)

        elif self.args.pred_target in ['dx']:
            return self.multi_label_multi_class(output, target)


    def binary_class(self, output, target):
        return self.BCE_loss(output['pred_output'], 
                            target.to(output['pred_output'].device)
                            )
    
    def multi_class(self, output, target):
        return self.CE_loss(
            output['pred_output'], F.one_hot(
            target.long(), 
            self.multi_label_dict[self.args.src_data][self.args.pred_target]
            ).float().to(output['pred_output'].device)
            )

    def multi_label_multi_class(self, output, target):
        return self.BCE_loss(
            output['pred_output'].view(-1), 
            target.view(-1).to(output['pred_output'].device)
            )
    

class PretrainLoss(BaseLoss):
    def __init__(self, args):
        super().__init__(args=args)
        self.args = args


    def __call__(self, output, target):
        if self.args.pretrain_task in ['mlm', 'spanmlm']:
            return self.MLM_tokens(output, target)


    def MLM_tokens(self, output, target):
        B, S, _ = output['input_ids'].shape
        Loss = []
        
        for victim in self.args.mask_list:
            
            Loss.append(self.CE_loss(output[victim+'_ids'].view(B*S, -1), 
                target[victim+'_label'].view(-1).to(output[victim+'_ids'].device)
                )
            )
        
        return sum(Loss)
       
    
class NoteLoss(BaseLoss):
    def __init__(self, args):
        super().__init__(args=args)

        self.args = args
    def __call__(self):
        return None
