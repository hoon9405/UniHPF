from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np

class BaseMetric(object):
    def __init__(self): 
        pass

    @property
    def compare(self):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError
    
    def __call__(self):
        raise NotImplementedError

class PredMetric(BaseMetric):
    def __init__(self, args, target='auprc'):
        self._update_target = target
        self.reset()
        
        self.pred_target = args.pred_target

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
        

        self.mlb = None
        if args.pred_target in ['fi_ac', 'im_disch']:
            self.mlb = MultiLabelBinarizer()
            class_n = self.multi_label_dict[args.src_data][args.pred_target]
            self.mlb.fit([[i] for i in range(class_n)])

    def reset(self):
        self.loss = 0
        self.truths = []
        self.preds = []

    def __call__(self, loss: float, preds: np.ndarray, truths: np.ndarray, **kwargs):
        self.truths += list(truths)
        self.preds += list(preds)
        self.loss += loss

    def get_epoch_dict(self, total_iter):
        self.epoch_dict = {
            'auroc' : self.auroc,
            'auprc' : self.auprc,
            'loss' :  self.loss / total_iter
            }
        self.reset()

        return self.epoch_dict

    @property
    def compare(self):
        return 'decrease' if self.update_target == 'loss' else 'increase'

    @property
    def update_target(self):
        return self._update_target

    @property
    def auroc(self):
        if self.mlb:
            self.truths = self.mlb.transform(
            np.expand_dims(np.array(self.truths, dtype='int'), axis=1)).flatten()
        return roc_auc_score(self.truths, self.preds)

    @property
    def auprc(self):
        return average_precision_score(self.truths, self.preds, average='micro')

class PretrainMetric(BaseMetric):
    def __init__(self, args, target=None):
        self.mask_list = args.mask_list
        self.reset()
    
    def reset(self):
        self.loss = 0
        self.total = {
            victim+'_ids': 0 for victim in self.mask_list
        }
        self.correct = {
            victim+'_ids': 0 for victim in self.mask_list
        }

    def __call__(self, loss, total, correct):
        self.loss += loss
        for victim in self.mask_list:
            self.total[victim+'_ids'] += total[victim+'_ids']
        for victim in self.mask_list:
            self.correct[victim+'_ids'] += correct[victim+'_ids']

    def get_epoch_dict(self, total_iter):
        log_dict = {'Loss': self.loss / total_iter}
        for victim in self.mask_list:
            log_dict[victim+'_ids_acc'] = self.correct[victim+'_ids'] / self.total[victim+'_ids']
        self.reset()

        return log_dict

    @property
    def compare(self):
        return 'decrease' if self.update_target == 'loss' else 'increase'

    @property
    def update_target(self):
        return 'input_ids_acc'

class W2VMetric(BaseMetric):
    def __init__(self, target='acc'):
        self._update_target = target
        self.reset()
    
    def reset(self):
        self.loss_0 = 0
        self.loss_1 = 0
        self.loss_2 = 0
        self.prob_ppl = 0
        self.code_ppl = 0
        self.total = 0
        self.correct = 0

    def update(
        self,
        loss_0=0,
        loss_1=0,
        loss_2=0,
        total=0,
        correct=0,
        prob_ppl=0,
        code_ppl=0,
    ):
        self.loss_0 += loss_0
        self.loss_1 += loss_1
        self.loss_2 += loss_2
        
        self.prob_ppl += prob_ppl
        self.code_ppl += code_ppl
        self.total += total
        self.correct += correct
    
    def get_epoch_dict(self, total_iter):
        epoch_dict = {
            'loss': (self.loss_0 + self.loss_1 + self.loss_2) / total_iter,
            'loss_0': self.loss_0 / total_iter,
            'loss_1': self.loss_1 / total_iter,
            'loss_2': self.loss_2 / total_iter,
            'prob_ppl': self.prob_ppl / total_iter,
            'code_ppl': self.code_ppl / total_iter,
            'acc': self.acc
        }
        self.reset()
    
        return epoch_dict
    
    @property
    def compare(self):
        return 'decrease' if self.update_target == 'loss' else 'increase'
    
    @property
    def update_target(self):
        return self._update_target
    
    @property
    def acc(self):
        return self.correct / self.total if self.total > 0 else float("nan")