import os
import sys
import logging
import random
import collections

import torch
import torch.utils.data
import tqdm
import numpy as np
import pandas as pd

from transformers import AutoTokenizer

logger = logging.getLogger(__name__)

class BaseEHRDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data,
        input_path,
        split,
        concept='descemb',
        feature='entire',
        train_task=None,
        ratio='100',   
        seed='2020',
        **kwargs,
    ):
        self.base_path = os.path.join(input_path, icu_type, data)
        self.feature = feature

        self.concept = concept
        self.structure = 'hi' if '_' in concept else 'fl'
        self.root = concept.split('_')[0]

        self.train_task = train_task

        self.seed = seed
        self.split = split

                
        self.seed = seed

        self.data_dir = os.path.join(
            self.base_path, self.feature, self.root, self.structure
        )

        #self.time_bucket_idcs = [idx for idx in range(6, 6+20+1)] #start range + bucket num + 1
        self.time_bucket_idcs = [idx for idx in range(4, 24)] #start range + bucket num + 1
        self.fold_file = os.path.join(
            self.base_path, "fold", "fold_{}.csv".format(ratio)
        )

        assert self.root in ['codeemb', 'descemb'], (
            'please ensure --concept starts one of ["codeemb", "descemb"] '
            '--concept: {}'.format(concept)
        )

    def __len__(self):
        raise NotImplementedError
    
    def __getitem__(self, index):
        raise NotImplementedError

    def get_fold_indices(self, return_all=False):
        if self.split == 'train':
            hit = 1
        elif self.split == 'valid':
            hit = 2
        elif self.split == 'test':
            hit = 0

        df = pd.read_csv(self.fold_file)
        if return_all:
            return np.arange(len(df))
        
        if self.train_task =='predict':
            col_name = self.pred_target + '_' + self.seed + '_strat'

        elif self.train_task=='pretrain':
            col_name = str(self.seed) + '_rand'
            if self.split =='train':
                df.loc[df[col_name] == 0, col_name] = 1

        splits = df[col_name].values
        idcs = np.where(splits == hit)[0]
    
        return idcs
    
    def get_num_events(self):
        df = pd.read_csv(self.fold_file)

        return df['num_events'].values

    def mask_tokens(self, inputs: torch.Tensor, mask_vocab_size : int, 
                    mask_token : int, special_tokens_mask: torch.Tensor, masked_indices=None):
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """
        inputs = torch.LongTensor(inputs)
        labels = inputs.clone()
        # We sample a few tokens in each sequence for MLM training (with probability `self.textencoder_mlm_probability`)
        probability_matrix = torch.full(labels.shape, self.mlm_prob)

        if 'descemb' in self.concept:
            special_tokens_mask = torch.tensor(
                self.tokenizer.get_special_tokens_mask(
                    labels, already_has_special_tokens=True
                ),
                dtype=torch.bool
            )
        else:
            special_tokens_mask = (inputs== 0 )| (inputs ==1) | (inputs ==2)
        
        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        if masked_indices is None:
            masked_indices = torch.bernoulli(probability_matrix).bool()

            while torch.equal(masked_indices, torch.zeros(len(masked_indices)).bool()):
                masked_indices = torch.bernoulli(probability_matrix).bool()

        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = mask_token

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(mask_vocab_size, labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]
        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs.numpy(), labels.numpy()

    def span_masking(self, inputs: torch.Tensor, mask_vocab_size : int, 
                        mask_token : int, special_tokens_mask: torch.Tensor, masked_indices=None):
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """
        inputs = torch.LongTensor(inputs)
        labels = inputs.clone()
        # time token idcs
        idcs = torch.where(
            (self.time_bucket_idcs[0]<=labels) & (labels<=self.time_bucket_idcs[-1]) 
        )[0]
        if masked_indices is None:
            masked_indices = torch.zeros_like(inputs).bool()
            # sampling event with mlm 
            if not len(idcs) <3:
                que = torch.randperm(len(idcs)-1)[:round((len(idcs)-1)*self.mlm_prob)]
                for mask_event_idx in que:
                    masked_indices[idcs[mask_event_idx]:idcs[mask_event_idx+1]] = True
        
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 1.0)).bool() & masked_indices
        inputs[indices_replaced] =mask_token
        
        return inputs.numpy(), labels.numpy(),

class EHRDataset(BaseEHRDataset):
    def __init__(
        self,
        data,
        input_path,
        split,
        vocab,
        concept='descemb',
        feature='entire',
        train_task=None,
        pretrain_task=None,
        ratio='100',
        pred_target='mort',
        seed='2020',
        mask_list='input, type',
        mlm_prob=0.3,
        activate_fold=True,
        reload=False,
        **kwargs,
    ):
        super().__init__(
            data=data,
            input_path=input_path,
            split=split,
            concept=concept,
            feature=feature,
            train_task=train_task,
            ratio=ratio,
            icu_type=icu_type,
            split_method=split_method,
            seed=seed,
            **kwargs,
        )

        self.vocab = vocab

        self.pred_target = pred_target

        self.tokenizer = None
        self.mask = None
        if (
            self.train_task == 'pretrain'
            and pretrain_task in ['mlm', 'spanmlm']
            and self.structure == 'fl'
        ):
            self.tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
            self.mask = self.span_masking if pretrain_task=='spanmlm' else self.mask_tokens

        self.mask_list = mask_list
        self.mlm_prob = mlm_prob
        label = np.load(
            os.path.join(
                self.base_path, 'label', pred_target + '.npy'
            ), allow_pickle=True
        )
        self.label = torch.tensor(label, dtype=torch.long)

        self.num_events = self.get_num_events()
        if activate_fold:
            self.hit_idcs = self.get_fold_indices()
            self.num_events = self.num_events[self.hit_idcs]
            self.label = self.label[self.hit_idcs]

        input_index_size_dict = {
                            'mimic3' : {
                                'select' : 6532,
                                'entire' :10389
                            },
                            'eicu' : {
                                'select' : 4151,
                                'entire' : 6305
                            },
                            'mimic4' : {
                                'select' : 5581,
                                'entire' : 9568
                            }
        } 

        type_index_size_dict = {
                            'mimic3' : {
                                'select' : 9,
                                'entire' :10
                            },
                            'eicu' : {
                                'select' : 10,
                                'entire' :10
                            },
                            'mimic4' : {
                                'select' : 10,
                                'entire' :9
                            }
        }

        self.mask_vocab_size = None
        self.mask_token = None
        if self.mask:
            if 'codeemb' in concept:
                self.mask_vocab_size = {
                    'input': input_index_size_dict[data][feature]-1,
                    'type': type_index_size_dict[data][feature]-1
                }
            else:
                self.mask_vocab_size = {
                    'input' : 28117,
                    'type' : 12,
                     'dpe' : 23
                    }            

            if 'codeemb' in self.concept:
               self.mask_token = {
                   'input': 3,
                    'type': 3,
                    }
            else:
                self.mask_token = {
                    'input': 103,
                    'type': 3,
                    'dpe': 1,
                }
        if not reload:
            logger.info(f'loaded {len(self.hit_idcs)} {self.split} samples')

    def __len__(self):
        return len(self.hit_idcs)

class HierarchicalEHRDataset(EHRDataset):
    def __init__(self, max_word_len=256, max_seq_len=512, **kwargs):
        super().__init__(**kwargs)

        self.max_word_len = max_word_len
        self.max_seq_len = max_seq_len
        
        self.cls = 101 if self.concept.startswith('descemb') else 1

    def crop_to_max_size(self, arr, target_size):
        """
        arr: 1d np.array of indices
        """
        size = len(arr)
        diff = size - target_size
        if diff <= 0:
            return arr

        return arr[:target_size]

    def collator(self, samples):
        samples = [s for s in samples if s['input_ids'] is not None]
        if len(samples) == 0:
            return {}
        
        input = dict()
        out = dict()

        input['input_ids'] = [s['input_ids'] for s in samples]
        input['type_ids'] = [s['type_ids'] for s in samples]
        if 'descemb' in self.concept:
            input['dpe_ids'] = [s['dpe_ids'] for s in samples]
        
        seq_sizes = []
        word_sizes = []
        for s in input['input_ids']:
            seq_sizes.append(len(s))
            for w in s:
                word_sizes.append(len(w))


        target_seq_size = min(max(seq_sizes), self.max_seq_len) #min 값 골라서 그거 기준으로 max 잡음
        target_word_size = min(max(word_sizes), self.max_word_len) 

        collated_input = dict()
        for k in input.keys():
            collated_input[k] = torch.zeros(
                (len(input['input_ids']), target_seq_size, target_word_size,)
            ).long()
 
        for i, seq_size in enumerate(seq_sizes):
            for j in range(len(input['input_ids'][i])):
                word_size = len(input['input_ids'][i][j])
                diff = word_size - target_word_size
                for k in input.keys():
                    if diff == 0:
                        pass
                    elif diff < 0:
                        try:
                            input[k][i][j] = np.append(input[k][i][j], [0] * -diff)
                        except ValueError:
                            input[k][i] = list(input[k][i])
                            input[k][i][j] = np.append(input[k][i][j], [0] * -diff)
                    else:
                        input[k][i][j] = np.array(input[k][i][j][:self.max_word_len])

            diff = seq_size - target_seq_size
            for k in input.keys():
                if k == 'input_ids':
                    prefix = self.cls
                else:
                    prefix = 1
                input[k][i] = np.array(list(input[k][i]))
                if diff == 0:
                    collated_input[k][i] = torch.from_numpy(input[k][i])
                elif diff < 0:
                    padding = np.zeros((-diff, target_word_size - 1,))
                    padding = np.concatenate(
                        [np.full((-diff, 1), fill_value=prefix), padding], axis=1
                    )
                    collated_input[k][i] = torch.from_numpy(
                            np.concatenate(
                            [input[k][i], padding], axis=0
                        )
                    )
                else:
                    collated_input[k][i] = torch.from_numpy(
                        self.crop_to_max_size(input[k][i], target_seq_size)
                    )

        out['net_input'] = collated_input
        if 'label' in samples[0]:
            out['label'] = torch.stack([s['label'] for s in samples])

        return out

    def __getitem__(self, index):
        num_event = self.num_events[index]
        fname = str(self.hit_idcs[index]) + '.npy'

        input_ids = np.load(os.path.join(self.data_dir, 'input_ids', fname), allow_pickle=True)
        type_ids = np.load(os.path.join(self.data_dir, 'type_ids', fname), allow_pickle=True)
        dpe_ids = None
        if self.concept.startswith('descemb'):
            dpe_ids = np.load(os.path.join(self.data_dir, 'dpe_ids', fname), allow_pickle=True)
        label = self.label[index]

        out = {
            'input_ids': input_ids[-num_event:],
            'type_ids': type_ids[-num_event:],
            'dpe_ids': dpe_ids[-num_event:] if dpe_ids is not None else None,
            'label': label,
        }

        return out

class FlattenEHRDataset(EHRDataset):
    def __init__(self, max_seq_len=8192, **kwargs):
        super().__init__(**kwargs)

        self.max_seq_len = max_seq_len

        self.time_token = 4
       
    def crop_to_max_size(self, arr, target_size):
        """
        arr: 1d np.array of indices
        """
        size = len(arr)
        diff = size - target_size
        if diff <= 0:
            return arr

        if not self.mask:
            return arr[:target_size]
        
        start = np.random.randint(0, diff + 1)
        end = size - diff + start
        return arr[start:end]

    def sample_crop_indices(self, size, diff):
        if self.mask:
            start = np.random.randint(0, diff + 1)
            end = size - diff + start
        else:
            start = 0
            end = size - diff
        return start, end

    def collator(self, samples):
        samples = [s for s in samples if s['input_ids'] is not None]
        if len(samples) == 0:
            return {}

        input = dict()
        out = dict()

        input['input_ids'] = [s['input_ids'] for s in samples]
        input['type_ids'] = [s['type_ids'] for s in samples]
        if self.concept.startswith('descemb'):
            input['dpe_ids'] = [s['dpe_ids'] for s in samples]

        if self.mask:
            for victim in self.mask_list:
                input[victim+'_label']= [s[victim+'_label'] for s in samples]

        sizes = [len(s) for s in input['input_ids']]
        if self.args.pred_model =='cnn':
            target_size = self.max_seq_len
        else:
            target_size = min(max(sizes), self.max_seq_len) # target size

        collated_input = dict()
        for k in input.keys():
            collated_input[k] = torch.zeros((len(input['input_ids']), target_size)).long()

        for i, size in enumerate(sizes):
            diff = size - target_size
            if diff > 0:
                start, end = self.sample_crop_indices(size, diff)
            for k in input.keys():
                if diff == 0:
                    collated_input[k][i] = torch.LongTensor(input[k][i])
                elif diff < 0:
                    collated_input[k][i] = torch.from_numpy(
                            np.concatenate(
                            [input[k][i], np.zeros((-diff,))], axis=0
                        )
                    )
                else:
                    collated_input[k][i] = torch.LongTensor(input[k][i][start:end])

        out['net_input'] = collated_input
        if 'label' in samples[0]:
            out['label'] = torch.stack([s['label'] for s in samples])

        return out

    def __getitem__(self, index):
        num_event = self.num_events[index]
        fname = str(self.hit_idcs[index]) + '.npy'

        input_ids = np.load(os.path.join(self.data_dir, 'input_ids', fname), allow_pickle=True)
        type_ids = np.load(os.path.join(self.data_dir, 'type_ids', fname), allow_pickle=True)
        dpe_ids = None
        if self.concept.startswith('descemb'):
            dpe_ids = np.load(os.path.join(self.data_dir, 'dpe_ids', fname), allow_pickle=True)
        label = self.label[index]

        out = {
            'input_ids': input_ids,
            'type_ids': type_ids,
            'dpe_ids': dpe_ids if dpe_ids is not None else None,
            'label': label,
        }

        if self.mask:
            masked_indices = None
            for victim in self.mask_list:
                victim_ids, victim_label = self.mask(
                    inputs=out[victim + '_ids'],
                    mask_vocab_size=self.mask_vocab_size[victim],
                    mask_token=self.mask_token[victim],
                    special_tokens_mask=None,
                    masked_indices=masked_indices
                )
                if masked_indices is None:
                    masked_indices = torch.tensor(victim_label!= -100)
                out[victim + '_ids'] = victim_ids
                out[victim + '_label'] = victim_label
        else:
            time_token_idcs = np.where(
                np.array(type_ids) == self.time_token
            )[0]

            if len(time_token_idcs) == num_event:
                onset = 0
            else:
                onset = time_token_idcs[-num_event - 1] + 1

            out['input_ids'] = out['input_ids'][onset:]
            out['type_ids'] = out['type_ids'][onset:]
            if dpe_ids is not None:
                out['dpe_ids'] = out['dpe_ids'][onset:]

        return out
