import os
import sys
import logging
import random
import collections

import torch
import torch.nn.functional as F
import torch.utils.data
import tqdm
import numpy as np
import pandas as pd

from transformers import AutoTokenizer
from sklearn.preprocessing import MultiLabelBinarizer
#np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)

logger = logging.getLogger(__name__)

# Dataset created from https://github.com/Jwoo5/integrated-ehr-pipeline

# 1. set input / fold directory
# 2. input split icustay ids (get fold indices)
# 3. masking functions (should i leave it here?)
# 4. label loading
class BaseEHRDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data,
        emb_type,
        feature,
        input_path,
        split,
        structure,
        train_task=None,
        ratio='100',   
        seed='2020',
        **kwargs,
    ):
        self.data_dir = os.path.join(input_path, data, emb_type, feature) #TODO: data sampling dir
        
        self.stay_id = {
            "mimiciii": "ICUSTAY_ID",
            "eicu": "patientunitstayid",
            "mimiciv": "stay_id"
        }[data]

        self.split = split

        self.structure = structure
        self.train_task = train_task

        self.seed = seed

        # Get split icustay ids
        self.fold_file = pd.read_csv(os.path.join(input_path, f"{data}_cohort.csv"))
       
        # If pretrain, use train-valid data of 5 seeds
        if 'pretrain' in self.train_task and self.split == 'train':
            self.train_valid_idcs = np.array([])
            self.test_idcs = np.array([])
            for seed in self.seed:
                col_name = f'split_{seed}'
                self.train_valid_idcs = np.append(self.train_valid_idcs, self.fold_file[self.fold_file[col_name] == 'train'][self.stay_id].values)
                self.train_valid_idcs = np.append(self.train_valid_idcs, self.fold_file[self.fold_file[col_name] == 'valid'][self.stay_id].values)
                self.test_idcs = np.append(self.test_idcs, self.fold_file[self.fold_file[col_name] == 'test'][self.stay_id].values)
            
            self.hit_idcs = np.unique(self.train_valid_idcs[~np.isin(self.train_valid_idcs, self.test_idcs)])

        # If scratch / finetune, use train/valid/test data of 1 seed each
        elif self.train_task in ['scratch', 'finetune']:
            col_name = f'split_{seed[0]}'
            self.hit_idcs = self.fold_file[self.fold_file[col_name] == self.split][self.stay_id].values

        logger.info(f'loaded {len(self.hit_idcs)} {self.split} samples')

    def __len__(self):
        raise NotImplementedError
    
    def __getitem__(self, index):
        raise NotImplementedError

    def mask_tokens(self, inputs: np.array, mask_vocab_size : int, 
                    mask_token : int, is_tokenized: bool, masked_indices=None):
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """
        inputs = torch.LongTensor(inputs)
        labels = inputs.clone()

        probability_matrix = torch.full(labels.shape, self.mlm_prob)
        
        if is_tokenized: #TODO: flatten 에도 적용하기
            special_tokens_mask = torch.tensor(
                self.tokenizer.get_special_tokens_mask(
                    labels, already_has_special_tokens=True
                ),
                dtype=torch.bool
            )
            probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        else: # Mask cls, sep for type/dpe ids (hierarchical)
            probability_matrix[0] = 0.0
            probability_matrix[-1] = 0.0

        if masked_indices is None:
            masked_indices = torch.bernoulli(probability_matrix).bool()

            while torch.equal(masked_indices, torch.zeros(len(masked_indices)).bool()): #TODO: 여기서 걸린거로 생각됨
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


    def hi_mask_tokens_wrapper(self, inputs: list, mask_vocab_size : int, 
                    mask_token : int, is_tokenized: bool, masked_indices=None):
        
        masked_events = []
        masked_labels = []
        for event in inputs:
            masked_event, masked_label = self.mask_tokens(event, mask_vocab_size,
                                                        mask_token, is_tokenized, masked_indices)
            masked_events.append(masked_event)
            masked_labels.append(masked_label)
        
        return masked_events, masked_labels


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
        emb_type,
        feature,
        input_path,
        split,
        train_task=None,
        ratio='100',
        pred_tasks = None,
        seed='2020',
        mask_list=['input', 'type', 'dpe'],
        mlm_prob=0.3,
        pretrain_task=None,
        **kwargs,
    ):
        super().__init__(
            data=data,
            emb_type=emb_type,
            feature=feature,
            input_path=input_path,
            split=split,
            train_task=train_task,
            ratio=ratio,
            seed=seed,
            **kwargs,
        )

        self.pred_tasks = pred_tasks
        self.pretrain_task = pretrain_task

        self.tokenizer = None

        if 'pretrain' in self.train_task:
            self.tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
        
        self.mask = None
        self.mask_list = mask_list
        self.mlm_prob = mlm_prob

        self.mask_vocab_size = { 
            'input' : 28996,
            'type' : 7, # 0-6 (pad, cls, sep 포함)
            'dpe' : 16 # 0 - 14, 15 mask (mimic3,eicu,mimic4)
            }            
        
        self.mask_token = { # 안쓰이는 토큰
            'input': 103,
            'type': 4, #TODO: need modification if use time bucket 
            'dpe': 15, 
        }

    def __len__(self):
        return len(self.hit_idcs)


class HierarchicalEHRDataset(EHRDataset):
    def __init__(self, max_seq_len=256, max_word_len=128, **kwargs):
        super().__init__(**kwargs)

        self.max_seq_len = max_seq_len
        self.max_word_len = max_word_len

        if 'pretrain' in self.train_task:
            if self.pretrain_task in ['simclr']:
                self.mask = self.hi_mask_tokens_wrapper

    def __len__(self):
        return len(self.hit_idcs)

    def collator(self, samples):
        samples = [s for s in samples if s['input_ids'] is not None]
        if len(samples) == 0:
            return {}
        
        input = dict()
        out = dict()

        if 'pretrain' in self.train_task and self.pretrain_task == "simclr":
            input['input_ids'] = [j.astype(np.int16) for s in samples for j in s['input_ids']]
            input['type_ids'] = [j.astype(np.int16) for s in samples for j in s['type_ids']]
            input['dpe_ids'] = [j.astype(np.int16) for s in samples for j in s['dpe_ids']]
        else:
            input['input_ids'] = [s['input_ids'] for s in samples]
            input['type_ids'] = [s['type_ids'] for s in samples]
            input['dpe_ids'] = [s['dpe_ids'] for s in samples]

        # Pad events for fixed event count per batch
        # Words per event already padded to max_word_len on data creation
        seq_sizes = []
        for s in input['input_ids']:
           seq_sizes.append(len(s))

        target_seq_size = min(max(seq_sizes), self.max_seq_len) 

        collated_input = dict()
        for k in input.keys():
            collated_input[k] = torch.zeros(
                (len(input['input_ids']), target_seq_size, self.max_word_len,)
            ).long()
 
        for i, seq_size in enumerate(seq_sizes):

            diff = seq_size - target_seq_size
            for k in input.keys():
                if k == 'input_ids':
                    prefix = 101
                elif k == "type_ids":
                    prefix = 5 
                elif k == "dpe_ids":
                    prefix = 0

                if diff == 0:
                    collated_input[k][i] = torch.from_numpy(input[k][i])
                elif diff < 0:
                    padding = np.zeros((-diff, self.max_word_len - 1,))
                    padding = np.concatenate(
                        [np.full((-diff, 1), fill_value=prefix), padding], axis=1
                    )
                    collated_input[k][i] = torch.from_numpy(
                            np.concatenate(
                            [input[k][i], padding], axis=0
                        )
                    )

        #TODO: 위와 코드 로직 합치기
        if 'pretrain' in self.train_task and self.pretrain_task == "simclr":
            collated_input['times'] = torch.from_numpy(np.stack([self.pad_to_max_size(j.astype(np.float64), target_seq_size) for s in samples for j in s['times']]))
        else:
            collated_input['times'] = torch.from_numpy(np.stack([self.pad_to_max_size(s['times'], target_seq_size) for s in samples]))

        out['net_input'] = collated_input

        if 'labels' in samples[0].keys():
            label_dict = dict()

            for k in samples[0]['labels'].keys():
                label_dict[k] = torch.stack([s['labels'][k] for s in samples])
            
            out['labels'] = label_dict
            
            out['icustays'] = [s['icustays'] for s in samples]

        return out

    def pad_to_max_size(self, sample, max_len):
        if len(sample) < max_len:
            sample = np.concatenate(
                [sample, np.zeros(max_len - len(sample), dtype=np.int16)]
            )
        return sample

    def __getitem__(self, index):
        fname = str(int(self.hit_idcs[index])) + '.pkl' #TODO: modify data pipeline cohort icustay type to int 
        data = pd.read_pickle(os.path.join(self.data_dir, fname))

        pack = {
            'input_ids': data[self.structure][:, 0, :],
            'type_ids': data[self.structure][:, 1, :],
            'dpe_ids': data[self.structure][:, 2, :],
            'times': data['time'],
        }
        
        # Labels
        if self.train_task in ["scratch", "finetune"]: 
            labels = dict()
            for task in self.pred_tasks:
                task_name = task.name
                task_prop = task.property
                task_class = task.num_classes
                labels[task_name] = self.fold_file[self.fold_file[self.stay_id] == self.hit_idcs[index]][task_name].values[0]
                if task_prop == "binary":
                    labels[task_name] = torch.tensor(labels[task_name], dtype=torch.float32)
                
                elif task_prop == "multi-label":
                    if task_name == 'diagnosis':
                        if labels[task_name] == -1 or labels[task_name]=='[]':
                            labels[task_name] = torch.zeros(task_class, dtype=torch.float32)
                            #labels[task_name] = torch.tensor([-100]*task_class, dtype=torch.float32)
                        else:
                            labels[task_name] = eval(labels[task_name]) #[3,5,2,1]
                            labels[task_name] = F.one_hot(torch.tensor(labels[task_name], dtype=torch.int64), num_classes=task_class).sum(dim=0).to(torch.float32)
     
                elif task_prop == "multi-class":
                    # Missing values are filled with -1 or Nan
                    if labels[task_name] == -1 or np.isnan(labels[task_name]):
                        labels[task_name] = torch.zeros(task_class).to(torch.float32)
                        #labels[task_name] = torch.torch.tensor(-100, dtype=torch.int64)
                    else:
                        labels[task_name] = F.one_hot(torch.tensor(labels[task_name]).to(torch.int64), num_classes=task_class).to(torch.float32)
                        #labels[task_name] = torch.tensor(labels[task_name], dtype=torch.int64)

            pack['labels'] = labels
            pack['icustays'] = self.hit_idcs[index]

        # Apply masking
        if self.mask:
            masked_indices = None

            input_ids = pack['input_ids']
            type_ids = pack['type_ids']
            dpe_ids = pack['dpe_ids']

            pack_no_pad = {
                'input_ids': [],
                'type_ids': [],
                'dpe_ids': [],
            }

            for input_id, type_id, dpe_id in zip(input_ids, type_ids, dpe_ids):
                input_id = input_id[input_id != 0]
                type_id = type_id[:len(input_id)]
                dpe_id = dpe_id[:len(input_id)]

                pack_no_pad['input_ids'].append(input_id)
                pack_no_pad['type_ids'].append(type_id)
                pack_no_pad['dpe_ids'].append(dpe_id)

            for victim in self.mask_list:
                is_tokenized = True if victim == 'input_ids' else False

                victim_ids, _ = self.mask( 
                    inputs=pack_no_pad[victim + "_ids"], 
                    mask_vocab_size=self.mask_vocab_size[victim],
                    mask_token=self.mask_token[victim],
                    is_tokenized=is_tokenized,
                    masked_indices=masked_indices
                )

                victim_ids = [self.pad_to_max_size(victim_id, self.max_word_len) for victim_id in victim_ids]
                victim_ids = np.stack(victim_ids).astype(np.int16)

                pack[victim + "_ids"] = victim_ids

            # Halve events when SimCLR pre-training
            if 'pretrain' in self.train_task and self.pretrain_task == 'simclr':
                for victim in self.mask_list:
                    half = int(len(pack[victim + "_ids"]) / 2)
                    pack[victim + "_ids"] = np.array([pack[victim + "_ids"][:half], pack[victim + "_ids"][half:]], dtype=object)
                pack['times'] = np.array([pack['times'][:half], pack['times'][half:]], dtype=object)

        return pack


class FlattenEHRDataset(EHRDataset):
    def __init__(self, max_seq_len=8192, **kwargs):
        super().__init__(**kwargs)

        self.max_seq_len = max_seq_len

        if 'pretrain' in self.train_task:
            if self.pretrain_task in ['mlm', 'spanmlm', 'simclr']:
                self.mask = self.span_masking if self.pretrain_task=='spanmlm' else self.mask_tokens

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

        if self.pretrain_task == "simclr": # Make positive pair adjacent (B -> 2*B)
            input['input_ids'] = [j for s in samples for j in s['input_ids']]
            input['type_ids'] = [j for s in samples for j in s['type_ids']]
            input['dpe_ids'] = [j for s in samples for j in s['dpe_ids']]
        else:
            input['input_ids'] = [s['input_ids'] for s in samples]
            input['type_ids'] = [s['type_ids'] for s in samples]
            input['dpe_ids'] = [s['dpe_ids'] for s in samples]

        if self.mask:
            for victim in self.mask_list:
                if self.pretrain_task == "simclr":
                    input[victim+'_label']= [j for s in samples for j in s[victim+'_label']]
                else:
                    input[victim+'_label']= [s[victim+'_label'] for s in samples]

        sizes = [len(s) for s in input['input_ids']]
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
        fname = str(self.hit_idcs[index]) + '.npy'

        input_ids = np.load(os.path.join(self.data_dir, 'input_ids', fname), allow_pickle=True)
        type_ids = np.load(os.path.join(self.data_dir, 'type_ids', fname), allow_pickle=True)
        dpe_ids = np.load(os.path.join(self.data_dir, 'dpe_ids', fname), allow_pickle=True)
        
        if self.train_task in['scratch', 'finetune']:
            label = self.label[index]

            out = {
                'input_ids': input_ids,
                'type_ids': type_ids,
                'dpe_ids': dpe_ids,
                'label': label,
            }
        else:
            out = {
                'input_ids': input_ids,
                'type_ids': type_ids,
                'dpe_ids': dpe_ids,
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

        if self.pretrain_task == "simclr":
            unique, counts = np.unique(type_ids, return_counts=True)
            half_event_count = int(dict(zip(unique, counts))[4]/2)
            half_len = [i_i for i_i, n_i in enumerate(type_ids) if n_i == 4][half_event_count - 1]
            for victim in ['input', 'type', 'dpe']:
                out[victim + '_ids'] = np.array([out[victim + '_ids'][:half_len+1], out[victim + '_ids'][half_len+1:]])
                if victim in self.mask_list:
                    out[victim + '_label'] = np.array([out[victim + '_label'][:half_len+1], out[victim + '_label'][half_len+1:]])


        return out

