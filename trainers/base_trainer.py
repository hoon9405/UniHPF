import os
import wandb
from itertools import chain
from typing import Any, Dict, List
import logging

import numpy as np
import torch
import torch.nn as nn

from metrics.metric import PredMetric, PretrainMetric
import models

import utils.trainer_utils as utils
import utils.distributed_utils as distributed_utils
from torch.utils.data import DataLoader, ConcatDataset
from loss.loss import PredLoss, PretrainLoss
from datasets.base_dataset import HierarchicalEHRDataset, UnifiedEHRDataset
import tqdm

import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist

logger = logging.getLogger(__name__)

class BaseTrainer:
    def __init__(self, args, seed):
        self.args = args
        self.seed = seed
        self.batch_size = args.batch_size
        self.save_dir = args.save_dir
        self.save_prefix = args.save_prefix
        
        if args.train_type =='transfer':
            self.train_data = args.eval_data[0]
        else:
            self.train_data = args.src_data

        self.datasets = dict()
        self.early_stopping_dict = dict()


        data_types = ['train'] + sorted(self.args.valid_subsets*len(self.args.eval_data))
        data_names = [self.train_data] + self.args.eval_data*len(self.args.valid_subsets)
        vocab = self.train_data
        
        logger.info(data_types, data_names)
        for split, data in zip(data_types, data_names):
            # logger.info('split : ', split, 'data_name : ', data)
            if not split in self.datasets.keys():
                self.datasets[split] = dict()
            self.datasets[split][data] = self.load_dataset(split, data, vocab, self.seed)
        

    def load_dataset(self, split, dataname, vocab, seed) -> None: 
        if self.args.structure == 'hi':
            dataset = HierarchicalEHRDataset(
                data=dataname,
                input_path=self.args.input_path,
                split=split,
                vocab=vocab,
                concept=self.args.input2emb_model,
                feature=self.args.feature,
                train_task=self.args.train_task,
                pretrain_task=self.args.pretrain_task,
                ratio=self.args.ratio,
                pred_target=self.args.pred_target,
                seed=self.args.seed,
                mask_list=self.args.mask_list,
                mlm_prob=self.args.mlm_prob,
                max_word_len=self.args.max_word_len,
                max_seq_len=self.args.max_seq_len,
            )
        elif self.args.structure == 'fl':
            dataset = UnifiedEHRDataset(
                data=dataname,
                input_path=self.args.input_path,
                split=split,
                vocab=vocab,
                concept=self.args.input2emb_model,
                feature=self.args.feature,
                train_task=self.args.train_task,
                pretrain_task=self.args.pretrain_task,
                ratio=self.args.ratio,
                pred_target=self.args.pred_target,
                seed=self.args.seed,
                mask_list=self.args.mask_list,
                mlm_prob=self.args.mlm_prob,
                max_seq_len=self.args.max_seq_len,
            )

        else:
            raise NotImplementedError(self.model_type)
 
        return dataset

    def dataloader_set(self, dataset, world_size, batch_size):
        if 1 < world_size:
            self.sampler = DistributedSampler(dataset)
            data_loader = DataLoader(
                dataset, 
                collate_fn=dataset.collator,
                batch_size=batch_size,
                num_workers=8,
                sampler=self.sampler,
                pin_memory=True,
            )
        else:
            self.sampler=None
            data_loader = DataLoader(
                dataset, 
                collate_fn=dataset.collator,
                batch_size=batch_size, 
                num_workers=8,
                shuffle=True,
                pin_memory=True,
            )
        return data_loader

    def setup_dist(self, rank, world_size):
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = self.args.port
        dist.init_process_group(
            backend = "nccl",
            rank = rank,
            world_size = world_size
        )

    @property
    def data_parallel_world_size(self):
        if self.args.world_size == 1:
            return 1
        return distributed_utils.get_data_parallel_world_size()
    
    @property
    def data_parallel_process_group(self):
        return distributed_utils.get_data_parallel_group()
    
    @property
    def data_parallel_rank(self):
        if self.args.world_size == 1:
            return 0
        return distributed_utils.get_data_parallel_rank()
    
    @property
    def is_data_parallel_master(self):
        return self.data_parallel_rank == 0

    def train(self):
        if 1 < self.args.world_size:
            mp.spawn(self.distributed_train,
                    args=(self.args.world_size,),
                    nprocs=self.args.world_size,
                    join=True)
        else:
            self.distributed_train(self.args.device_num, self.args.world_size)    

    def distributed_train(self, rank, world_size):
        if 1 < world_size:
            self.setup_dist(rank, world_size)
            torch.cuda.set_device(rank)

        # Wandb init
        if self.is_data_parallel_master and not self.args.debug:
            wandb.init(
                project=self.args.wandb_project_name,
                entity="kaggle-wandb",
                config=self.args,
                reinit=True
            )
            wandb.run.name = self.args.wandb_run_name

        model = models.build_model(self.args)
        num_params = utils.count_parameters(model)
        
        if 1 < world_size:
            device = torch.device(f'cuda:{rank}' if torch.cuda.is_available() else 'cpu')
            self.model = model.to(device)
            self.model = DistributedDataParallel(self.model, device_ids=[rank], find_unused_parameters=False)
        else:
            self.model = nn.DataParallel(model, device_ids=self.args.device_ids).to('cuda')
            
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr)  
                  
        if self.args.train_type =='transfer':
            load_path = self.save_path(self.args.src_data).replace('transfer', 'single') +'.pkl'
            logger.info('transfer learning, load model from : ', load_path)
            state_dict = torch.load(load_path, map_location='cpu')['model_state_dict']
            if self.args.input2emb_model.startswith('codeemb'):
                state_dict = {
                    k: v for k,v in state_dict.items() if (
                        ('input2emb' not in k) and ('pos_enc' not in k)
                    )
                }
            else:   
                state_dict = {
                        k: v for k,v in state_dict.items() if (
                            'pos_enc' not in k
                        )
                    }       
            
            self.model.load_state_dict(state_dict, strict = False)
            print('transfer learning mode -- ratio : ', self.args.ratio)

        logger.info(f'device_ids = {self.args.device_ids}')

        data_loader = self.dataloader_set(
            self.datasets['train'][self.train_data],
            world_size,
            self.args.batch_size
        )

        Loss = PredLoss if self.args.train_task =='predict' else PretrainLoss
        self.criterion = Loss(self.args)

        metric = PredMetric if self.args.train_task =='predict' else PretrainMetric
        self.metric = metric(self.args)

        for data in self.datasets['valid'].keys():
            self.early_stopping_dict[data] = utils.EarlyStopping(
                patience=self.args.patience, 
                compare=self.metric.compare,
                metric=self.metric.update_target
            )

        break_token= False
        start_epoch = load_dict['n_epoch'] if load_dict is not None else 1
        if not self.args.ratio=='0':
            for n_epoch in range(start_epoch, self.args.epochs + 1):
                
                logger.info('[Epoch] {}'.format(n_epoch))
                self.model.train()

                for iter, sample in tqdm.tqdm(enumerate(data_loader)):
                    self.optimizer.zero_grad(set_to_none=True)
                    
                    output = self.model(**sample['net_input'])
                    target = self.model.module.get_targets(sample)
                    
                    loss = self.criterion(output, target)
                    loss.backward()
                    
                    self.optimizer.step()

                    preds = torch.sigmoid(output['pred_output']).view(-1).detach()
                    truths = target.view(-1)

                    logging_outputs = {
                        'loss': float(loss.detach().cpu()),
                        'preds': preds,
                        'truths': truths,
                    }
                    if self.data_parallel_world_size > 1:
                        _logging_outputs = self._all_gather_list_sync([logging_outputs])

                        for key in logging_outputs.keys():
                            if key == 'loss':
                                logging_outputs[key] = float(sum(log[key] for log in _logging_outputs))
                            elif key in ['preds', 'truths']:
                                logging_outputs[key] = np.concatenate(
                                    [log[key].numpy() for log in _logging_outputs]
                                )
                            else:
                                raise NotImplementedError(
                                    "What else?"
                                )
                        del _logging_outputs
                    else:
                        logging_outputs['preds'] = logging_outputs['preds'].cpu().numpy()
                        logging_outputs['truths'] = logging_outputs['truths'].cpu().numpy()
                    
                    self.metric(**logging_outputs) # iter_uddate
                    
                with torch.no_grad():
                    train_metric_dict = self.metric.get_epoch_dict(
                        len(data_loader)
                    )
        
                log_dict = utils.log_from_dict(train_metric_dict, 'train', self.train_data, n_epoch)
                
                if self.is_data_parallel_master and self.args.debug == False:
                    wandb.log(log_dict) 

                break_token = self.evaluation(n_epoch)
        
                if break_token:
                    break
        
        self.test(n_epoch)
        print(f'test finished at epoch {n_epoch}')

        if self.is_data_parallel_master and self.args.debug == False:
            wandb.finish(0)

        if self.data_parallel_world_size > 1:
            dist.destroy_process_group()


    def inference(self, data_loader, data_type, data_name, n_epoch):
        self.model.eval()
        with torch.no_grad():
            for iter, sample in tqdm.tqdm(enumerate(data_loader)):
                self.optimizer.zero_grad(set_to_none=True)
                
                output = self.model(**sample['net_input'])
                target = self.model.module.get_targets(sample)
              
                loss = self.criterion(output, target)
                
                preds = torch.sigmoid(output['pred_output']).view(-1).detach()
                truths = target.view(-1)

                logging_outputs = {
                    'loss': float(loss.detach().cpu()),
                    'preds': preds,
                    'truths': truths,
                }
                if self.data_parallel_world_size > 1:
                    _logging_outputs = self._all_gather_list_sync([logging_outputs])

                    for key in logging_outputs.keys():
                        if key == 'loss':
                            logging_outputs[key] = float(sum(log[key] for log in _logging_outputs))
                        elif key in ['preds', 'truths']:
                            logging_outputs[key] = np.concatenate(
                                [log[key].numpy() for log in _logging_outputs]
                            )
                        else:
                            raise NotImplementedError(
                                "What else?"
                            )
                    del _logging_outputs
                else:
                    logging_outputs['preds'] = logging_outputs['preds'].cpu().numpy()
                    logging_outputs['truths'] = logging_outputs['truths'].cpu().numpy()
                
                self.metric(**logging_outputs) # iter_uddate

            metric_dict = self.metric.get_epoch_dict(
                len(data_loader)
            )

        log_dict = utils.log_from_dict(metric_dict, data_type, data_name, n_epoch)

        if self.is_data_parallel_master and self.args.debug == False:
            wandb.log(log_dict) 
        
        return metric_dict


    def evaluation(self, n_epoch):
        self.model.eval()
        break_token = False
        stop_list = []

        for data_name, dataset in self.datasets['valid'].items():
            data_loader = self.dataloader_set(
                dataset,
                self.data_parallel_world_size,
                self.args.batch_size
            )
            metric_dict = self.inference(data_loader, 'valid', data_name, n_epoch)
            
            if self.early_stopping_dict[data_name](metric_dict[self.metric.update_target]):
                if self.is_data_parallel_master:
                    logger.info(
                        "Saving checkpoint to {}".format(
                            os.path.join(self.save_dir, self.save_prefix + "_best.pt")
                        )
                    )
                    best_model_path = self.save_path(self.save_dir + self.save_prefix +'_best.pt')
                    utils.model_save(self.model, best_model_path, n_epoch, self.optimizer)

            if self.early_stopping_dict[data_name].early_stop:
                logger.info(f'data_name : {data_name}, Early stopping!')
                stop_list.append(data_name)
        
        for data_name in stop_list:
            del self.datasets['valid'][data_name]

        if self.datasets['valid'] == {}:
            break_token = True
            logger.info(f'all valid finished at {n_epoch}')
        
        return break_token
        
    def test(self, n_epoch, load_checkpoint=None):
        print('test start .. ')
        if load_checkpoint is not None:
            for data_name, dataset in self.datasets['test'].items():

                load_path = load_checkpoint
                state_dict = torch.load(load_path, map_location='cpu')['model_state_dict']
                self.model.load_state_dict(state_dict, strict = True)

                data_loader = self.dataloader_set(
                    dataset,
                    self.data_parallel_world_size,
                    self.args.batch_size
                )
                metric_dict = self.inference(
                    data_loader, 'test', data_name, n_epoch
                )
        else:
            for data_name, dataset in self.datasets['test'].items():
                load_path = self.save_path(data_name)+'.pkl'
                state_dict = torch.load(load_path, map_location='cpu')['model_state_dict']
                self.model.load_state_dict(state_dict, strict = True)

                data_loader = self.dataloader_set(
                    dataset,
                    self.data_parallel_world_size,
                    self.args.batch_size
                )
                metric_dict = self.inference(data_loader, 'test', data_name, n_epoch)
        
        return metric_dict


    def _all_gather_list_sync(
        self,
        logging_outputs: List[Dict[str, Any]],
        ignore = False
    ):
        """
        Sync logging outputs across workers. all_gather_list_sync is
        suifeature when logging outputs are complex types.
        """
        if ignore:
            logging_outputs = []
        results = list(
            zip(
                *distributed_utils.all_gather_list(
                    [logging_outputs],
                    max_size = getattr(self, "all_gather_list_size", 1048576),
                    group = self.data_parallel_process_group
                )
            )
        )
        logging_outputs = results[0]
        logging_outputs = list(chain.from_iterable(logging_outputs))
        return logging_outputs
