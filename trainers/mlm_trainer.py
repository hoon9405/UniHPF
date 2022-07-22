import os
import torch
import torch.nn as nn
import wandb
from itertools import chain
from typing import Any, Dict, List
import logging

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

# # trainer에서 해결해줄것
# args.src_data = (
#     args.src_data.split('_')
#     if args.train_type in ['transfer', 'pooled']
#     else args.src_data
# )


class MLMTranier(object):
    def __init__(self, args, seed):
        self.args = args
        self.dp = args.dp
        self.world_size = args.world_size

        self.seed = seed
        self.batch_size = args.batch_size
        base_dir = (
            f'src_data_{args.src_data}_icu_type_{args.icu_type}_bert_model_{args.bert_model}'
            f'_pred_model_{args.pred_model}_type_token_{args.type_token}_dpe_{args.dpe}_lr_{args.lr}_seed_{seed}'
        )
        
        pred_filename = (
            f'text_post_proj_{args.text_post_proj}_pred_pooling_{args.pred_pooling}_map_layers_{args.map_layers}'
            f'_n_layers_{args.n_layers}_n_heads_{args.n_heads}_embed_dim_{args.embed_dim}_pred_dim_{args.pred_dim}_dropout_{args.dropout}'
        )
  
        pretrain_filename = (
            f'mlm_prob_{args.mlm_prob}_mask_list_{" ".join(args.mask_list)}'
        )


        base_path = os.path.join(
            args.output_path, 
            args.train_task,
            args.input2emb_model+ '_'+ args.feature
        ) 

        task_path = {
            'predict': (
                os.path.join(args.pred_target, args.train_type), 
                os.path.join(base_dir, pred_filename)
                ),
            'pretrain': (
                args.pretrain_task, 
                os.path.join(base_dir, pretrain_filename)
                )
        }

        (suffix_path, self.filename) = task_path[args.train_task]
        
        self.path = os.path.join(
            base_path, 
            suffix_path,
            )

        self.train_data = args.src_data
      
        self.datasets = dict() 
        self.early_stopping_dict = dict()
        
        data_types = ['train'] + sorted(self.args.valid_subsets*len(self.args.eval_data))
        data_names = [self.train_data] + self.args.eval_data*len(self.args.valid_subsets)
        vocab = self.train_data
        logger.info(data_types, data_names)

        for split, data in zip(data_types, data_names):
            self.load_dataset(split, data, vocab, self.seed)
      
        # save_directory
        for data in args.eval_data:
            if not os.path.exists(self.save_path(data)):
                os.makedirs(self.save_path(data))


    def save_path(self, data): 
        return os.path.join(self.path, data, self.filename)  


    def load_dataset(self, split, dataname, vocab, seed) -> None:
        structure = 'hi' if '_' in self.args.input2emb_model else 'uni'
        if structure == 'hi':
            dataset = HierarchicalEHRDataset(
                data=dataname,
                input_path=self.args.input_path,
                split=split,
                vocab=vocab,
                concept=self.args.input2emb_model,
                column_embed=self.args.column_embed,
                feature=self.args.feature,
                train_task=self.args.train_task,
                pretrain_task=self.args.pretrain_task,
                ratio=self.args.ratio,
                icu_type=self.args.icu_type,
                pred_target=self.args.pred_target,
                split_method=self.args.split_method,
                seed=self.args.seed,
                mask_list=self.args.mask_list,
                mlm_prob=self.args.mlm_prob,
                max_word_len=self.args.max_word_len,
                max_seq_len=self.args.max_seq_len,
            )
        elif structure == 'uni':
            dataset = UnifiedEHRDataset(
                data=dataname,
                input_path=self.args.input_path,
                split=split,
                vocab=vocab,
                concept=self.args.input2emb_model,
                column_embed=self.args.column_embed,
                feature=self.args.feature,
                train_task=self.args.train_task,
                pretrain_task=self.args.pretrain_task,
                ratio=self.args.ratio,
                icu_type=self.args.icu_type,
                pred_target=self.args.pred_target,
                split_method=self.args.split_method,
                seed=self.args.seed,
                mask_list=self.args.mask_list,
                mlm_prob=self.args.mlm_prob,
                max_seq_len=self.args.max_seq_len,
            )
        else:
            raise NotImplementedError
            #datasets_list.append(_dataset)
        #dataset = ConcatDataset(datasets_list)
        if not split in self.datasets.keys():
            self.datasets[split] = dict()

        self.datasets[split][dataname] = dataset

    def dataloader_set(self, dataset, world_size, batch_size):
        if 1 < world_size and not self.dp:
            sampler = DistributedSampler(dataset)
            data_loader = DataLoader(
                dataset, 
                collate_fn=dataset.collator,
                batch_size = batch_size, 
                num_workers=8, 
                sampler = sampler
            )
        else:
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
        torch.cuda.set_device(rank)

        if torch.cuda.is_available():
            dist.all_reduce(torch.zeros(1).cuda())

    @property
    def data_parallel_world_size(self):
        if self.args.world_size == 1 or self.dp:
            return 1
        return distributed_utils.get_data_parallel_world_size()
    
    @property
    def data_parallel_process_group(self):
        return distributed_utils.get_data_parallel_group()
    
    @property
    def data_parallel_rank(self):
        if self.args.world_size == 1 or self.dp:
            return 0
        return distributed_utils.get_data_parallel_rank()
    
    @property
    def is_data_parallel_master(self):
        return self.data_parallel_rank == 0

    def train(self):
        if 1 < self.args.world_size and not self.dp:
            mp.spawn(self.distributed_train,
                    args=(self.args.world_size,),
                    nprocs=self.args.world_size,
                    join=True)
        else:
            self.distributed_train(self.args.device_num, self.args.world_size)
        
    def distributed_train(self, rank, world_size):
        if 1 < world_size and not self.dp:
            self.setup_dist(rank, world_size)
        
        # wandb
        if not self.args.debug and (not self.dp or self.is_data_parallel_master):
            wandb.init(
                project=self.args.wandb_project_name,
                entity="kaggle-wandb",
                config=self.args,
                reinit=True
            )
            wandb.run.name = self.args.wandb_run_name

        model = models.build_model(self.args)
         #resume code 
        
        #transfer
        if self.args.train_type =='transfer':
            load_path = self.save_path(self.args.src_data).replace('transfer', 'single')
            print('transfer learning, load model from : ', load_path)
            state_dict = torch.load()['model_state_dict']
            model.load_state_dict(state_dict, strict = False)
            self.train_data = self.args.eval_data # finetune train data

        #num_params = count_parameters(model) 
        # print(
        #     'Model build :' , 
        #     'Number of parameters: ', num_params
        # )

        if 1 < world_size and not self.dp:
            device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
            self.model = model.to(device)
            self.model = DistributedDataParallel(self.model, device_ids = [rank], find_unused_parameters=False)
        else:       
            self.model = nn.DataParallel(model, device_ids=self.args.device_ids).to('cuda')

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr)
       
        load_dict= None
        load_path = self.save_path(self.args.eval_data[0]) + '.pkl'
        print('load target path : ', load_path)
        if os.path.exists(load_path):
            print('resume training start ! load checkpoint from : ', load_path)
            load_dict = torch.load(load_path, map_location='cpu')
            model_state_dict = load_dict['model_state_dict']
            optimizer_state_dict = load_dict['optimizer_state_dict']

            self.model.load_state_dict(model_state_dict, strict = True)
            self.optimizer.load_state_dict(optimizer_state_dict)
        else:
            print('training from scratch start !')   

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
            
        start_epoch = load_dict['n_epoch'] if load_dict is not None else 1
        for n_epoch in range(start_epoch, self.args.epochs + 1):
            logger.info('[Epoch] {}'.format(n_epoch))
            self.model.train()
            
            for iter, sample in tqdm.tqdm(enumerate(data_loader)):
           
                self.optimizer.zero_grad(set_to_none=True)
                output = self.model(**sample['net_input'])
                target = self.model.module.get_targets(sample)
                
                for victim in self.args.mask_list:
                    target[victim+'_label'] = torch.where(
                        target[victim+'_label'] == 0, -100, target[victim+'_label']
                    )

                loss = self.criterion(output, target)
                loss.backward()
                
                self.optimizer.step()

                logging_outputs = {
                    'loss': loss.detach().cpu()
                }
                for victim in self.args.mask_list:
                    preds = torch.argmax(output[victim+'_ids'], dim=2).view(-1).detach().cpu()
                    target_label = target[victim+'_label'].view(-1)
                    mask_idcs = (target_label != -100) & (target_label != 0)
                    total = mask_idcs.sum()
                    correct = (preds[mask_idcs] == target_label[mask_idcs]).sum().float()

                    logging_outputs[victim+'_ids_correct'] = correct
                    logging_outputs[victim+'_ids_total'] = total
                # logging_outputs['total'] = total

                if self.world_size > 1 and not self.dp:
                    _logging_outputs = self._all_gather_list_sync([logging_outputs])

                    for key in logging_outputs.keys():
                        logging_outputs[key] = float(sum(log[key] for log in _logging_outputs))

                        if 'loss' in key:
                            logging_outputs[key] /= len(_logging_outputs)
                        else:
                            logging_outputs[key] = int(logging_outputs[key])

                    del _logging_outputs

                logging_outputs['total'] = {
                    victim+'_ids': logging_outputs.pop(victim+'_ids_total')
                    for victim in self.args.mask_list
                }
                logging_outputs['correct'] = {
                    victim+'_ids': logging_outputs.pop(victim+'_ids_correct')
                    for victim in self.args.mask_list
                }
                self.metric(**logging_outputs) # iter_uddate

            with torch.no_grad():
                train_metric_dict = self.metric.get_epoch_dict(
                    len(data_loader)
                )
    
            log_dict = utils.log_from_dict(train_metric_dict, 'train', self.train_data, n_epoch)
            
            if not self.args.debug and (not self.dp or self.is_data_parallel_master):
                wandb.log(log_dict) 
            
            break_token = self.evaluation(n_epoch)
      
            if break_token:
                break
            
        self.final_epoch = n_epoch
        print(f'train finished at epoch {n_epoch}')

        if self.is_data_parallel_master and self.args.debug == False:
            wandb.finish(0)

        if self.world_size > 1 and not self.dp:
            dist.destroy_process_group()

    def inference(self, data_loader, data_type, data_name, n_epoch):
        self.model.eval()
        with torch.no_grad():
            for iter, sample in tqdm.tqdm(enumerate(data_loader)):
                self.optimizer.zero_grad(set_to_none=True)
           
                output = self.model(**sample['net_input'])
                target = self.model.module.get_targets(sample)
            
                loss = self.criterion(output, target)
                
                logging_outputs = {
                    'loss': loss.detach().cpu()
                }
                for victim in self.args.mask_list:
                    preds = torch.argmax(output[victim+'_ids'], dim=2).view(-1).detach().cpu()
                    target_label = target[victim+'_label'].view(-1)
                    mask_idcs = (target_label != -100) & (target_label != 0)
                    total = mask_idcs.sum()
                    correct = (preds[mask_idcs] == target_label[mask_idcs]).sum().float()

                    logging_outputs[victim+'_ids_correct'] = correct
                    logging_outputs[victim+'_ids_total'] = total
                # logging_outputs['total'] = total

                if self.world_size > 1 and not self.dp:
                    _logging_outputs = self._all_gather_list_sync([logging_outputs])

                    for key in logging_outputs.keys():
                        logging_outputs[key] = float(sum(log[key] for log in _logging_outputs))

                        if 'loss' in key:
                            logging_outputs[key] /= len(_logging_outputs)
                        else:
                            logging_outputs[key] = int(logging_outputs[key])

                    del _logging_outputs

                logging_outputs['total'] = {
                    victim+'_ids': logging_outputs.pop(victim+'_ids_total')
                    for victim in self.args.mask_list
                }
                logging_outputs['correct'] = {
                    victim+'_ids': logging_outputs.pop(victim+'_ids_correct')
                    for victim in self.args.mask_list
                }
                self.metric(**logging_outputs) # iter_uddate

            metric_dict = self.metric.get_epoch_dict(
                len(data_loader)
            )

        log_dict = utils.log_from_dict(metric_dict, data_type, data_name, n_epoch)

        if not self.args.debug and (not self.dp or self.is_data_parallel_master):
            wandb.log(log_dict) 
        
        return metric_dict


    def evaluation(self, n_epoch):
        self.model.eval()
        break_token = False
            
        for data_name, dataset in self.datasets['valid'].items():
            data_loader = self.dataloader_set(
                dataset, 
                self.world_size,
                self.args.batch_size
            )
            metric_dict = self.inference(data_loader, 'valid', data_name, n_epoch)

            if self.early_stopping_dict[data_name](metric_dict[self.metric.update_target]):
                if not self.dp or self.is_data_parallel_master:
                    best_model_path = self.save_path(data_name)
                    utils.model_save(self.model, best_model_path, n_epoch, self.optimizer)

            if self.early_stopping_dict[data_name].early_stop:
                logger.info(f'data_name : {data_name}, Early stopping!')
                del self.datasets['valid'][data_name]

        if self.datasets['valid'] == {}:
            break_token = True
            logger.info(f'all valid finished at {n_epoch}')
    
        return break_token
  

  # pooled 일때 test 손 봐야함.

    def test(self, load_checkpoint=None):
        print('test start .. ')  
        
        #state_dict = {k.replace('module.',''): state_dict[k] for k in state_dict if (not 'embedding' in k and not 'bert_embed' in k)}
        if load_checkpoint is not None:
            load_path = load_checkpoint
            state_dict = torch.load(load_path, map_location='cpu')['model_state_dict']
            self.model.load_state_dict(state_dict, strict = False)
            data_loader = self.dataloader_set(
                self.datasets['test'],
                1, 
                self.args.batch_size
            )

            metric_dict = self.inference(
            data_loader, 'test', data_name, self.final_epoch
            )
            
           
        else:
            for data_name, dataset in self.datasets['test']:
                load_path = self.save_path(data_name)

                state_dict = torch.load(load_path, map_location='cpu')['model_state_dict']
                self.model.load_state_dict(state_dict, strict = False)
                data_loader = self.dataloader_set(
                    dataset, 
                    1, 
                    self.args.batch_size
                )

                metric_dict = self.inference(
                data_loader, 'test', data_name, self.final_epoch
                )
    
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
                    max_size = getattr(self, "all_gather_list_size", 65536),
                    group = self.data_parallel_process_group
                )
            )
        )
        logging_outputs = results[0]
        logging_outputs = list(chain.from_iterable(logging_outputs))
        return logging_outputs
