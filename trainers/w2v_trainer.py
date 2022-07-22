import os
import logging
import math
from itertools import chain
from typing import Any, Dict, List
import wandb

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from utils import utils
import utils.trainer_utils as tr_utils
import utils.distributed_utils as dist_utils
from datasets.base_dataset import HierarchicalEHRDataset, UnifiedEHRDataset
from metrics.metric import W2VMetric

import models

import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist

from tqdm import tqdm

logger = logging.getLogger(__name__)

class W2VTrainer(object):
    def __init__(self, args, seed):
        self.args = args
        
        self.seed = seed
        self.batch_size = args.batch_size

        self.loss_weights = [0.1, 5]
        self.max_loss_weights = [0.1, 5]
        self.min_loss_weights = [0.005, 0.25]
        self.log_interval = 25
        self.epoch_save_interval = 50
        self.epoch_validate_from = 10
        self.no_validation = True

        self.args.loss_weights = self.loss_weights

        self.codebook_negatives = args.codebook_negatives

        base_dir = (
            f'src_data_{args.src_data}_icu_type_{args.icu_type}_bert_model_{args.bert_model}'
            f'_pred_model_{args.pred_model}_type_token_{args.type_token}_dpe_{args.dpe}_lr_{args.lr}_seed_{seed}'
        )
        
        pred_filename = (
            f'text_post_proj_{args.text_post_proj}_pred_pooling_{args.pred_pooling}_map_layers_{args.map_layers}'
            f'_n_layers_{args.n_layers}_n_heads_{args.n_heads}_embed_dim_{args.embed_dim}_pred_dim_{args.pred_dim}_dropout_{args.dropout}'
        )
  
        pretrain_filename = (
            f'{str(self.loss_weights[0])}_{str(self.loss_weights[1])}'
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

        vocab = self.train_data

        self.datasets['train'] = self.load_dataset(split='train', vocab=vocab, seed=seed)
        for v in args.valid_subsets:
            self.datasets[v] = self.load_dataset(split=v, vocab=vocab, seed=seed)

        # save_directory
        for data in args.eval_data:
            rindex = self.save_path(data).rfind('/')
            if not os.path.exists(self.save_path(data)[:rindex]):
                os.makedirs(self.save_path(data)[:rindex])

        # self.save_dir = os.path.join(
        #     args.output_path, 'w2v', args.input2emb_model, args.feature
        # )
        # if not os.path.exists(self.save_dir):
        #     os.makedirs(self.save_dir)

    def save_path(self, data): 
        return os.path.join(self.path, data, self.filename) 
    
    def load_dataset(self, split, vocab, seed):
        structure = 'hi' if '_' in self.args.input2emb_model else 'uni'
        if structure =='hi':
            dataset = HierarchicalEHRDataset(
                data=self.args.src_data,
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
        else:
            raise AttributeError(
                'pre-train w2v requires the structure of EHR to be hierarchical. '
                'current structure: {}'.format(structure)
            )
        
        return dataset

    def dataloader_set(self, dataset, world_size, batch_size):
        if 1 < world_size:
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
        if self.args.world_size == 1:
            return 1
        return dist_utils.get_data_parallel_world_size()
    
    @property
    def data_parallel_process_group(self):
        return dist_utils.get_data_parallel_group()
    
    @property
    def data_parallel_rank(self):
        if self.args.world_size == 1:
            return 0
        return dist_utils.get_data_parallel_rank()
    
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

        # wandb
        if not self.args.debug and self.is_data_parallel_master:
            wandb.init(
                project=self.args.wandb_project_name,
                entity="kaggle-wandb",
                config=self.args,
                reinit=True
            )
            wandb.run.name = self.args.wandb_run_name

        model = models.build_model(self.args)
    

        if 1 < world_size:
            device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
            self.model = model.to(device)
            self.model = DistributedDataParallel(self.model, device_ids = [rank], find_unused_parameters=False)
        else:
            self.model = nn.DataParallel(model, device_ids=self.args.device_ids).to('cuda')

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr, weight_decay=0.01)

        start_epoch = 0
        if self.args.resume:
            resume_path = self.save_path(self.args.eval_data[0])
            logger.info('resume training from {}'.format(resume_path))

            state_dict = torch.load(resume_path+'.pkl', 'cpu')
            start_epoch = state_dict['n_epoch']
            self.model.load_state_dict(state_dict['model_state_dict'], strict=True)
            self.optimizer.load_state_dict(state_dict['optimizer_state_dict'])

            del state_dict

        data_loader = self.dataloader_set(
            self.datasets['train'], 
            world_size,  
            self.args.batch_size
        )

        self.criterion = nn.CrossEntropyLoss(reduction='sum')
        self.metric = W2VMetric(target='acc')

        for v in self.args.valid_subsets:
            if 'valid' in v:
                if not self.early_stopping_dict:
                    self.early_stopping_dict[v] = tr_utils.EarlyStopping(
                        patience=self.args.patience,
                        compare=self.metric.compare,
                        metric=self.metric.update_target
                    )
                else:
                    raise AttributeError(
                        "please ensure that there are only one valid subset in --valid_subsets. "
                        "--valid_subsets: {}".format(self.args.valid_subsets)
                    )

        inner_metric = W2VMetric()
        for n_epoch in range(start_epoch, self.args.epochs + 1):
            self.model.train()
            inner_metric.reset()

            with tqdm(total=len(data_loader)) as pbar:
                for i, sample in enumerate(data_loader):
                    sample = self._prepare_sample(sample)

                    self.optimizer.zero_grad(set_to_none=True)

                    # #####################################
                    # state_dict = torch.load('/home/edlab/ghhur/Pretrained_DescEmb/data/output/pretrain/descemb_bert_whole/w2v/eicu/src_data_eicu_icu_type_ticu_bert_model_bert_tiny_pred_model_transformer_type_token_True_dpe_True_lr_5e-05_seed_2020/mlm_prob_0.3_mask_list_input type.pkl', 'cpu')
                    # self.model.load_state_dict(state_dict['model_state_dict'])
                    # with torch.no_grad():
                    #     net_output = self.model(**sample['net_input'])
                    #     breakpoint()
                    #     print()
                    # ########################################
                    net_output = self.model(**sample['net_input'])

                    logits = self.model.module.get_logits(net_output)
                    targets = self.model.module.get_targets(sample, net_output)

                    loss = self.criterion(logits, targets)

                    losses = [loss.detach().clone()]

                    sample_size = targets.numel()

                    extra_losses = self.model.module.get_extra_losses(net_output)
                    if torch.is_tensor(extra_losses):
                        extra_losses = [extra_losses]
                    for p, coef in zip(extra_losses, self.loss_weights):
                        if coef == 0:
                            losses.append(torch.tensor(0))
                        elif p is not None:
                            p = coef * p.float() * sample_size
                            loss += p
                            losses.append(p)

                    loss.backward()
                    self.optimizer.step()
                    # scheduler.step()


                    num_updates = 1 + i + n_epoch * len(data_loader)
                    self.model.module.set_num_updates(num_updates)
                    # for j in range(len(self.loss_weights)):
                    #     self.loss_weights[j] = max(
                    #         self.max_loss_weights[j] * 0.9995 ** num_updates, self.min_loss_weights[j]
                    #     )

                    logging_outputs = {
                        'loss_0': losses[0].item() / sample_size / math.log(2),
                        'loss_1': losses[1].item() / sample_size / math.log(2),
                        'loss_2': losses[2].item() / sample_size / math.log(2),
                        'prob_ppl': net_output['prob_perplexity'].item(),
                        'code_ppl': net_output['code_perplexity'].item(),
                    }

                    with torch.no_grad():
                        if logits.numel() == 0:
                            corr = 0
                            count = 0
                        else:
                            assert logits.dim() > 1, logits.shape
                            _max = logits.argmax(-1) == 0
                            _min = logits.argmin(-1) == 0

                            both = _max & _min
                            corr = _max.long().sum().item() - both.long().sum().item()
                            count = float(_max.numel())
                        
                        logging_outputs['correct'] = corr
                        logging_outputs['total'] = count
                    
                    if self.data_parallel_world_size > 1:
                        _logging_outputs = self._all_gather_list_sync([logging_outputs])
                        for key in logging_outputs.keys():
                            logging_outputs[key] = float(sum(log[key] for log in _logging_outputs))

                            if 'loss' in key or 'ppl':
                                logging_outputs[key] = (
                                    logging_outputs[key] / len(_logging_outputs)
                                )
                            else:
                                logging_outputs[key] = int(logging_outputs[key])
                        del _logging_outputs
                    
                    self.metric.update(**logging_outputs)
                    inner_metric.update(**logging_outputs)

                    pbar.set_postfix_str(
                        f'[Epoch {n_epoch + 1}] loss_0: {losses[0].item() / sample_size / math.log(2):.3f}, '
                        f'loss_1: {losses[1].item() / sample_size / math.log(2):.4f}, '
                        f'loss_2: {losses[2].item() / sample_size / math.log(2):.3f}, '
                        f'acc: {corr / count:.4f}'
                    )
                    pbar.update(1)

                    if (i + 1) % self.log_interval == 0:
                        mid_stats = inner_metric.get_epoch_dict(
                            total_iter=self.log_interval
                        )

                        # with torch.no_grad():
                        #     q = self.model.module.quantize(**sample['net_input'])

                        # print(
                        #     f'[{i+1}] loss_0: {mid_stats["loss_0"]:.3f}, '
                        #     f'loss_1: {mid_stats["loss_1"]:.5f}, '
                        #     f'loss_2: {mid_stats["loss_2"]:.5f}, '
                        #     f'acc: {mid_stats["acc"]:.3f}\n'
                        #     f'[{i+1}] n_code: {q[1][net_output["mask_indices"]].unique().numel()}, '
                        #     f'prob: {mid_stats["prob_ppl"]:.3f}, '
                        #     f'code: {mid_stats["code_ppl"]:.3f}, '
                        #     f'temp: {net_output["temp"]:.3f}, '
                        #     f'w1: {self.loss_weights[0]:.3f}, w2: {self.loss_weights[1]:.3f}'
                        # )
                        # with torch.no_grad():
                        #     self.model.eval()
                            # k = self.model.module.extract_features(**sample['net_input'])
                            # p = k['features'][net_output['mask_indices']]
                            # mask_q = self.model.module.quantizer.forward_idx(p.view(k['features'].size(0), -1, k['features'].size(-1)))
                            # keep_q = self.model.module.quantizer.forward_idx(keep_features)
                            # self.model.train()
                            # breakpoint()
                            # pass
                        if self.is_data_parallel_master and not self.args.debug:
                            prefix = 'train_inner/'
                            for key in mid_stats.keys():
                                wandb.log({prefix + key: mid_stats[key]}, step=(i+1) + n_epoch*len(data_loader))
                            del mid_stats
 
            train_stats = self.metric.get_epoch_dict(
                total_iter = len(data_loader)
            )

            train_stats['epoch'] = n_epoch
            if self.is_data_parallel_master and self.args.debug == False:
                prefix = 'train/'
                for key in train_stats.keys():
                    wandb.log({prefix + key: train_stats[key]})

            if not self.no_validation and (n_epoch + 1) >= self.epoch_validate_from:
                valid_stats, early_stop = self.validate(n_epoch)
                if self.is_data_parallel_master and self.args.debug == False:
                    prefix = 'valid/'
                    for key in valid_stats.keys():
                        wandb.log({prefix + key: valid_stats[key]})
                
                if early_stop:
                    logger.info(
                        'early stop since valid performance has not improved for '
                        'last {} runs'.format(self.args.patience)
                    )
                    break
            
            if (n_epoch + 1) % self.epoch_save_interval == 0:
                if self.is_data_parallel_master:
                    best_model_path = self.save_path(self.args.eval_data[0]) + '_last'
                    tr_utils.model_save(self.model, best_model_path, n_epoch, self.optimizer)

        logger.info('train finished at epoch {}'.format(n_epoch))

        if self.is_data_parallel_master and self.args.debug == False:
            wandb.finish(0)

        if world_size > 1:
            dist.destroy_process_group()

    @torch.no_grad()
    def validate(self, n_epoch, subset='valid'):
        data_loader = self.dataloader_set(
            self.datasets[subset],
            self.data_parallel_world_size,
            self.args.batch_size
        )
        self.model.eval()
        with tqdm(total=len(data_loader)) as pbar:
            for sample in data_loader:
                sample = self._prepare_sample(sample)

                net_output = self.model(**sample['net_input'])

                logits = self.model.module.get_logits(net_output)
                targets = self.model.module.get_targets(sample, net_output)

                loss = self.criterion(logits, targets)

                losses = [loss.detach().clone()]

                sample_size = targets.numel()

                extra_losses = self.model.module.get_extra_losses(net_output)
                if torch.is_tensor(extra_losses):
                    extra_losses = [extra_losses]
                for p, coef in zip(extra_losses, self.loss_weights):
                    if coef != 0 and p is not None:
                        p = coef * p.float() * sample_size
                        loss += p
                        losses.append(p)

                logging_outputs = {
                    'loss_0': losses[0].item() / sample_size / math.log(2),
                    'loss_1': losses[1].item() / sample_size / math.log(2),
                    'loss_2': losses[2].item() / sample_size / math.log(2),
                    'prob_ppl': net_output['prob_perplexity'].item(),
                    'code_ppl': net_output['code_perplexity'].item(),
                }

                with torch.no_grad():
                    if logits.numel() == 0:
                        corr = 0
                        count = 0
                    else:
                        assert logits.dim() > 1, logits.shape
                        _max = logits.argmax(-1) == 0
                        _min = logits.argmin(-1) == 0

                        both = _max & _min
                        corr = _max.long().sum().item() - both.long().sum().item()
                        count = float(_max.numel())
                    
                    logging_outputs['correct'] = corr
                    logging_outputs['total'] = count
                
                if self.data_parallel_world_size > 1:
                    _logging_outputs = self._all_gather_list_sync([logging_outputs])
                    for key in logging_outputs.keys():
                        logging_outputs[key] = float(sum(log[key] for log in _logging_outputs))

                        if 'loss' in key or 'ppl':
                            logging_outputs[key] = (
                                logging_outputs[key] / len(_logging_outputs)
                            )
                        else:
                            logging_outputs[key] = int(logging_outputs[key])
                    del _logging_outputs


                self.metric.update(**logging_outputs)

                pbar.set_postfix_str(
                    f'[Epoch {n_epoch + 1}] loss_0: {losses[0].item() / sample_size / math.log(2):.3f}, '
                    f'loss_1: {losses[1].item() / sample_size / math.log(2):.4f}, '
                    f'loss_2: {losses[2].item() / sample_size / math.log(2):.3f}, '
                    f'acc: {corr / count:.4f}'
                )
                pbar.update(1)
        
        valid_stats = self.metric.get_epoch_dict(
            total_iter=len(data_loader)
        )
        valid_stats['epoch'] = n_epoch

        if self.early_stopping_dict[subset](valid_stats[self.metric.update_target]):
            if self.is_data_parallel_master:
                # best_model_path = os.path.join(
                #     self.save_dir, f'{self.seed}_pretrained_{self.args.pretrained_load}'
                # )
                best_model_path = self.save_path(self.args.eval_data[0])
                tr_utils.model_save(self.model, best_model_path, n_epoch, self.optimizer)

        stopped = False
        if self.early_stopping_dict[subset].early_stop:
            stopped = True
        
        return valid_stats, stopped

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
                *dist_utils.all_gather_list(
                    [logging_outputs],
                    max_size = getattr(self, "all_gather_list_size", 1048576),
                    group = self.data_parallel_process_group
                )
            )
        )
        logging_outputs = results[0]
        logging_outputs = list(chain.from_iterable(logging_outputs))
        return logging_outputs
    
    def _prepare_sample(self, sample):
        if torch.cuda.is_available():
            sample = utils.move_to_cuda(sample)
        
        return sample