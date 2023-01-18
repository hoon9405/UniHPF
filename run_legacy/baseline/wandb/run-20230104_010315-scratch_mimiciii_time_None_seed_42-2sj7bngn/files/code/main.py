import argparse
import os, requests
import sys
import logging
import random
import pprint
from typing import OrderedDict, Tuple
from  numbers import Number
import traceback

import torch.distributed as dist
import utils.distributed_utils as dist_utils
import wandb

import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler
from datetime import date

logging.basicConfig(
    format="%(asctime)s | %(levelname)s %(name)s %(message)s)))",
    datefmt="%Y-%m-%d %H:%M:%S",
    level = os.environ.get("LOGLEVEL", "INFO").upper(),
    stream = sys.stdout
)
logger = logging.getLogger("train")

from utils import utils
import utils.trainer_utils as trainer_utils
from loggings import metrics, meters
from loggings.meters import AverageMeter, StopwatchMeter, TimeMeter
from datasets.base_dataset import HierarchicalEHRDataset, FlattenEHRDataset, EHRDataset
import models
import criterions

import signal
import types


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, default='/nfs_edlab/ghhur/UniHPF/input')
    #parser.add_argument('--input_path', type=str, default='/home/edlab/jykim/UniHPF_pretrain/input')
    # parser.add_argument('--input_path', type=str, default='/home/jykim/no_time_filter_data_augment')
    parser.add_argument('--output_path', type=str, default='/nfs_edlab/ghhur/UniHPF/output')
    #parser.add_argument('--output_path', type=str, default='/home/edlab/jykim/UniHPF_pretrain/output')
    # parser.add_argument('--output_path', type=str, default='/home/jykim/UniHPF_pretrain/checkpoints/20220909')
    parser.add_argument('--load_checkpoint', type=str, default=None)
    parser.add_argument('--start_epoch', type=int, default=1)

    # dataset
    parser.add_argument(
            '--pretrain_src', 
            choices=[
                'mimiciii', 'eicu', 'mimiciv', 'mimiciii_eicu_mimiciv'
            ], 
            type=str, 
            default='mimic3'
    )
    parser.add_argument('--pred_src', choices=['mimiciii', 'eicu', 'mimiciv', 'mimiciii_eicu_mimiciv'], type=str, default='mimiciii'
    )
    parser.add_argument('--ft_pt_src', choices=['mimiciii', 'eicu', 'mimiciv', 'mimiciii_eicu_mimiciv'], type=str, help="the pre-trained data source (use arg for finetuning)")
    parser.add_argument('--ft_pt_task',
        choices=['mlm', 'spanmlm', 'text_encoder_mlm', 'w2v', 'simclr', None], 
        type=str, default=None, help="the pre-training method applied")
    parser.add_argument('--ratio', choices=['0', '10', '100'], type=str, default='100')

    parser.add_argument('--eval_data', choices=['mimiciii', 'eicu', 'mimiciv', 'mimiciii_eicu_mimiciv'], type=str, default=None)
    parser.add_argument(
        '--pred_tasks',
        default=['mortality', 'long_term_mortality', 'los_3day', 'los_7day', 'readmission', 'final_acuity', 'imminent_discharge', 'diagnosis', 'creatinine', 'bilirubin', 'platelets', 'wbc'],
        choices=['mortality', 'long_term_mortality', 'los_3day', 'los_7day', 'readmission', 'final_acuity', 'imminent_discharge', 'diagnosis', 'creatinine', 'bilirubin', 'platelets', 'wbc'],
        type=list,
        help=""
    )

    # trainer
    parser.add_argument('--train_task', choices=['pretrain', 'sampled_pretrain', 'finetune', 'scratch'], type=str, default=None)
    parser.add_argument('--seed', type=str, default='42') #TODO: seed args for scratch / finetune in run file
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--max_epoch', type=int, default=300)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--valid_subset', type=str, default="")
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--criterion', type=str, default='cross_entropy')
    parser.add_argument( # 'loss' for pretrain, 'auFprc' for finetune
        '--best_checkpoint_metric', type=str, default='loss', choices=['loss', 'auprc', 'acc', 'auroc']
    )
    parser.add_argument( # 'False' for loss, 'True' for auprc
        '--maximize_best_checkpoint_metric', action='store_true', default=False
    )

    # pretrain
    parser.add_argument(
        '--pretrain_task', 
        choices=['mlm', 'spanmlm', 'text_encoder_mlm', 'w2v', 'simclr', None], 
        type=str, default=None
    )
    parser.add_argument('--mlm_prob', type=float, default=0.15)
    parser.add_argument('--mask_list', type=str, default='dpe, input, type')
    
    # model
    parser.add_argument(
        '--model', choices=['ehr_model', 'unihpf_simclr', 'unihpf_w2v', 'unihpf_mlm'], type=str, default='ehr_model',
        help='name of the model to be trained'
    )

    parser.add_argument(
        '--emb_type_model', type=str, required=False, default='descemb',
        help='name of the encoder model in the --emb_type_model'
    )
    parser.add_argument('--structure', choices=[None, 'hi', 'fl'], type=str, default=None)
    parser.add_argument(
        '--pred_model', type=str, required=False, default=None,
        help='name of the encoder model in the --pred_model'
    )

    # model hyper-parameter configs
    parser.add_argument('--pred_dim', type=int, default=128)  
    parser.add_argument('--embed_dim', type=int, default=128)
    parser.add_argument('--n_heads', type=int, default=4)
    parser.add_argument('--n_layers', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--type_token', action='store_true', default=True)
    parser.add_argument('--dpe', action='store_true', default=True)
    parser.add_argument('--pos_enc', action='store_true', default=True)
    parser.add_argument('--pred_pooling', choices=['cls', 'mean'], default='mean')
    parser.add_argument('--text_post_proj', action='store_true')
    parser.add_argument('--map_layers', type=int, default=1)
    parser.add_argument('--max_word_len', type=int, default=128)
    parser.add_argument('--max_seq_len', type=int, default=8192)
    parser.add_argument('--time_embed', type=str, choices=['None', 'Alibi'], default="None")

    
    # decoder arguments
    parser.add_argument('--decoder-embed-dim', type=int, default=128)
    parser.add_argument('--decoder-n-heads', type=int, default=4)
    parser.add_argument('--decoder-n-layers', type=int, default=4)
    parser.add_argument('--decoder-dropout', type=float, default=0.2)
    parser.add_argument('--note-min-seq-len', type=int, default=0)
    parser.add_argument('--note-max-seq-len', type=int, default=512)
    parser.add_argument(
        '--sample-strategy', type=str, default='topp',
        choices=['topp']
    )
    parser.add_argument('--sample-k', type=int, default=5)
    parser.add_argument('--gen-max-len', type=int, default=512)
    parser.add_argument('--test_batch_size', type=int, default=32)

    # for w2v
    parser.add_argument('--feature_grad_mult', type=float, default=0.1)
    parser.add_argument('--num_negatives', type=int, default=25)
    parser.add_argument('--codebook_negatives', type=int, default=0)
    parser.add_argument('--logit_temp', type=float, default=0.1)
    parser.add_argument('--latent_vars', type=int, default=320)
    parser.add_argument('--latent_groups', type=int, default=2)
    parser.add_argument('--latent_temp', type=Tuple[float, float, float], default=(2, 0.5, 0.999995))
    parser.add_argument('--final_dim', type=int, default=128)
    parser.add_argument('--dropout_input', type=float, default=0.1)
    parser.add_argument('--dropout_features', type=float, default=0.1)

    parser.add_argument('--mask_prob', type=float, default=0.65)
    parser.add_argument('--mask_length', type=int, default=1)
    parser.add_argument('--mask_selection', type=str, default='static')
    parser.add_argument('--mask_other', type=float, default=0)
    parser.add_argument('--no_mask_overlap', type=bool, default=False)
    parser.add_argument('--mask_min_space', type=int, default=0)
    parser.add_argument('--perp_weight', type=float, default=0.1)
    parser.add_argument('--reg_weight', type=int, default=5)
    parser.add_argument(
        '--log_interval', type=int, default=10,
    )

    # for ddp setting
    parser.add_argument('--dp', action='store_true')
    parser.add_argument('--world_size', type=int, default=1)
    parser.add_argument('--port', type=str, default = '12355')
    
    # MLM pretrained load
    parser.add_argument('--pretrained_path', type=str)
    parser.add_argument('--pretrained_load', type=str, 
        choices=[None, 'mlm', 'spanmlm', 'w2v'], default=None)

    # resume
    parser.add_argument('--resume', action='store_true')

    parser.add_argument('--edlab_resume', action='store_true')

    parser.add_argument(
        '--wandb', action='store_true', default=False,
    )
    parser.add_argument(
        '--wandb_project_name', type=str, default='test'
    )
    parser.add_argument(
        '--wandb_entity', type=str, default='hoon9405'
    )
    parser.add_argument(
        '--wandb_run_name', type=str, default=None
    )
    parser.add_argument(
         '--sub_id', type=str, default=None
     )

    # nsml
    parser.add_argument( # True when using nsml
        '--auto_resume', action='store_true', default=False,
    )   
    parser.add_argument( # resume directory
        '--resume_path', type=str, default=None
    )

    return parser


def sigterm_handler(s: signal.Signals, f: types.FrameType) -> None:
    raise KeyboardInterrupt


def load_dataset(args, split, dataname, seed) -> None:
    
    # ds = EHRDataset(
    #     data=dataname,
    #     input_path=args.input_path,
    #     split=split,
    #     structure=args.structure,
    #     train_task=args.train_task,
    #     ratio=args.ratio,
    #     pred_tasks=args.pred_tasks,
    #     seed=seed,
    #     mask_list=args.mask_list,
    #     pretrain_task=args.pretrain_task,
    #     mlm_prob=args.mlm_prob,
    #     max_word_len=args.max_word_len,
    #     max_seq_len=args.max_seq_len,
    # )

    if args.structure =='hi':
        ds = HierarchicalEHRDataset(
            data=dataname,
            input_path=args.input_path,
            split=split,
            structure=args.structure,
            train_task=args.train_task,
            ratio=args.ratio,
            pred_tasks=args.pred_tasks,
            seed=args.seed,
            mask_list=args.mask_list,
            pretrain_task=args.pretrain_task,
            mlm_prob=args.mlm_prob,
            max_word_len=args.max_word_len,
            max_seq_len=args.max_seq_len,
        )
    elif args.structure=='fl':
        ds = FlattenEHRDataset(
            data=dataname,
            input_path=args.input_path,
            split=split,
            structure=args.structure,
            train_task=args.train_task,
            ratio=args.ratio,
            pred_tasks=args.pred_tasks,
            seed=args.seed,
            mask_list=args.mask_list,
            pretrain_task=args.pretrain_task,
            mlm_prob=args.mlm_prob,
            max_word_len=args.max_word_len,
            max_seq_len=args.max_seq_len,
        )
    
    return ds

def load_dataloader(dataset, batch_size, seed, collator) -> None:
    sampler = None if not dist.is_initialized() else (
    DistributedSampler(dataset, seed=seed)
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True if not dist.is_initialized() else False,
        num_workers=8,
        collate_fn=collator,
        sampler=sampler,
    )  
 
    return dataloader, sampler


def pre_main(args):
    if args.valid_subset and len(args.valid_subset) > 0:
        args.valid_subset = args.valid_subset.replace(' ','').split(',')

    if args.mask_list and len(args.mask_list) > 0:
        args.mask_list = args.mask_list.replace(' ','').split(',')

    if args.seed and len(args.seed) > 0:
        args.seed = [int(s) for s in args.seed.replace(' ','').split(',')]

    if args.train_task in ['scratch', 'finetune']: #TODO: check 명령어 for finetune / pretrain
        assert len(args.seed) == 1, "Scratch / Finetune should run on one seed"
    elif 'pretrain' in args.train_task:
        assert len(args.seed) == 5, "Pretrain should run on 5 seeds"
        assert len(args.valid_subset) == 0, "Pretrain should not have valid subset"


    if 'pretrain' in args.train_task:
        args.checkpoint_prefix = f'{args.structure}_{args.train_task}_{args.pretrain_task}_{args.pretrain_src}_time_{args.time_embed}_'
    elif 'finetune' in args.train_task:
        args.checkpoint_prefix = f'{args.structure}_{args.ft_pt_task}_{args.ft_pt_src}_{args.train_task}_{args.pred_src}_seed_{args.seed[0]}_time_{args.time_embed}_'
    elif args.train_task == 'scratch':
        # args.checkpoint_prefix = f'{args.structure}_{args.train_task}_{args.pred_src}_{args.pred_target}_seed_{args.seed}_'
        args.checkpoint_prefix = f'{args.structure}_{args.train_task}_{args.pred_src}_seed_{args.seed[0]}_time_{args.time_embed}_'
    elif args.train_task in ['note_scratch', 'note_finetune']:
        args.checkpoint_prefix = f'{args.structure}_{args.train_task}_{args.pred_src}_seed_{args.seed[0]}_'

    if args.auto_resume:
        args.resume = False
        os.makedirs(args.resume_path, exist_ok=True)
        prev_exps = os.listdir(args.resume_path)

        for i in prev_exps:
            if i == f'{args.checkpoint_prefix}checkpoint_last.pt':
                args.resume = True
                logger.info(f"Resume checkpoint from {i}")
                break
    
    args.checkpoint_save_path = os.path.join(args.output_path, args.train_task)

    os.makedirs(args.checkpoint_save_path, exist_ok=True)

    if args.train_task in ['finetune', 'scratch', 'sampled_finetune']:
        # args.wandb_project_name = f'unihpf_pretrain_{args.structure}_{args.pred_target}'
        args.wandb_project_name = f'{args.structure}_prediction'
        if args.train_task in ['finetune', 'sampled_finetune']:
            args.wandb_run_name = f'{args.ft_pt_task}_{args.ft_pt_src}_{args.train_task}_{args.pred_src}_seed_{args.seed[0]}_time_{args.time_embed}'
        elif args.train_task == 'scratch':
            # args.wandb_run_name = f'{args.structure}_{args.train_task}_{args.pred_src}_{args.pred_target}_ratio_{args.ratio}_seed_{args.seed}'
            args.wandb_run_name = f'{args.train_task}_{args.pred_src}_time_{args.time_embed}_seed_{args.seed[0]}'
    elif 'pretrain' in args.train_task:
        args.wandb_project_name = f'{args.pretrain_task}_{args.structure}'
        args.wandb_run_name = f'{args.structure}_{args.train_task}_{args.pretrain_task}_{args.pretrain_src}_time_{args.time_embed}'
    elif args.train_task in ['note_scratch', 'note_finetune']:
        args.wandb_project_name = f'note_{args.structure}'
        args.wandb_run_name = f'{args.structure}_{args.train_task}_{args.pred_src}_ratio_{args.ratio}_seed_{args.seed[0]}'

    if args.train_task in ['finetune', 'scratch', 'sampled_finetune']:
        args.log_prefix = f"{args.pred_src}_"
    elif args.train_task in ['note_scratch', 'note_finetune']:
        args.log_prefix = f"{args.wandb_project_name}_{args.pred_src}_"
    else:
        args.log_prefix = f"{args.wandb_project_name}_{args.pretrain_src}_"

    dist_utils.call_main(args, main)


def model_load(path):
    state_dict = torch.load(path, map_location='cpu')
    model = state_dict['model']
    epoch = state_dict['epoch']
    optimizer = state_dict['optimizer']
    valid_losses = state_dict['valid_losses']
    try:
        num_runs = state_dict['patience']
    except KeyError:
        num_runs = 0

    return model, epoch, optimizer, num_runs, valid_losses


def main(args) -> None:
    signal.signal(signal.SIGTERM, sigterm_handler)
    np.random.seed(args.seed[0])
    random.seed(args.seed[0])
    utils.set_torch_seed(args.seed[0])

    args.pred_tasks = [trainer_utils.get_task(task, args.pred_src) for task in args.pred_tasks]

    logger.info(pprint.pformat(args))

    if args.train_task in ['note_scratch', 'note_finetune']:
        args.max_seq_len = (
            args.max_seq_len
            if args.max_seq_len and args.max_seq_len >= args.note_max_seq_len
            else args.note_max_seq_len
        )

    # model & criterion build
    model = models.build_model(args)
    if (args.auto_resume and args.resume) or args.edlab_resume:
        if args.auto_resume and args.resume:
            args.ckpt_load_path = os.path.join(args.resume_path, f'{args.checkpoint_prefix}checkpoint_last.pt')
        elif args.edlab_resume:
            args.ckpt_load_path = os.path.join(args.checkpoint_save_path, f'{args.checkpoint_prefix}checkpoint_last.pt')
        model_loaded, args.start_epoch, _, num_runs, valid_losses = model_load(args.ckpt_load_path)
        utils.should_stop_early.num_runs = num_runs
        utils.should_stop_early.best = valid_losses[0]
        model = model.from_pretrained(args, model_loaded) 
        print("Resume training from checkpoint "+ args.ckpt_load_path + " ...")
        
        model = model.from_pretrained(args, model_loaded) 
    elif args.train_task in ['finetune', 'sampled_finetune', 'note_finetune'] or args.load_checkpoint is not None:
        if args.train_task in ['finetune', 'note_finetune']:
            args.ckpt_load_path = os.path.join(args.output_path, "pretrain", f"{args.structure}_pretrain_{args.ft_pt_task}_{args.ft_pt_src}_time_{args.time_embed}_checkpoint_best.pt")
            # print(args.ckpt_load_path)
        elif args.train_task == 'sampled_finetune':
            args.ckpt_load_path = os.path.join(args.output_path, "sampled_pretrain", f"{args.structure}_sampled_pretrain_{args.ft_pt_task}_{args.ft_pt_src}_seed_{args.seed[0]}_checkpoint_best.pt")
        model = model.from_pretrained(args) 
        print("Loaded checkpoint from "+ args.ckpt_load_path + " ...")
    criterion = criterions.build_criterion(args)

    logger.info(model)
    logger.info('model: {}'.format(model.__class__.__name__))
    logger.info(
        'num. shared model params: {:,} (num. trained: {:,})'.format(
            sum(p.numel() for p in model.parameters()),
            sum(p.numel() for p in model.parameters() if p.requires_grad)
        )
    )

    # dataloader build
    dataloaders = {}
    samplers = {}
    if args.valid_subset and len(args.valid_subset) > 0:
        data_split = ['train'] + [s for s in args.valid_subset]
    else:
        print("Only in train mode (no evaluation)")
        data_split = ['train']
    datanames = args.pretrain_src if args.train_task in ['pretrain', 'sampled_pretrain'] else args.pred_src
    datanames = datanames.split('_')
    logger.info(f'{args.train_task}, {datanames}, {data_split}')

    for split in data_split:
        dataset_list = []
        for dataname in datanames:
            dataset= load_dataset(args, split, dataname, args.seed)
            dataset_list.append(dataset)
            logger.info(f'{split}, {len(dataset)}')

        dataloader, sampler = load_dataloader(
            torch.utils.data.ConcatDataset(dataset_list), 
            args.batch_size, 
            args.seed[0], 
            dataset_list[0].collator
            )

        dataloaders[split]= dataloader
        samplers[split] = sampler

    # trainer build
    from trainers.base_trainer import BaseTrainer as Trainer

    trainer = Trainer(args, model, criterion)

    logger.info(
        'training on {} devices (GPUs)'.format(
            args.world_size
        )
    )

    max_epoch = args.max_epoch
    lr = args.lr

    if args.wandb and dist_utils.is_master(args): # TODO: need update for automatic resume
         if (args.resume or args.edlab_resume) and (args.sub_id is not None):
             wandb.init(
                 project=args.wandb_project_name,
                 entity=args.wandb_entity,
                 config=args,
                 id=args.wandb_run_name + '-' + args.sub_id,
                 resume="must"
             )
         else:
             args.sub_id = wandb.util.generate_id()
             wandb.init(
                 project=args.wandb_project_name,
                 entity=args.wandb_entity,
                 config=args,
                 id=args.wandb_run_name + '-' + args.sub_id,
                 reinit=True
             )

         wandb.run_name = args.wandb_run_name

    cum_data_count = 0

    train_meter = meters.StopwatchMeter()
    train_meter.start()

    try:
        if (args.auto_resume and args.resume) or args.edlab_resume:
            print("Start patience: ", utils.should_stop_early.num_runs)

        for i in range(args.start_epoch, max_epoch + 1):
            cum_data_count += len(dataloaders['train'])
            validation = True if 'valid' in data_split else False
            valid_losses, should_stop = train(args, trainer, cum_data_count, epoch_itr=dataloaders, epoch=i, sampler=samplers, validation=validation)
            if should_stop:
                break
    
        if 'test' in args.valid_subset:
            best_state_dict = torch.load(os.path.join(args.checkpoint_save_path, f'{args.checkpoint_prefix}checkpoint_best.pt'), map_location='cpu')['model']
            if not isinstance(trainer.model, torch.nn.parallel.DistributedDataParallel):
                trainer.model.load_state_dict(best_state_dict, strict=True) # load best ckpt for testing
            else:
                trainer.model.module.load_state_dict(best_state_dict, strict=True)
            print("loaded best checkpoint")
            
            valid_losses = validate(args, trainer, dataloaders, 'test')
            logger.info(f'valid_losses = {valid_losses}')

            train_meter.stop()

        if args.wandb and dist_utils.is_master(args):
            wandb.finish(0)
        logger.info('done training in {:.1f} seconds'.format(train_meter.sum))

    except KeyboardInterrupt: # when sigterm in nsml -> save last checkpoint and reschedule
        if args.auto_resume and dist_utils.is_master(args):

            print("INTERRUPTED!!")

            resume_state_dict = torch.load(os.path.join(args.checkpoint_save_path, f'{args.checkpoint_prefix}checkpoint_last.pt'), map_location='cpu')

            os.makedirs(args.resume_path, exist_ok=True)

            torch.save(
                resume_state_dict,
                os.path.join(args.resume_path, f'{args.checkpoint_prefix}checkpoint_last.pt')
            )

            # logger.info("Finish saving before interrupt")
            print("Finish saving before interrupt")

            try:
                api_host = os.environ["NSML_RUN_METADATA_API"]
                api_secret = os.environ["NSML_RUN_SECRET"]
                requests.put(
                    f"{api_host}/v1/rescheduled",
                    headers={"X-NSML-Run-Secret": api_secret},
                    json={"rescheduled": True},
                ).raise_for_status()
                print("send rescheduling request")
            except:
                # Sometimes, the HTTP request might fail, but the training process should not be stopped.
                traceback.print_exc()

        else:
            print("[KeyboardInterrupt] Not run in auto-resume mode")



@metrics.aggregate('train')
def train(args, trainer, cum_data_count, epoch_itr, epoch, sampler, validation):
    logger.info('begin training epoch {}'.format(epoch))

    if dist.is_initialized():
        sampler['train'].set_epoch(epoch)

    should_stop = False
    num_updates = trainer.get_num_updates()
    logger.info('Start iterating over samples')
    logger.info(f'len(epoch_itr[train]) ={len(epoch_itr["train"])}')
    
    for i, sample in enumerate(epoch_itr['train']):

        with metrics.aggregate('train_inner'):
            log_output = trainer.train_step(sample)

        if log_output is not None:
            num_updates = trainer.get_num_updates()
            if num_updates % args.log_interval == 0:
                stats = get_training_stats(metrics.get_smoothed_values('train_inner'))
                progress_log(
                    args,
                    stats, 
                    tag='train_inner', 
                    step=num_updates, 
                    size=cum_data_count, 
                    log_wandb=args.wandb
                    )
                # print(metrics.state_dict()['train_inner'])
                metrics.reset_meters('train_inner')

    stats = get_training_stats(metrics.get_smoothed_values('train'))
    progress_print(args, stats, tag='train', log_wandb=args.wandb)

    if validation:
        logger.info('Evaluation start')    
        valid_losses, should_stop = validate_and_save(args, trainer, epoch_itr, epoch, 'valid')

    else:
        valid_losses, should_stop = train_save(args, trainer, epoch, stats)

    logger.info('end of epoch {} (average epoch stats below)'.format(epoch))
        
    metrics.reset_meters('train')
    return valid_losses, should_stop

def train_save(args, trainer, epoch, stats):
    should_stop = False

    valid_losses = []
    stats = get_train_stats(args, stats)
    valid_losses.append(stats[f'{args.log_prefix}{args.best_checkpoint_metric}'])
    logger.info(f'train_losses = {valid_losses}')
    should_stop |= utils.should_stop_early(
        args.patience,
        valid_losses[0],
        descending=(not args.maximize_best_checkpoint_metric)
    )

    state_dict = {
        'model': trainer.model.state_dict() if not (
            isinstance(trainer.model, torch.nn.parallel.DistributedDataParallel)
        ) else trainer.model.module.state_dict(),
        'epoch': epoch,
        'optimizer': trainer.optimizer.state_dict(),
        'valid_losses': valid_losses,
        'patience': utils.should_stop_early.num_runs,
    }

    if 'pretrain' in args.train_task:
        torch.save(
            state_dict,
            os.path.join(args.checkpoint_save_path, f'{args.checkpoint_prefix}epoch_{epoch}_checkpoint_last.pt')
        )

        if utils.should_stop_early.best == valid_losses[0]:
            torch.save(
                state_dict,
                os.path.join(args.checkpoint_save_path, f'{args.checkpoint_prefix}epoch_{epoch}_checkpoint_best.pt')
            )

    torch.save(
        state_dict,
        os.path.join(args.checkpoint_save_path, f'{args.checkpoint_prefix}checkpoint_last.pt')
    )

    if utils.should_stop_early.best == valid_losses[0]:
        torch.save(
            state_dict,
            os.path.join(args.checkpoint_save_path, f'{args.checkpoint_prefix}checkpoint_best.pt')
        )
    
    return valid_losses, should_stop

def validate_and_save(args, trainer, epoch_itr, epoch, valid_subset):
    num_updates = trainer.get_num_updates()

    should_stop = False
    training_time_hours = trainer.cumulative_training_time() / (60 * 60)

    valid_losses = validate(args, trainer, epoch_itr, valid_subset)
    logger.info(f'valid_losses = {valid_losses}')
    should_stop |= utils.should_stop_early(
        args.patience,
        valid_losses[0],
        descending=(not args.maximize_best_checkpoint_metric)
    )

    state_dict = {
        'model': trainer.model.state_dict() if not (
            isinstance(trainer.model, torch.nn.parallel.DistributedDataParallel)
        ) else trainer.model.module.state_dict(),
        'epoch': epoch,
        'optimizer': trainer.optimizer.state_dict(),
        'valid_losses': valid_losses,
        'patience': utils.should_stop_early.num_runs,
    }

    if 'pretrain' in args.train_task:
        torch.save(
            state_dict,
            os.path.join(args.checkpoint_save_path, f'{args.checkpoint_prefix}epoch_{epoch}_checkpoint_last.pt')
        )

        if utils.should_stop_early.best == valid_losses[0]:
            torch.save(
                state_dict,
                os.path.join(args.checkpoint_save_path, f'{args.checkpoint_prefix}epoch_{epoch}_checkpoint_best.pt')
            )

    torch.save(
        state_dict,
        os.path.join(args.checkpoint_save_path, f'{args.checkpoint_prefix}checkpoint_last.pt')
    )

    if utils.should_stop_early.best == valid_losses[0]:
        torch.save(
            state_dict,
            os.path.join(args.checkpoint_save_path, f'{args.checkpoint_prefix}checkpoint_best.pt')
        )
    
    return valid_losses, should_stop

def get_training_stats(stats):
    stats['wall'] = round(metrics.get_meter('default', 'wall').elapsed_time, 0)
    return stats

def validate(args, trainer, epoch_itr, valid_subset):
    valid_losses = []
    # for subset in args.valid_subset:
    logger.info('begin validation on "{}" subset'.format(valid_subset))

    with metrics.aggregate(new_root=True) as agg:
        for i, sample in enumerate(epoch_itr[valid_subset]):
            trainer.valid_step(sample, subset=valid_subset)

    stats = get_valid_stats(args, trainer, valid_subset, agg.get_smoothed_values())

    progress_print(args, stats, tag=valid_subset, prefix=f'valid on {valid_subset} subset', log_wandb=args.wandb)
    valid_losses.append(stats[f'{args.log_prefix}{args.best_checkpoint_metric}'])
    return valid_losses

def get_valid_stats(args, trainer, subset, stats):
    stats['num_updates'] = trainer.get_num_updates()

    if not hasattr(get_valid_stats, 'best'):
        get_valid_stats.best = dict()
    
    prev_best = getattr(get_valid_stats, 'best').get(
        subset, stats[f'{args.log_prefix}{args.best_checkpoint_metric}']
    )
    best_function = max if args.maximize_best_checkpoint_metric else min
    get_valid_stats.best[subset] = best_function(
        stats[f'{args.log_prefix}{args.best_checkpoint_metric}'], prev_best
    )

    key = 'best_{0}'.format(f'{args.log_prefix}{args.best_checkpoint_metric}')
    stats[key] = get_valid_stats.best[subset]

    return stats

def get_train_stats(args, stats):

    if not hasattr(get_train_stats, 'best'):
        get_train_stats.best = dict()
    
    prev_best = getattr(get_train_stats, 'best').get(
        'train', stats[f'{args.log_prefix}{args.best_checkpoint_metric}']
    )
    best_function = max if args.maximize_best_checkpoint_metric else min
    get_train_stats.best['train'] = best_function( # epsilon
        stats[f'{args.log_prefix}{args.best_checkpoint_metric}'], prev_best
    )

    key = 'best_{0}'.format(f'{args.log_prefix}{args.best_checkpoint_metric}')
    stats[key] = get_train_stats.best['train']

    return stats


def format_stat(stat):
    if isinstance(stat, Number):
        stat = "{:g}".format(stat)
    elif isinstance(stat, AverageMeter):
        stat = "{:.3f}".format(stat.avg)
    elif isinstance(stat, TimeMeter):
        stat = "{:g}".format(round(stat.avg))
    elif isinstance(stat, StopwatchMeter):
        stat = "{:g}".format(round(stat.sum))
    elif torch.is_tensor(stat):
        stat = stat.tolist()
    return stat


def _format_stats(stats):
    postfix = OrderedDict(stats)
    # Preprocess stats according to datatype
    for key in postfix.keys():
        postfix[key] = str(format_stat(postfix[key]))
    return postfix

def _str_pipes(stats):
    return ' | '.join(key + ' ' + stats[key].strip() for key in stats.keys())

def _str_commas(stats):
    return ', '.join(key + '=' + stats[key].strip() for key in stats.keys())

def progress_log(args, stats, tag=None, step=0, size=1, prefix='', log_wandb=False):
    stats = _format_stats(stats)
    postfix = _str_commas(stats)
    with utils.rename_logger(logger, tag):
        logger.info(
            '{}: {:5d} / {:d} {}'.format(
                prefix, step, size, postfix
            )
        )
        if log_wandb and dist_utils.is_master(args):
            _stats = {}
            for key in stats:
                _stats[tag + '/' + key] = float(stats[key])
            wandb.log(_stats)

def progress_print(args, stats, tag=None, prefix='', log_wandb=False):
    postfix = _str_pipes(_format_stats(stats))
    with utils.rename_logger(logger, tag):
        logger.info('{} | {}'.format(prefix, postfix))
        if log_wandb and dist_utils.is_master(args):
            _stats = {}
            for key in stats:
                _stats[tag + '/' + key] = float(stats[key])
            wandb.log(_stats)


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    pre_main(args)
