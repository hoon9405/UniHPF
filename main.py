import argparse
from ast import Store
import logging
import logging.config
import random
import os
import sys
import glob

from typing import List, Tuple

# should setup root logger before importing any relevant libraries.
logging.basicConfig(
    format="%(asctime)s | %(levelname)s %(name)s %(message)s)))",
    datefmt="%Y-%m-%d %H:%M:%S",
    level = os.environ.get("LOGLEVEL", "INFO").upper(),
    stream = sys.stdout
)
logger = logging.getLogger(__name__)

import numpy as np

import torch
import torch.multiprocessing as mp


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--device_num', type=str, default='0'
    )

    # checkpoint configs
    parser.add_argument('--input_path', type=str, default='/home/edlab/ghhur/Pretrained_DescEmb/data/input')
    parser.add_argument('--output_path', type=str, default='/home/edlab/ghhur/Pretrained_DescEmb/data/output/')
    parser.add_argument('--save_dir', type=str, default='checkpoints')
    parser.add_argument('--save_prefix', type=str, default='checkpoint')
    parser.add_argument('--load_checkpoint', type=str, default=None)

    parser.add_argument('--disable_validation', action='store_true', help='disable validation')
    parser.add_argument('--disable_save', action='store_true', help='disable save')

    # dataset
    parser.add_argument('--train_type', choices=['single', 'transfer', 'pooled'], type=str, default=None)
    parser.add_argument(
            '--src_data', 
            choices=[
                'mimic3', 
                'eicu', 
                'mimic4', 
                'mimic3_eicu', 
                'mimic3_mimic4', 
                'mimic4_eicu', 
                'mimic3_mimic4_eicu', 
                'benchmark_mimic', 
                'benchmark_eicu'
            ], 
            type=str, 
            default='mimic3'
    )
    parser.add_argument('--ratio', choices=['0', '10', '100'], type=str, default='100')
    parser.add_argument('--feature', choices=['select', 'entire', 'lab_only'], default='whole')
    parser.add_argument('--eval_data', choices=['mimic3', 'eicu', 'mimic4'], type=str, default=None)

    parser.add_argument(
        '--pred_target',
        choices=['mort', 'los3', 'los7', 'readm', 'dx', 'im_disch', 'fi_ac'],
        type=str,
        default='mort',
        help=""
    )

    # trainer
    parser.add_argument('--train_task', choices=['predict', 'pretrain'], type=str, default=None)
    parser.add_argument('--seed', type=int, default=2022)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--valid_subsets', type=str, default="valid, test")
    parser.add_argument('--patience', type=int, default=10)
    
    # pretrain
    parser.add_argument(
        '--pretrain_task', 
        choices=['mlm', 'spanmlm', 'cont', 'autore', 'text_encoder_mlm', 'w2v', None], 
        type=str, default=None
    )
    parser.add_argument('--mlm_prob', type=float, default=0.15)
    parser.add_argument('--mask_list', type=str, default='input, type, dpe')
    
    # model
    parser.add_argument(
        '--model', choices=['SAnD', 'Rajikomar','DescEmb', 'UniHPF'], type=str, required=True, default='UniHPF',
        help='name of the model to be trained'
    )

    parser.add_argument(
        '--input2emb_model', type=str, required=False, default=None,
        choices=['codeemb', 'descemb'], 
        help='name of the encoder model in the --input2emb_model'
    )
    parser.add_arugment('--structure', choices=[None, 'hi', 'fl'] type=str, default=None)
    parser.add_argument(
        '--pred_model', type=str, required=False, default=None,
        help='name of the encoder model in the --pred_model'
    )
    parser.add_argument('--apply_mean', action='store_true', default=None)

    # model hyper-parameter configs
    parser.add_argument('--pred_dim', type=int, default=128)  
    parser.add_argument('--embed_dim', type=int, default=128)
    parser.add_argument('--n_heads', type=int, default=4)
    parser.add_argument('--n_layers', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--type_token', action='store_true')
    parser.add_argument('--dpe', action='store_true')
    parser.add_argument('--pos_enc', action='store_true')
    parser.add_argument('--pred_pooling', choices=['cls', 'mean'], default='cls')
    parser.add_argument('--text_post_proj', action='store_true')
    parser.add_argument('--map_layers', type=int, default=1)
    parser.add_argument('--max_word_len', type=int, default=256)
    parser.add_argument('--max_seq_len', type=int, default=8192)

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


    # for ddp setting
    parser.add_argument('--DDP', action='store_true') # Temporal argument should be removed
    parser.add_argument('--dp', action='store_true')
    parser.add_argument('--world_size', type=int, default=1)
    parser.add_argument('--port', type=str, default = '12355')


    parser.add_argument('--debug', action='store_true')
    
    # MLM pretrained load
    parser.add_argument('--pretrained_path', type=str)
    parser.add_argument('--pretrained_load', type=str, 
        choices=[None, 'mlm', 'spanmlm', 'w2v'], default=None)

    # resume
    parser.add_argument('--resume', action='store_true')

    return parser

def main():
    args = get_parser().parse_args()
   
    #parsing args
    args.valid_subsets = (
        args.valid_subsets.replace(' ','').split(',')
        if (
            not args.disable_validation
            and args.valid_subsets
            and bool(args.valid_subsets.strip())
        )
        else []
    )
   
    args.mask_list = (
        args.mask_list.replace(' ','').split(',')
    )

    if args.train_type =='pooled':
        if args.train_task =='predict':
            args.eval_data = ([args.src_data] + 
                    args.src_data.split('_')
                )  
        else:
            args.eval_data = [args.src_data] 
    elif args.train_type=='single':
        args.eval_data = [args.src_data]
    else:
        args.eval_data= [args.eval_data]

    model_configs = {
    'SAnD': ('fl', 'codeemb', 'select'), 
    'Rajikomar' : ('hi', 'codeemb', 'entire'),
    'DescEmb' : ('hi', 'descemb', 'select') 
    'UniHPF': ('hi', 'descemb', 'entire')  
    }

    structure, input2emb, feature = model_configs[args.model]

    if args.structure is None:
        args.structrue = structure
    if args.input2emb is None:
        args.input2emb = input2emb
    if args.feature is None:
        args.feature = feature

    if args.train_type =='single' and len(args.eval_data)!=1:
        raise AssertionError('single domain training must select single dataset')

    if args.train_type =='pooled' and args.src_data in ['mimic3', 'eicu', 'mimic4']:
        raise AssertionError('pooled must select at least two datasets')
    
    if args.train_type =='transfer' and (args.eval_data[0] in args.src_data):
        raise AssertionError('transfer target should not be in trained src data')

    if args.train_task =='pretrain' and args.pretrain_task is None:
        raise AssertionError('should select pretrain task')

   
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device_num)

    args.device_ids = list(range(len(args.device_num.split(','))))
    print('device_number : ', args.device_ids)
    args.world_size = len(args.device_ids)
    ckpt_root = set_struct(vars(args))

    #seed pivotting
    mp.set_sharing_strategy('file_system')
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
        
    if args.train_task == 'predict':
        from trainers import BaseTrainer as Trainer
    elif args.train_task == 'pretrain' and args.pretrain_task == 'w2v':
        from trainers import W2VTrainer as Trainer
    elif args.train_task == 'pretrain' and args.pretrain_task == 'mlm'::
        from trainers import MLMTranier as Trainer
    else:
        raise NotImplementedError("Need proper trainer")

    trainer=Trainer(args, seed)
    trainer.train()
    logger.info("done training")

def set_struct(cfg: dict):
    root = os.path.abspath(
        os.path.dirname(__file__)
    )
    from datetime import datetime
    now = datetime.now()
    from pytz import timezone
    # apply timezone manually
    now = now.astimezone(timezone('Asia/Seoul'))

    output_dir = os.path.join(
        root,
        "outputs",
        now.strftime("%Y-%m-%d"),
        now.strftime("%H-%M-%S")
    )

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    os.chdir(output_dir)

    job_logging_cfg = {
        'version': 1,
        'formatters': {
            'simple': {
                'format': '[%(asctime)s][%(name)s][%(levelname)s] - %(message)s'
            }
        },
        'handlers': {
            'console': {
                'class': 'logging.StreamHandler', 'formatter': 'simple', 'stream': 'ext://sys.stdout'
            },
            'file': {
                'class': 'logging.FileHandler', 'formatter': 'simple', 'filename': 'train.log'
            }
        },
        'root': {
            'level': 'INFO', 'handlers': ['console', 'file']
            },
        'disable_existing_loggers': False
    }
    logging.config.dictConfig(job_logging_cfg)

    cfg_dir = ".config"
    os.mkdir(cfg_dir)
    os.mkdir(cfg['save_dir'])

    with open(os.path.join(cfg_dir, "config.yaml"), "w") as f:
        for k, v in cfg.items():
            print("{}: {}".format(k, v), file=f)

if __name__ == '__main__':
    main()
