import argparse
import os
import pandas as pd
import h5py
import torch
def get_parser():
    parser = argparse.ArgumentParser()
    #parser.add_argument('--input_path', type=str, default='/nfs_edlab/ghhur/UniHPF/input_test2/')
    parser.add_argument('--input_path', type=str, default='/home/ghhur/.cache/ehr')
    parser.add_argument('--ehr', type=str, default='mimiciii')
    #parser.add_argument('--tasks', type=str, default='creatinine')
    parser.add_argument('--tasks', type=str, default='mortality,los_3day,los_7day,readmission,final_acuity,imminent_discharge,diagnosis,creatinine,bilirubin,platelets,wbc')
    parser.add_argument('--seed', type=int, default=46)
    parser.add_argument('--target', 
            choices=['label', 'max_len', 'unique_code', 'unique_subword'], 
            type=str, 
            default='max_event_len'
    )
    parser.add_argument('--emb_type', choices=['codebase', 'textbase'], type=str, default=None)
    parser.add_argument('--feature', choices=['select', 'whole'], type=str, default=None)
    return parser

def main(args):
    if args.target == 'label':
        label(args)
    elif args.target =='max_len':
        max_len(args)
    elif args.target =='unique_code':
        unique_code(args)
    elif args.target =='unique_subword':
        unique_subword(args)
     
def label(args):
    args.tasks = [i for i in args.tasks.replace(' ','').split(',')]
    cohort = pd.read_csv(os.path.join(args.input_path, f'{args.ehr}_cohort.csv'))
    total = len(cohort)
    
    train = cohort[cohort[f'split_{args.seed}'] == 'train']
    valid = cohort[cohort[f'split_{args.seed}'] == 'valid']
    test = cohort[cohort[f'split_{args.seed}'] == 'test']

    for task in args.tasks:
        print(task)
        print("======= Total =======")
        print('null sample ratio {:.2f} %'.format(cohort[task].isnull().sum()/len(cohort[task])*100))
        print(cohort[task].value_counts(normalize=True).mul(100).round(1).astype(str) + '%')
        print("======= Train =======")
        print('null sample ratio {:.2f} %'.format(train[task].isnull().sum()/len(train[task])*100))
        print(train[task].value_counts(normalize=True).mul(100).round(1).astype(str) + '%')
        print("======= Valid =======")
        print('null sample ratio {:.2f} %'.format(valid[task].isnull().sum()/len(valid[task])*100))
        print(valid[task].value_counts(normalize=True).mul(100).round(1).astype(str) + '%')
        print("======= Test =======")
        print('null sample ratio {:.2f} %'.format(test[task].isnull().sum()/len(test[task])*100))
        print(test[task].value_counts(normalize=True).mul(100).round(1).astype(str) + '%')
        print()
        if task in ['imminent_discharge', 'final_acuity']:
            print("======= Total =======")
            print('null sample ratio {:.2f} %'.format(cohort[task].isnull().sum()))
            print(cohort[task].value_counts())
            print("======= Train =======")
            print('null sample ratio {:.2f} %'.format(train[task].isnull().sum()))
            print(train[task].value_counts())
            print("======= Valid =======")
            print('null sample ratio {:.2f} %'.format(valid[task].isnull().sum()))
            print(valid[task].value_counts())
            print("======= Test =======")
            print('null sample ratio {:.2f} %'.format(test[task].isnull().sum()))
            print(test[task].value_counts())
                
def max_event_len(args):
    args.tasks = [i for i in args.tasks.replace(' ','').split(',')]
    cohort = pd.read_csv(os.path.join(args.input_path, f'{args.ehr}_cohort.csv'))
    total = len(cohort)
    
    train = cohort[cohort[f'split_{args.seed}'] == 'train']
    valid = cohort[cohort[f'split_{args.seed}'] == 'valid']
    test = cohort[cohort[f'split_{args.seed}'] == 'test']

    for task in args.tasks:
        print(task)
        print("======= Total =======")
        print(cohort[task].value_counts(normalize=True).mul(100).round(1).astype(str) + '%')
        print("======= Train =======")
        print(train[task].value_counts(normalize=True).mul(100).round(1).astype(str) + '%')
        print("======= Valid =======")
        print(valid[task].value_counts(normalize=True).mul(100).round(1).astype(str) + '%')
        print("======= Test =======")
        print(test[task].value_counts(normalize=True).mul(100).round(1).astype(str) + '%')
        print()


def max_len(args):
    args.tasks = [i for i in args.tasks.replace(' ','').split(',')]
    cohort = pd.read_csv(os.path.join(args.input_path, f'{args.ehr}_cohort.csv'))
    total = len(cohort)
    
    data_path = os.path.join(args.input_path, f"{args.emb_type}_{args.feature}_{args.ehr}.h5")
    print(f'data load from {args.input_path}, {args.emb_type}_{args.feature}_{args.ehr}.h5')
    data = h5py.File(data_path, "r")['ehr']
    key = list(data.keys())
    word_len_list_hi = []
    word_len_list_fl = []
    event_len_list = []
    for k in key:
        word_len_tmp = torch.where(torch.IntTensor(data[k]['hi'][:,0,:])!=0)[1].tolist()
        word_len_hi = [ word_len_tmp[i] for i in range(len(word_len_tmp)-1)
                       if i<len(word_len_tmp) and (word_len_tmp[i+1] <= word_len_tmp[i]) ]
        word_len_fl = torch.where(torch.IntTensor(data[k]['fl'][0,:])!=0)[0].max().item()
        event_len = torch.IntTensor(data[k]['hi'][:,0,:]).shape[0]
        
        word_len_list_fl.append(word_len_fl)
        word_len_list_hi.extend(word_len_hi)
        event_len_list.append(event_len)
        
    max_word_len_hi=max(word_len_list_hi)
    max_word_len_fl=max(word_len_list_fl)
    max_event_len=max(event_len_list)
    
    
    print(f'max_word_len_hi {max_word_len_hi}')
    print('hi word describe', pd.Series(word_len_list_hi).describe())
    
    print(f'max_seq_len_fl {max_word_len_fl}')
    print('fl seq describe', pd.Series(word_len_list_fl).describe())
    
    print(f'max_event_len {max_event_len}')
    print('event len describe', pd.Series(event_len_list).describe())


def unique_code(args):
    args.tasks = [i for i in args.tasks.replace(' ','').split(',')]
    cohort = pd.read_csv(os.path.join(args.input_path, f'{args.ehr}_cohort.csv'))
    total = len(cohort)
    
    data_path = os.path.join(args.input_path, args.ehr, f'codebase_code2idx_{args.feature}.pkl')
    print(f'data load from {args.input_path}, {args.ehr}, codebase_code2idx_{args.feature}.pkl')
    data = pd.read_pickle(data_path)
    for i, (k,v) in enumerate(data.items()):
        print(k, v)
        if i>100:
            break
    print('length of code ', max(data.values()))

    data_path = os.path.join(args.input_path, f"{args.emb_type}_{args.feature}_{args.ehr}.h5")
    print(f'data load from {args.input_path}, {args.emb_type}_{args.feature}_{args.ehr}.h5')
    data = h5py.File(data_path, "r")['ehr']
    key = list(data.keys())
    code_list = []
    for k in key:
        code = torch.IntTensor(data[k]['hi'][:,0,:]).unique().tolist()
        code_list.extend(code)
    code_unique = list(set(code_list))
    print('code_unique ', len(code_unique))
    print('code_unique max', max(code_unique) )
    breakpoint()
    
def unique_subword(args):
    args.tasks = [i for i in args.tasks.replace(' ','').split(',')]
    cohort = pd.read_csv(os.path.join(args.input_path, f'{args.ehr}_cohort.csv'))
    total = len(cohort)
    
    data_path = os.path.join(args.input_path, f"{args.emb_type}_{args.feature}_{args.ehr}.h5")
    print(f'data load from {args.input_path}, {args.emb_type}_{args.feature}_{args.ehr}.h5')
    data = h5py.File(data_path, "r")['ehr']
    key = list(data.keys())
    word_len_list = []
    for k in key:
        word_len = torch.where(torch.IntTensor(data[k]['hi'][:,0,:])!=0)[1].tolist()
        word_len_list.extend(word_len)
    
    max_word_len=max(word_len_list)
    hist = pd.Series(word_len_list) 
    print(f'max_word_len {max_word_len}')
    print(hist.describe())


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    main(args)
    
# python3 data_distribution.py --ehr mimiciii