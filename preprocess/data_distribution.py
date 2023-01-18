import argparse
import os
import pandas as pd

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, default='/nfs_edlab/ghhur/UniHPF/tmp/')
    parser.add_argument('--ehr', type=str, default='mimiciii')
    #parser.add_argument('--tasks', type=str, default='creatinine')
    parser.add_argument('--tasks', type=str, default='mortality,los_3day,los_7day,readmission,final_acuity,imminent_discharge,diagnosis,creatinine,bilirubin,platelets,wbc')
    parser.add_argument('--seed', type=int, default=46)
    parser.add_argument('--target', 
            choices=['label', 'max_event_len', 'max_word_len', 'unique_code', 'unique_subword'], 
            type=str, 
            default='max_event_len'
    )
    return parser

def main(args):
    if args.target == 'label':
        label(args)
    elif args.target =='max_event_len':
        max_event_len(args)
    elif args.target =='max_word_len':
        max_word_len(args)
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


def max_word_len(args):
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


def unique_code(args):
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

def unique_subword(args):
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


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    main(args)
    
# python3 data_distribution.py --ehr mimiciii