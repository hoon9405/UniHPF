import pandas as pd
import os
import numpy as np
import torch
import tqdm
from transformers import AutoTokenizer
from ICU_class import Event, ColContent
from datetime import date
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
from preprocess_utils import *
from joblib import Parallel, delayed
import pdb
from easydict import EasyDict as edict
from functools import partial
import random
from operator import itemgetter
import pickle

def Table2event_multi(row, columns, table_type, column_select):
    pid = int(row['ID'])
    time = row['TIME']
    
    event = Event(
            src = 'mimic', 
            pid = pid, 
            event_time = time, 
            table = table_type,
            columns =columns,
            value= round(row['value'],4) if column_select==True and type(row['value']) in ['float', 'int'] else 0
        )
    col_contents =  [ ColContent(
                      col= col,
                      content= round_digits(row[col]), 
                    )  for col in columns if row[col] != ' ' ]
    event.col_contents.extend(col_contents)
    
    if column_select ==True:
        event.column_textizing()
        
    return event


# multi event append 
def table2event(
    config,
    args,
    src : str,
    icu : pd.DataFrame,
    table_dict: Dict[str, str],
    drop_cols_dict: Dict[str, List],
    column_select
):  
    df_dict = {}
    event_dict = {}
    for table_name, df_path in table_dict.items(): 
        df = pd.read_csv(df_path, skiprows=lambda i: i>0 and random.random() > 0.05)

        print(f'Table loaded from {df_path} ..')
        print('src = ', src, '\n table = ', table_name)
        
        # preprocess start-----------------------------
        df = filter_ICU_ID(icu, df, config, src)
        df = name_dict(df, table_name, config, args, src, column_select)
        if column_select == True:
            df = column_select_filter(df, config, src, table_name)
            
        elif column_select == False:
            df = used_columns(df, table_name, config, src)
            df = ID_rename(df, table_name, config, src)
        columns = df.columns.drop(drop_cols_dict[src])
        df = ICU_merge_time_filter(icu, df, src) 
        # preprocess finish---------------------------
        
        # fill na 
        df.fillna(' ', inplace=True)
        df.replace('nan', ' ', inplace=True)
        df.replace('n a n', ' ', inplace=True)
        
        # df for sanity check
        df_dict[table_name]=df

        events_list = Parallel(n_jobs=32, verbose=5)(
            delayed(Table2event_multi)(
                df.loc[i],
                columns,
                table_name,
                column_select
                ) 
            for i in range(len(df))
            )
        print('generate event_list finish!')
        event_dict[table_name] = events_list
    return event_dict, df_dict


def icu_class_gen(src, config, args):
    column_select = args.column_select
    print('column select : ', column_select)
    # ICU cohort load
    icu_path = os.path.join(os.path.join(
        args.save_path, 'preprocess', args.icu_type, 'time_gap_'+str(args.time_gap_hours),f'{src}_cohort.pkl')
        )
    icu = pd.read_pickle(icu_path)
    icu.rename(columns = {config['ID'][src] : 'ID'}, inplace=True)
    print(f'ICU loaded from {icu_path}')
    print(icu.info())
    # prepare ICU class
    icu_dict = prepare_ICU_class(icu, src)
    # generate table_dict for table2event
    table_dict = dict()
    for idx in range(len(config['Table'][src])):
        table_name = config['Table'][src][idx]['table_name']
        df_path = os.path.join(args.input_path, src, table_name+'.csv')
        table_dict[table_name] = df_path
    print('table dict check \n', table_dict)
    drops_cols_dict = {
        'mimic3': ['ID', 'TIME'],
        'eicu' : ['ID', 'TIME'],
        'mimic4': ['ID', 'TIME']
    }

    # Generate event_dict from tab
    event_dict, df_dict = table2event(
        config, args, src, icu, table_dict, drops_cols_dict, column_select
        ) #event_dict
    
    #fail check.
    fail = []
    # icu update using events
    icu_dict = prepare_ICU_class(icu, src)
    for table_name, event_list in event_dict.items():
        for event in event_list:
            if event.pid in icu_dict.keys():
                icu_dict[event.pid].events.append(event)
            else:
                fail.append(event) # check fail ID
    print('Add all events to icu_dict finish ! ')
    print("Let's generate data input as numpy file! ")
    
    # min event delete
    min_event_list = []
    for idx, icu in icu_dict.items():
        if len(icu.events) <6:
            min_event_list.append(idx)
    print('delete event list number : ', len(min_event_list))
    for idx in min_event_list:
        del icu_dict[idx]


    # preparation vocab
    vocab = prep_vocab(icu_dict.values(), column_select)
    if column_select == True:
        code_vocab = pd.DataFrame(columns=['code', 'index'])
        code_vocab['code'] = pd.Series(vocab['code_index'].keys())
        code_vocab['index'] = pd.Series(vocab['code_index'].values())
        code_vocab.to_pickle(os.path.join(
            args.save_path, 'input', args.icu_type, src, 'select', f'code_vocab_{src}.pkl'))

    # time bucektting
    tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    time_delta= []
    for icu in tqdm.tqdm(icu_dict.values()):
        icu.make_sub_token_event(vocab)
        get_sample(icu, args) 
        get_time_bucket(icu, args, src)
        time_delta.extend([event.time_delta for event in icu.events])

    bucket_dict = bucketize(pd.Series(time_delta), quant_num=20)

    for icu in tqdm.tqdm(icu_dict.values()):
        convert2bucket(icu, bucket_dict)

    return icu_dict, vocab



def prep_vocab(icu_list, column_select):
    vocab = dict()
    vocab['token_type_col_content'] = dict({
      '[PAD]': 0, '[CLS]': 1, '[Table]': 2, '[Content]' : 3, '[Col]' : 4, '[Time]' : 5
    })
    # PAD = 0 / CLS = 1 / Table = 2 / Col = 3 / Content = 4 / Time = 5
    vocab['token_class'] = dict({
        '[PAD]' : 0, '[CLS]' : 1, '[Time]' : 2, 
    })
    # PAD = 0 / CLS = 1 / Time = 2 / Col_class = 3 ~~~ 
    
    vocab_content = list([])
    vocab_column = list([])
    vocab_code = list([])
    for icu in tqdm.tqdm(icu_list):
        icu_content_list =  [
                col_content.content 
                for event in icu.events 
                for col_content in event.col_contents]
        vocab_content.extend(icu_content_list)
            
    # columns set for col class & Code set
        for event in icu.events:
            vocab_column.extend(list(event.columns))
            if column_select:
                vocab_code.append(event.code_text)
    vocab_content = set(vocab_content)
    vocab_column = set(vocab_column)
    vocab_code = set(vocab_code)
    
    vocab['content_index'] = dict(zip(
        list(vocab_content), range(14, len(vocab_content)+14)
             )
    )
    vocab['code_index'] = dict(zip(
        list(vocab_code), range(4, len(vocab_code)+14)
             )
    )
    vocab['code_index']['[PAD]']=0
    vocab['code_index']['[CLS]']=1
    vocab['code_index']['[SEP]']=2
    vocab['code_index']['[MASK]']=3

    # PAD = 0 / CLS = 1 / NaN = 2 / Time = 3~13 / cotent 14~
    for index, col in enumerate(list(vocab_column)):
        vocab['token_class'][col] = 3 + index                           
    
    return vocab # set