import pandas as pd
import os
import numpy as np
import tqdm
import pdb
from ICU_class import ICU
import re

#1. Prepare ICU class from ICU table 
def prepare_ICU_class(icu, src):
    # ICU class generate from ICU table
    if src =='eicu':
        icu['INTIME'] = pd.to_datetime(icu['hospitaladmittime24']) 
        icu['OUTTIME'] = pd.to_datetime(icu['hospitaldischargetime24']) 
        icu['12h_obs'] = icu['INTIME'] + pd.Timedelta(12, "h")
        icu['24h_obs'] = icu['INTIME'] + pd.Timedelta(24, "h")
    icu_dict = {}
    for i in range(len(icu)):
        row = icu.loc[i]
        pid = row['ID']
        icu_class = ICU(
            src =src, 
            pid = pid,
            intime = row['INTIME'], 
            outtime = row['OUTTIME'], 
            icu_num=1, 
            label = {
                'motarlity': row['mortality'],
                'los_3day' : row['los_3day'],
                'los_7day' : row['los_7day'],
                'readmission' : row['readmission']
            }
        )
        icu_dict[pid]= icu_class
    print('1. prepare_ICU_class finish !')
    return icu_dict

#2.FILTER with ID from ICU ID 
def filter_ICU_ID(icu, df, config, src):
    ID = config['ID'][src]
    print('ID unique ', len(icu["ID"].unique()))
    print('before filter ICU', len(df))
    df = df[df[ID].isin(icu['ID'])]
    print('after filter ICU', len(df))
    print('2. filter ICU finish!')
    return df
  
# 3. ITEMID convert with definition file
def name_dict(df, table_name, config, args, src, column_select):
    if src=='eicu' and column_select:
        if table_name =='medication':
            df = eicu_med_revise(df)
        elif table_name =='infusionDrug':
            df = eicu_inf_revise(df)

    file_dict = config['DICT_FILE'][src]
    if table_name in file_dict:
        dict_name= file_dict[table_name][0]
        column_name= file_dict[table_name][1]
        dict_path = os.path.join(args.input_path, src, dict_name+'.csv')
        code_dict = pd.read_csv(dict_path)
        key = code_dict['ITEMID']
        value = code_dict['LABEL']
        code_dict = dict(zip(key,value))
        df[column_name] = df[column_name].map(code_dict)
        print('3. name dict finish!')
    return df

# 4. clarify used columns
def used_columns(df, table_name, config, src):
    table_info_dict = [tmp for tmp in config['Table'][src] if tmp['table_name'] == table_name][0]
    print('table_info dict \n', table_info_dict)
    columns_used = df.columns.tolist()
    print('before remove columns : ', columns_used)
    # remove time excluded, id excluded
    for column in ['time_excluded', 'id_excluded']:
        targets = table_info_dict[column]
        print('targets to remove : ', targets)
        if targets is not None:
            for target in targets:
                if target in columns_used:
                    columns_used.remove(target)
                    print(f'target column {target} removed !')
    print('after remove columns : ', columns_used)
    print('4. used columns finish!')
    return df[columns_used]

# 5. table ID / Time rename
def ID_rename(df, table_name, config, src):
    table_info_dict = [tmp for tmp in config['Table'][src] if tmp['table_name'] == table_name][0]
    print('table_info dict \n', table_info_dict)
    df.rename(columns = {table_info_dict['time_column']: 'TIME'}, inplace=True)
    print(f'Time check {table_name}', 'TIME' in df.columnms)
    df.rename(columns = {config['ID'][src] : 'ID'}, inplace=True)
    print('5. ID_rename finish!')
    return df


# 6. ICU add and time filter (INTIME, OUTTIME)
def ICU_merge_time_filter(icu, df, src):
    icu = icu[['ID', 'INTIME', 'OUTTIME']]
    df = pd.merge(df, icu, how='left', on=['ID'])
    print(f'before time filtered : {len(df)}')
    if src =='mimic':
        df = df[ 
                (df['TIME'] <= df['OUTTIME']) & 
                (df['TIME'] >= df['INTIME']) & 
                (df['TIME'] <= df['INTIME'] + pd.Timedelta(12, unit="h"))
                ]
    elif src=='eicu':
        df = df[ 
                (df['TIME'] <= 12*60) & 
                (df['TIME'] >= 0)
                ]
    print(f'after_time_filtered : {len(df)}')
    print('6. ICU_merge_time_filter finish!')
    return df.reset_index(drop=True)

# Column selection filter
def column_select_filter(df, config, src, table_name):
    select_dict = config['selected'][src][table_name]
    df.rename(columns = select_dict, inplace=True)
    df = df[select_dict.values()]
    print(f'column select src= {src}, table_name = {table_name}, columns = {df.columns}')
    # check value column is float
    df['value']=df['value'].fillna(0)


    return df

def eicu_med_revise(df):
    df['split'] = df['dosage'].apply(lambda x: str(re.sub(',', '',str(x))).split())
    def second(x):
        try:
            if len(pd.to_numeric(x))>=2:
                x = x[1:]
            return x
        except ValueError:
            return x

    df['split'] = df['split'].apply(second).apply(lambda s:' '.join(s))
    punc_dict = str.maketrans('', '', '.-')
    df['uom'] = df['split'].apply(lambda x: re.sub(r'[0-9]', '', x))
    df['uom'] = df['uom'].apply(lambda x: x.translate(punc_dict)).apply(lambda x: x.strip())
    df['uom'] = df['uom'].apply(lambda x: ' ' if x=='' else x)
    
    def hyphens(s):
        if '-' in str(s):
            s = str(s)[str(s).find("-")+1:]
        return s
    df['value'] = df['split'].apply(hyphens)
    df['value'] = df['value'].apply(lambda x: [float(s) for s in re.findall(r'-?\d+\.?\d*', x)])
    df['value'] = df['value'].apply(lambda x: x[-1] if len(x)>0 else x)
    df['value'] = df['value'].apply(lambda d: str(d).replace('[]',' '))
    
    return df


def eicu_inf_revise(df):
    df['split'] = df['drugname'].apply(lambda x: str(x).rsplit('(', maxsplit=1))
    def addp(x):
        if len(x)==2:
            x[1] = '(' + str(x[1])
        return x

    df['split'] = df['split'].apply(addp)
    df['split']=df['split'].apply(lambda x: x +[' '] if len(x)<2 else x)

    df['drugname'] = df['split'].apply(lambda x: x[0])
    df['uom'] = df['split'].apply(lambda x: x[1])
    df['uom'] = df['uom'].apply(lambda s: s[s.find("("):s.find(")")+1])

    toremove = ['()','', '(Unknown)', '(Scale B)', '(Scale A)',  '(Human)', '(ARTERIAL LINE)']

    df['uom'] = df['uom'].apply(lambda uom: ' ' if uom in toremove else uom)
    df = df.drop('split',axis=1)
    
    testing = lambda x: (str(x)[-1].isdigit()) if str(x)!='' else False
    code_with_num = list(pd.Series(df.drugname.unique())[pd.Series(df.drugname.unique()).apply(testing)==True])
    add_unk = lambda s: str(s)+' [UNK]' if s in code_with_num else s
    df['drugname'] = df['drugname'].apply(add_unk)
    
    return df

def prep_vocab(icu_list):
    vocab = dict()
    vocab['token_type_col_content'] = dict({
      '[PAD]': 0, '[CLS]': 1, '[Table]': 2, '[Content]' : 3, '[Col]' : 4, '[Time]' : 5
    })
    # PAD = 0 / CLS = 1 / Table = 2 / Col = 3 / Content = 4 / Time = 5
    vocab['token_class'] = dict({
        '[PAD]' : 0, '[CLS]' : 1, '[Time]' : 2, 
    })
    
    vocab_content = list([])
    vocab_column = list([])
    vocab_code = list([])
    for icu in tqdm.tqdm(icu_list):
        icu_content_list =  [
                col_content.content 
                for event in icu.events 
                for col_content in event.col_contents]
           #if col_content.col in columns ]
        vocab_content.extend(icu_content_list)
            
    # columns set for col class & Code set
        for event in icu.events:
            vocab_column.extend(list(event.columns))
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


def get_sample(icu, args): # input icu / output list[icu] 
    length_events = len(icu.events)
    if length_events <= args.max_event_len:
        events_list = [icu.events]
    else:
        if args.sampling is None:
            events_list = [icu.events[-args.max_event_len:]] 
        elif args.sampling == 'constant':
            walk_len = int(args.max_event_len * args.sampling_prob)
            seq_index_start = [i*walk_len  for i in range((length_events-walk_len)//walk_len)]
            events_list = [icu.events[i:(i+args.max_event_len)] for i in seq_index_start]
   

# Time bucket   
def bucketize(col : pd.Series, quant_num):
    bucket = [col.quantile(q=i/quant_num, interpolation='nearest') for i in range(1, quant_num+1)] 
    bucket.insert(0, 0)
    labels =  list(range(6, 6 +len(bucket)+1)) # pad = 0 cls = 1 sep =2 mask = 3 eos 4 
    unique_labels =  _unique_label(labels, bucket)
    bucket_range_list = []
    for i in range(len(bucket)):
        if i == 0:
            start = 0 
        else: 
            start = bucket[i-1] +1
        end = bucket[i]+1
        
        bucket_range_list.append((start, end))
        
    bucket_dict = dict(zip(bucket_range_list, unique_labels))
    return bucket_dict


def _unique_label(label, bin_list):
    remove_list = []
    for idx, b in enumerate(bin_list):
        if b not in remove_list:
            cnt = bin_list.count(b)
            if cnt > 1:
                central = (idx  + idx + cnt -1)//2
                value = label[central]      
                label[idx:idx+cnt] = [value] * cnt
                remove_list.append(b)

    label_unique = []
    for x in label:
        if x not in label_unique:
            label_unique.append(x)
            
    return label_unique


def get_time_bucket(icu, args, src):
    intime = icu.intime.to_datetime64()
    for event in icu.events:
        if src=='eicu':
            event.time_delta = event.event_time
        elif src=='mimic3' or src=='mimic4':
            if type(event.event_time) == object:
                event_time = np.datetime64(event.event_time)
           
            
            time_delta = (event_time -intime).astype('timedelta64[m]')
            event.time_delta = time_delta.astype('int')
 
   

def convert2bucket(icu, bucket_dict):
    for event in icu.events:
        event.time_bucket = None
        for k,v in bucket_dict.items():
            if k[0] <= event.time_delta <k[1]:
                event.time_bucket = v
    return icu
    


# ICU class utils
def digit_split(digits : str):
    return [' '.join(d) for d in digits]


def isnumeric(text):
    '''returns True if string s is numeric'''    
            
    return all(s in "0123456789." for s in text) and any(s in "0123456789" for s in text)

def digit_split_in_text(text : str):
    join_list = []
    split = text.split()
    for i, d in enumerate(split):
        if isnumeric(d):
            while d.count('.') > 1:
                target = d.rfind('.')
                if target  == (len(d)-1) :
                    d = d[:target]
                else:
                    d = d[:target] + d[(target+1):]
            join_list.append(digit_split(d))
            if not i==(len(split)-1) and isnumeric(split[i+1]):
                    join_list.append(['|'])
        else:
            join_list.append([d])

    
    return ' '.join(sum(join_list, []))

def split(word):
    return [char for char in word]

    #split and round
def round_digits(digit : str or float or int):
    if isinstance(digit, str):
        return digit_split_in_text(digit)
    elif digit is np.NAN:
        return ' '
    elif isinstance(digit, float):
        return " ".join(split(str(round(digit, 4))))
    elif isinstance(digit, int):
        return " ".join(split(str(digit)))
    elif isinstance(digit, np.int64):
        return str(digit)
    else: 
        return digit

def text_digit_round(text_list):
    if '.' in text_list:
        decimal_point =  text_list.index('.')
        if len(text_list[decimal_point:])> 5:
            return text_list[:decimal_point+5]
        else:
            return text_list
    else:
        return text_list