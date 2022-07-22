import pandas as pd
import os
import numpy as np
import tqdm


def DescEmb_preparation(vocab,
                        icu,
                        
):
    target_data = ['input', 'type', 'dpe']
    #target_data = ['input', 'type', 'dpe', 'text']
    time_bucket_type_ct = vocab['token_type_col_content']['[Time]']
    time_bucket_type_cc = vocab['token_class']['[Time]']
    
    event_ct_dict = dict(zip(target_data, [[] for i in range(4)]))
    event_cc_dict = dict(zip(target_data, [[] for i in range(4)]))
    
    
    for e in icu.events:
        
        time_bucket_input = e.time_bucket
        ct_dict =  dict(zip(target_data, [[] for i in range(4)]))
        cc_dict =  dict(zip(target_data, [[] for i in range(4)]))
    
        for c in e.col_contents:
            for k,v in _ct_ct(c).items():
                ct_dict[k].extend(v)
            for k,v in _cc_ct(c).items():   
                cc_dict[k].extend(v)
                
        ct_dict['input'] = e.table_tok_id + ct_dict['input'] + [time_bucket_input]
        ct_dict['type'] = e.table_type_id + ct_dict['type'] + [time_bucket_type_ct]
        ct_dict['dpe'] = [0]*len(e.table_tok_id) + ct_dict['dpe'] + [0]
        #ct_dict['text'] = e.table_tok + ct_dict['text'] + [time_delta]
        
        cc_dict['input'] = cc_dict['input'] + [time_bucket_input]
        cc_dict['type'] =  cc_dict['type'] + [time_bucket_type_cc]
        cc_dict['dpe'] = cc_dict['dpe'] + [0]
       # cc_dict['text'] = cc_dict['text'] + [time_delta]
    
        for t in target_data:
            event_ct_dict[t].append(ct_dict[t])
            event_cc_dict[t].append(cc_dict[t])

    return [
        event_ct_dict, 
        event_cc_dict
    ]


def _ct_ct(c):
    ct_ct_input = c.col_tok_id + c.content_tok_id
    if c.content_tok_id == 0:
        print('content_tok_id length is 0', c, c.content_tok_id)
    type_col = [c.token_type_col]*len(c.col_tok_id)
    type_content = [c.token_type_content]*len(c.content_tok_id)
    ct_ct_type = type_col + type_content
    ct_ct_DPE = [0]*len(c.col_tok_id) + c.dpe
    ct_ct_text = c.col_tok + c.content_tok
    
    assert len(ct_ct_input) == len(ct_ct_type) == len(ct_ct_DPE), f'input token: {len(ct_ct_input)}, type token: {len(ct_ct_type)}, DPE : {len(ct_ct_DPE)}, {c}, {c.dpe} ' 
    return {
        'input' : ct_ct_input,
        'type' : ct_ct_type,
        'dpe' : ct_ct_DPE,
        'text' : ct_ct_text,
    }

def _cc_ct(c): 
    cc_ct_input = c.content_tok_id
    cc_ct_type = [c.col_class_type_col]*len(c.content_tok_id)
    cc_ct_DPE = c.dpe
    cc_ct_text = c.content_tok
    
    assert len(cc_ct_input) == len(cc_ct_type) == len(cc_ct_DPE), f'input token: {len(cc_ct_input)} type token: {len(cc_ct_type)}, DPE : {len(cc_ct_DPE)}'
    return {
        'input' : cc_ct_input,
        'type' : cc_ct_type,
        'dpe' : cc_ct_DPE,
        'text' : cc_ct_text
    }

def _pad(event, max_len, pad_token):
    pad_length = max_len - len(event) -2 
    if pad_length > 0:
        return [pad_token] * pad_length
    else:
        return []

def token_decorate(seq, data_type, form, args):
    # seq = > (S, W)
    token_dict = {
        'input': (101, 102, 0),
        'type': (1, 0, 0),   
        'dpe': (0, 0, 0),
        'text' : ('[CLS]', '[EOS]', '[PAD]')
    }
    CLS, EOS, PAD = token_dict[data_type]
    
    if form == 'hi':
        if args.pad==True:
            seq=[[CLS] + event[-(args.word_max_length-2):] + [EOS] + _pad(event, args.word_max_length, PAD) for event in seq]
        else:
            seq=[[CLS] + event + [EOS] for event in seq]
        
        if len(seq) < args.max_event_len and args.pad:
            pad_event = args.max_event_len - len(seq)
            seq.extend([[CLS, EOS] + [PAD]*(args.word_max_length-2)]*pad_event)
        elif len(seq) > args.max_event_len and args.pad:
            seq = seq[-args.max_event_len:]
         
            
    elif form =='uni':
        linearized_events = sum(seq, [])
        if args.pad:
            linearized_events = linearized_events[-(args.unified_max_length-2):]
        seq= [CLS] + linearized_events + [EOS]
        if args.pad:
             seq +=_pad(linearized_events, args.unified_max_length, PAD)
    return seq


def convert2numpy_DescEmb(icu_dict, src, args, vocab):
    target_data = ['input', 'type', 'dpe', 'text']

    icu_ct_hi_dict = dict(zip(target_data, [[] for i in range(4)]))
    icu_cc_hi_dict = dict(zip(target_data, [[] for i in range(4)]))
    icu_ct_uni_dict = dict(zip(target_data, [[] for i in range(4)]))
    icu_cc_uni_dict = dict(zip(target_data, [[] for i in range(4)]))
    print('finish')

    icu_label_list = []
    for idx, icu in tqdm.tqdm(enumerate(icu_dict.values())):
        pid_icu_num = {'pid': icu.pid, 'icu_num': icu.icu_num}
        pid_icu_num.update(icu.label)
        icu_label_list.append(pid_icu_num)
        descemb_ct, descemb_cc = DescEmb_preparation(vocab,icu)
    
        for target in target_data:
            icu_ct_hi_dict[target].append(token_decorate(descemb_ct[target], target, 'hi', args))
            icu_cc_hi_dict[target].append(token_decorate(descemb_cc[target],  target, 'hi', args))
            icu_ct_uni_dict[target].append(token_decorate(descemb_ct[target], target, 'uni', args))
            icu_cc_uni_dict[target].append(token_decorate(descemb_cc[target], target, 'uni', args))
        
        print(np.array(icu_ct_hi_dict['input'], dtype='int16').shape)
        print(np.array(icu_cc_hi_dict['input'], dtype='int16').shape)
        print(np.array(icu_ct_uni_dict['input'], dtype='int16').shape)
        print(np.array(icu_ct_uni_dict['input'], dtype='int16').shape)

    print(f'{src} convert2numpy_DescEmb finish !')

def CodeEmb_preparation(icu, vocab, max_events):
    for e in icu.events:
        e.code_index_gen(vocab)
    code_list = [e.code_index for e in icu.events]
    value_list = [e.value for e in icu.events]
    code_list.insert(0, vocab['code_index']['[CLS]'])
    value_list.insert(0, vocab['code_index']['[CLS]'])
    # pad
    if max_events > len(code_list)-1:
        pad_len = max_events -(len(code_list)-1)
        code_list.extend(vocab['code_index']['[PAD]']*pad_len)
        value_list.extend(vocab['code_index']['[PAD]']*pad_len)
    return code_list, value_list


def convert2numpy_CodeEmb(icu_dict, src, args, vocab):
    code_numpy_list = []
    value_numpy_list = []
    for pid, icu in icu_dict.items():
        code_list, value_list = convert2numpy_CodeEmb(icu, vocab, args.max_event_len)
        code_numpy_list.append(code_list)
        value_numpy_list.append(value_list)
        
    code_numpy = np.array(code_numpy_list).astype('int16')
    value_numpy = np.array(value_numpy_list).astype('int16')
    np.save(os.path.join(args.out_dir, f'codeemb_{src}'), code_numpy)
    np.save(os.path.join(args.out_dir, f'codeemb_value_{src}'), value_numpy)

    print(f'{src} convert2numpy_CodeEMb finish !')