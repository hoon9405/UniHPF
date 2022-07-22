import pandas as pd
import os
import numpy as np
import tqdm
from covert2numpy import convert2numpy_CodeEmb, convert2numpy_DescEmb
from icuclass_gen import icu_class_gen
from datasets_construct import *
from easydict import EasyDict as edict
from functools import partial
import random
import argparse

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, default='/home/edlab/ghhur/Pretrained_DescEmb/raw_file/')
    parser.add_argument('--save_path', type=str, default='/home/edlab/ghhur/Pretrained_DescEmb/data/')
    parser.add_argument('--max_event_len', type=int, default=150)
    parser.add_argument('--min_event_len', type=int, default=5)
    parser.add_argument('--event_window_hours',  type=int, default=12)
    parser.add_argument('--time_gap_hours',  type=int, default=12)
    parser.add_argument('--pred_window_hours',  type=int, default=48)
    parser.add_argument('--icu_type', type=str, choices=['micu', 'ticu', 'ticu_multi'], default='ticu')
    parser.add_argument('--src_dataset', type=str, default='mimic3 eicu mimic4')
    parser.add_argument('--column_select', action='store_true', default=True)
    parser.add_argument('--sampling', type=str, choices=[None, 'constant'], default=None)
    return parser

def main():
    args = get_parser().parse_args()
    # file names
    config = {
    'Table':{
        'mimic3': 
        [
            {'table_name':'LABEVENTS', 'time_column': 'CHARTTIME', 'table_type':'lab',
             'time_excluded': ['ENDTIME'], 'id_excluded':['ROW_ID', 'ICUSTAY_ID', 'SUBJECT_ID'] 
            },
            {'table_name':'PRESCRIPTIONS', 'time_column': 'STARTDATE', 'table_type':'med',
             'time_excluded': ['ENDDATE'], 'id_excluded':['GSN','NDC', 'ROW_ID', 'ICUSTAY_ID', 'SUBJECT_ID']
            },
            {'table_name':'INPUTEVENTS_MV', 'time_column': 'STARTTIME', 'table_type':'inf',
             'time_excluded': ['ENDTIME', 'STORETIME'], 'id_excluded':['CGID','ORDERID','LINKORDERID', 'ROW_ID', 'ICUSTAY_ID', 'SUBJECT_ID']
             },
            {'table_name':'INPUTEVENTS_CV', 'time_column': 'CHARTTIME', 'table_type':'inf',
             'time_excluded': ['STORETIME'], 'id_excluded':['CGID','ORDERID','LINKORDERID', 'ROW_ID', 'ICUSTAY_ID', 'SUBJECT_ID']
             }
          ],
        'eicu':
         [
             {'table_name':'lab', 'time_column': 'labresultoffset', 'table_type':'lab',
              'time_excluded': ['labresultrevisedoffset'], 'id_excluded': ['labid']
             },
             {'table_name':'medication', 'time_column': 'drugstartoffset', 'table_type':'med',
              'time_excluded': ['drugorderoffset, drugstopoffset'], 'id_excluded': ['medicationid', 'GTC', 'drughiclseqno']
             },
             {'table_name':'infusionDrug',  'time_column': 'infusionoffset', 'table_type':'inf',
              'time_excluded': None, 'id_excluded':None
             } 
          ],
        'mimic4':
         [
          {'table_name':'labevents', 'time_column': 'CHARTTIME', 'table_type':'lab',
             'time_excluded': ['STORETIME', 'INTIME', 'OUTTIME', 'common_time'], 
               'id_excluded':['ICUSTAY_ID', 'SUBJECT_ID', 'SPECIMEN_ID'] 
            },
            {'table_name':'prescriptions', 'time_column': 'STARTTIME', 'table_type':'med',
             'time_excluded': ['STOPTIME',  'INTIME', 'OUTTIME', 'common_time'], 
             'id_excluded':['GSN','NDC', 'ICUSTAY_ID', 'SUBJECT_ID', 'PHARMACY_ID']
            },
            {'table_name':'inputevents', 'time_column': 'STARTTIME', 'table_type':'inf',
             'time_excluded': ['ENDTIME', 'STORETIME', 'INTIME', 'OUTTIME', 'common_time'], 
             'id_excluded':['ORDERID','LINKORDERID', 'ICUSTAY_ID', 'SUBJECT_ID']
             }

          ]
        },
    'selected':{
        'mimic3': {
                'LABEVENTS':{
                        'HADM_ID':'ID',
                        'CHARTTIME':'TIME',
                        'ITEMID':'code',
                        'VALUENUM':'value',
                        'VALUEUOM':'uom',
                    },
                'PRESCRIPTIONS':{
                    'HADM_ID':'ID',
                    'STARTDATE':'TIME',
                    'DRUG':'code', 
                    'ROUTE':'route', 
                    'PROD_STRENGTH':'prod',
                    'DOSE_VAL_RX':'value',
                    'DOSE_UNIT_RX':'uom',
                    },                                      
                'INPUTEVENTS_MV':{
                    'HADM_ID':'ID',
                    'STARTTIME':'TIME', 
                    'ITEMID':'code',
                    'RATE':'value', 
                    'RATEUOM':'uom',
                    },
                'INPUTEVENTS_CV':{
                    'HADM_ID':'ID',
                    'CHARTTIME':'TIME', 
                    'ITEMID':'code',
                    'RATE':'value', 
                    'RATEUOM':'uom',
                    }
        },

        'eicu':  {
            'lab':{
                'patientunitstayid':'ID', 
                'labresultoffset':'TIME',
                'labname':'code',
                'labresult':'value',
                'labmeasurenamesystem':'uom'
                },
            'medication':{
                'patientunitstayid':'ID',
                'drugstartoffset':'TIME',
                'drugname':'code', 
                'routeadmin':'route',
                 },      
            'infusionDrug':{
                'patientunitstayid':'ID',
                'infusionoffset':'TIME',
                'drugname':'code',
                'infusionrate':'value'
                 }
        },
        'mimic4': {
                'labevents':{
                        'HADM_ID':'ID',
                        'CHARTTIME':'TIME',
                        'ITEMID':'code',
                        'VALUENUM':'value',
                        'VALUEUOM':'uom',
                    },
                'prescriptions':{
                    'HADM_ID':'ID',
                    'STARTDATE':'TIME',
                    'DRUG':'code', 
                    'PROD_STRENGTH':'prod',
                    'DOSE_VAL_RX':'value',
                    'DOSE_UNIT_RX':'uom',
                    },                                      
                'inputevents':{
                    'HADM_ID':'ID',
                    'STARTTIME':'TIME', 
                    'DRUG':'code',
                    'RATE':'value', 
                    'RATEUOM':'uom',
                    },
           }
    
   },
    'DICT_FILE':{
            'mimic3':{
                'LABEVENTS':['D_LABITEMS','ITEMID'], 
                'INPUTEVENTS_CV':['D_ITEMS', 'ITEMID'], 
                'INPUTEVENTS_MV':['D_ITEMS', 'ITEMID']
                },
        
            'eicu':{
            },
        
            'mimic4':{
                'labevents':['d_labitems','ITEMID'], 
                'inputevents':['d_items', 'ITEMID'], 
                },
        
        
    },
    'ID':{
        'mimic3':
              'HADM_ID',
        
        'eicu':
            'patientunitstayid',
        
        'mimic4':
            'HADM_ID'
        },
    
}
    
    #1.ICU dataset construction
    # input : ADMISSION / PATIENT / ICU 
    # output : mimic_cohort.pkl
    print('create MIMIC3 ICU start!')
    create_MIMIC3_ICU(args)
    print('create eICU ICU start!')
    create_eICU_ICU(args)
    print('create MIMIC4 ICU start!')
    create_MIMIC4_ICU(args)

    #2. Table to ICU class 
    input : mimic3_corhot.pkl / LAB.csv / PRESCRIPTION.csv / INPUTEVENTS.csv 
    output : mimic3_icu_class.pkl
    src_dataset = args.src_dataset.split(' ')
    print(f'icu class generation target list : ')
    for src in src_dataset:
        print(f'icu_class gen start : {src} ')
        icu_dict, vocab= icu_class_gen(src, config, args)
        #3. Generating Numpy file from ICU class
        convert2numpy_CodeEmb(icu_dict, src,args, vocab)
        convert2numpy_DescEmb(icu_dict, src, args, vocab)
        print(f'{src} preprocess finish!!')
    

if __name__ == '__main__':
    main()

    