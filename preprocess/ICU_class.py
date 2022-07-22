import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Set
from transformers import AutoTokenizer
from operator import itemgetter
from itertools import groupby

tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")

@dataclass(unsafe_hash=True)
class ColContent:
    col : str
    content : str
        
    col_tok : List[str] = field(default_factory=list)
    content_tok : List[str] = field(default_factory=list)
    col_tok_id : List[int] = field(default_factory=list)
    content_tok_id : List[int] = field(default_factory=list)
    dpe : List[int] = field(default_factory=list)
        
    # Vocab
    content_index : int = 0
    col_class_type_col : int = 0
    token_type_content : int = 0
    token_type_col : int = 0
        
    # Bio_clinical_bert
    number_token_list: List[int] = field(default_factory=list)
    integer_start : int = 5
    
    def __post_init__(self):
        self.content_tok_id = tokenizer.encode(self.content)[1:-1] if type(self.content) is str else self.content
        self.col_tok_id = tokenizer.encode(self.col)[1:-1] if type(self.col) is str else self.col
        self.content_tok = [tokenizer.decode(token) for token in self.content_tok_id] if type(self.content) is str else self.content
        self.col_tok = [tokenizer.decode(token) for token in self.col_tok_id] if type(self.col) is str else self.col
        
        self.number_token_list = [121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 119] 
        self.dpe = self.make_dpe(self.content_tok_id)
        
    def __len__(self):
        return {'col_tok':len(self.col_tok), 'content_tok':len(self.content_tok)}

    def get_content_index(self, vocab):
        self.content_index = vocab['content_index'][self.content]

    def get_token_class(self, vocab):
        self.token_class = vocab['token_class'][self.col] # following column 
    
    def get_token_type(self, vocab):
        self.token_type_content = vocab['token_type_col_content']['[Content]']
        self.token_type_col = vocab['token_type_col_content']['[Col]']
        
    
    def make_dpe(self, target):
        if type(target) is not list:
            return None
        dpe = [0]*len(target)
        scanning = [pos for pos, char in enumerate(target) if char in self.number_token_list]

        #grouping
        ranges = []
        for k,g in groupby(enumerate(scanning),lambda x:x[0]-x[1]):
            group = (map(itemgetter(1),g))
            group = list(map(int,group))
            ranges.append((group[0],group[-1]))

        # making dpe     
        dpe_group_list = []
        for (start, end) in ranges:
            group = target[start:end+1]
            digit_index = [pos for pos, char in enumerate(group) if char == self.number_token_list[-1]] #digit_token
            assert len(digit_index) < 3, "More than 3 digit index in sing group"
            if len(digit_index)==2:
                # ex) 1. 0 2 5.  ..53.24
                if digit_index[0] == 0:
                    group= group[1:]
                    digit_index = digit_index[1:]
                    start=start+1
                else:
                    raise AssertionError(f"The value is invalid {target} {self.content} {ranges} {digit_index}")
            # case seperate if digit or integer only
            if len(digit_index)== 0:
                dpe_group = [self.integer_start-1+len(group)-i for i in range(len(group))]
            else:
                dpe_int = [self.integer_start-1+len(group[:digit_index[0]])-i for i in range(len(group[:digit_index[0]]))]
                dpe_digit = [i for i in range(len(group[digit_index[0]:]))]
                dpe_group = dpe_int + dpe_digit
            dpe_group_list.append(((start,end), dpe_group))

        for (start, end), dpe_group in dpe_group_list:
            dpe[start:end+1] = dpe_group

        return dpe    
        

@dataclass(unsafe_hash=True)
class Event:
    src : str 
    pid : int
    event_time : pd.datetime
    table : str
    columns : List[str]
    value : float
    
    table_tok_id : List[int] = field(default_factory=list)
    table_tok : List[str] = field(default_factory=list)
    table_type_id : List[int] = field(default_factory=list)
    col_contents : List[ColContent] = field(default_factory=list)

        
    def __len__(self):
        return len(self.col_contents)
    
    def __post_init__(self):
        self.table_tok_id = tokenizer.encode(self.table)[1:-1] if type(self.table) is str else self.table
        self.table_tok = [tokenizer.decode(token) for token in self.table_tok_id] if type(self.table) is str else self.table
       
    def column_textizing(self):
        self.code_text = " ".join([col_content.content for col_content in self.col_contents if col_content.col !='value'])
    
    def make_sub_token_colcontent(self, vocab):
        self.table_type_id = [vocab['token_type_col_content']['[Table]']]*len(self.table_tok_id)
        list(map(lambda content : content.get_content_index(vocab), self.col_contents))
        list(map(lambda content : content.get_token_class(vocab), self.col_contents))
        list(map(lambda content : content.get_token_type(vocab), self.col_contents))
    
    def coode_index_gen(self, vocab):
        self.code_index = vocab['code_index'][self.code_text]
        

@dataclass(unsafe_hash=True)
class ICU:
    src : str
    pid : int
    intime : pd.datetime
    outtime: pd.datetime
    icu_num : int
    label : Dict
    events : List[Event] = field(default_factory=list)
    
    def __len__(self):
        return len(self.events)

    def make_sub_token_event(self, vocab):
        list(map(lambda event : event.make_sub_token_colcontent(vocab), self.events))
        list(map(lambda event : event.make_sub_token_colcontent(vocab), self.events))
        list(map(lambda event : event.make_sub_token_colcontent(vocab), self.events))






