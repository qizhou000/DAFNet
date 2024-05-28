#%%
from abc import ABC, abstractmethod
from dataclasses import dataclass
import json, yaml
from transformers import  AutoTokenizer, AutoModel
from utils.utils import set_tokenizer_pad_id
from typing import Dict, List, Tuple



@dataclass
class EditorConfig:
    @classmethod
    def from_json(cls, fpath):
        with open(fpath, "r") as f:
            data = json.load(f)
        return cls(**data)

    @classmethod
    def from_yaml(cls, fpath):
        with open(fpath, "r") as f:
            data = yaml.safe_load(f)
        return cls(**data)

 
class BaseEditor(ABC):
    def __init__(self, model:AutoModel, tokenizer:AutoTokenizer, device='cuda'):
        set_tokenizer_pad_id(tokenizer)
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        assert self.if_model_decoder_only()  
    
    @abstractmethod
    def name_of_editor_and_model(self)->Tuple[str, str]:
        '''
        Assume:
        return editor_name:str, model_name:str
        '''

    def if_model_decoder_only(self)->bool:
        if self.model.config.is_encoder_decoder:
            return False
        return True

    @abstractmethod
    def if_can_batch_edit(self)->bool:
        pass

    @abstractmethod
    def edit_one_piece(self, request:Dict):
        '''
        Assume: 
        requests = {'prompt': str, 'target_new': str, ...}
        '''

    @abstractmethod
    def edit_batch(self, requests:List[Dict]):
        '''
        Assume: 
        requests = [
          {'prompt': str, 'target_new': str, ...},
          {'prompt': str, 'target_new': str, ...},
          ...
        ]
        '''
    
    @abstractmethod
    def restore_to_original_model(self):
        '''
        A method for restoring the original model weights after editing with as 
        low GPU memory usage as possible.
        '''

