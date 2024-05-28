from transformers import  AutoTokenizer
from typing import List
import os

def set_tokenizer_pad_id(tokenizer:AutoTokenizer):
    if tokenizer.pad_token_id == None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        print('Set [pad_token] as [eos_token].')

def get_model_path_editor_config_name(model_name:str):
    model_name = model_name.lower()
    if 'gptj' in model_name or 'gpt-j' in model_name:
        config_name = 'gpt-j-6b.yaml'
        model_path = 'models/gpt-j-6b'
    elif 'llama' in model_name:
        config_name = 'llama-7b.yaml'
        model_path = 'models/llama-2-7b-hf'
    elif 'gpt2' in model_name:
        config_name = 'gpt2-xl.yaml'
        model_path = 'models/gpt2-xl'
    else:
        raise
    return model_path, config_name

 
def get_editor(editor_name:str, edit_model_name:str, device:int, 
               extra_devices:List[int] = [], editor_ckpt_path = None):
    from transformers import  AutoTokenizer, AutoModelForCausalLM
    from editors.dafnet import DAFNet, DAFNetConfig
    editor_name = editor_name.lower() 
    model_path, config_name = get_model_path_editor_config_name(edit_model_name)
    model = AutoModelForCausalLM.from_pretrained(model_path) 
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    config_path = os.path.join('configs', editor_name, config_name)
    if editor_name == 'dafnet':
        if len(extra_devices) == 0:
            device_gradient_signal = device
            devices_aux_models = [device]
        elif len(extra_devices) == 1:
            device_gradient_signal = extra_devices[0]
            devices_aux_models = extra_devices
        else:
            device_gradient_signal = extra_devices[0]
            devices_aux_models = extra_devices[1:]
        config = DAFNetConfig.from_yaml(config_path)
        editor = DAFNet(model, tokenizer, config, device, device_gradient_signal, devices_aux_models, editor_ckpt_path)
    else:
        raise
    return editor
