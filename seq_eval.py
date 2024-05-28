#%%
import os, argparse, sys
from typing import Any, Dict, List, Tuple
from utils.utils import get_editor

def eval_sequential_edit(editor, eval_dataset = 'ZSRE', seq_edit_n:int = 10, 
        data_sample_n:int = None, shuffle = True, seed = 0,
        zsre_path:str = 'data/evaluation/zsre/zsre_mend_eval.json', 
        cf_path:str = 'data/evaluation/cf/counterfact-edit.json', 
        ripple_effect_path:str = 'data/evaluation/ripple_effect/ripple_effect.json', 
        extra_evaluation_name = None):
    from utils.data import TestSampleList
    from evaluation.evaluation import Evaluation 
    if eval_dataset == 'ZSRE':
        evaluation_name = 'ZSRE'
        test_sample_list = TestSampleList.zsre(zsre_path, data_sample_n, shuffle, seed)
    elif eval_dataset == 'CF':
        evaluation_name = 'CF'
        test_sample_list = TestSampleList.counterfact(cf_path, data_sample_n, shuffle, seed)
    elif eval_dataset == 'CF+':
        evaluation_name = 'CF_plus'
        test_sample_list = TestSampleList.counterfact_plus(cf_path, data_sample_n, shuffle, seed)
    elif eval_dataset == 'RIPE':
        evaluation_name = 'RIPE'
        test_sample_list = TestSampleList.ripple_effect(ripple_effect_path, data_sample_n, shuffle, seed)
    else:
        raise
    if extra_evaluation_name != None:
        evaluation_name += '-' + extra_evaluation_name
    ev = Evaluation(editor, test_sample_list, evaluation_name) 
    if seq_edit_n == 1:
        ev.evaluate_single_edit()
    else:
        ev.evaluate_sequential_edit(seq_edit_n, True, seed) 

def test(editor, request):
    from editors.utils.generate import generate_fast
    editor.edit_one_piece(request)
    result = generate_fast(editor.model, editor.tokenizer, [request['prompt']], 1, 1, 30)
    print(result)
    editor.restore_to_original_model()
    result = generate_fast(editor.model, editor.tokenizer, [request['prompt']], 1, 1, 30)
    print(result)


def has_evaluated(editor_name:str, model_name:str, data_name:str, edit_n:int):
    editor_name = editor_name.lower()
    model_name = model_name.lower()
    if 'llama' in model_name:
        model_name = 'llama-7b'
    elif 'gpt-j' in model_name or 'gptj' in model_name:
        model_name = 'gpt-j-6b'
    elif 'gpt2' in model_name:
        model_name = 'gpt2-xl'
    else:
        raise
    if 'CF+' in data_name:
        data_name = 'CF_plus'
    if edit_n == 1:
        dir_name = 'single_edit'
    else:
        dir_name = 'sequential_edit_'+str(edit_n)+'/non_random'
    path = os.path.join('eval_results', editor_name, model_name, data_name, dir_name, 'results.json')
    if os.path.exists(path):
        return True
    print('Save path: ', path)
    return False

def get_attr():
    parser = argparse.ArgumentParser()
    parser.add_argument('-en', '--editor_name', type=str, help='Editor name: DAFNet.', required=True)
    parser.add_argument('-mn', '--edit_model_name', type=str, help='Editing model name: GPT-J, LLAMA...', required=True)
    parser.add_argument('-dvc', '--device', type=int, help='CUDA device for editing.', required=True)
    parser.add_argument('-edvc', '--extra_devices', type=int, nargs='+', default = [], help='Extra CUDA devices, default empty.')
    parser.add_argument('-ckpt', '--editor_ckpt_path', type=str, default = None, help='For Editors that needs training.')
    parser.add_argument('-eds', '--eval_dataset', type=str, default = 'ZSRE', help = 'Evaluating dataset.')
    parser.add_argument('-sen', '--seq_edit_n', type=int, default = 10, help = 'Sequential editing number.')
    parser.add_argument('-dsn', '--data_sample_n', type=int, default = None, help = 'Sample number for evaluation.')
    parser.add_argument('-sd', '--seed', type=int, default = 0, help = 'Random seed.')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    cfg = get_attr()
    if has_evaluated(cfg.editor_name, cfg.edit_model_name, cfg.eval_dataset, cfg.seq_edit_n):
        sys.exit()
    print(cfg)
    editor = get_editor(cfg.editor_name, cfg.edit_model_name, cfg.device, cfg.extra_devices, cfg.editor_ckpt_path)
    eval_sequential_edit(editor, cfg.eval_dataset, cfg.seq_edit_n, cfg.data_sample_n, True, cfg.seed, extra_evaluation_name = None) 


