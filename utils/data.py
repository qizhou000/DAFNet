#%%
import threading, time
import numpy as np
import torch, os, json, re
from typing import Dict, List, Tuple, Union
from torch.nn.utils.rnn import pad_sequence
from copy import deepcopy
from utils.utils import set_tokenizer_pad_id
from transformers import  AutoTokenizer 
from datasets import load_dataset
from queue import Queue 
from collections import defaultdict
 

################################################################################
# A Parallel Dataset class: Preprocessing and generating data batches through  #
# sub processes.                                                               #
################################################################################
class ParallelDataset():
    def __init__(self, sample_count:int, get_data_by_ids_func,
        batch_size:Union[int, List[int]] = 256, shuffle = True, 
        buffer_size = 64, drop_last = False, random_seed = None) -> None:
        '''
        Basic dataset class, subclasses need to implement the `__get_data_by_ids__(self, ids)` 
            function for retrieving data by ID. The preprocessing of data obtained 
            by ID can be centralized in this function because this class fetches 
            data using extra thread and stores the acquired data in a queue.
        `sample_count`: Total number of samples
        `get_data_by_ids_func`: Function to retrieve data by ID.
        `batch_size`: Batch size for generating data each time. If it is a list or 
            a one-dimensional array, random numbers are drawn from the list each 
            time to form a batch.
        `shuffle`: Whether to shuffle the dataset
        `buffer_size`: Buffer size for subprocesses to store data
        `drop_last`: Whether to discard the remaining data that does not form a 
            complete batch. If not discarded, it will be combined with the data 
            from the next epoch for output.
        '''
        self.sample_count = sample_count
        self.set_batch_size(batch_size)
        # batch_size = [batch_size] if type(batch_size) == int else batch_size
        # self.batch_size = np.array([min(bs, sample_count) for bs in batch_size])
        self.shuffle = shuffle
        self.rng = np.random.default_rng(random_seed)
        self.select_ids = np.array(range(sample_count))
        if shuffle: 
            self.rng.shuffle(self.select_ids)
        self.drop_last = drop_last
        self.now_buffer_i = 0 # the idex of data has added into buffer
        self.now_yield_i = 0 # the idex of data has yielded
        self.buffer_size = buffer_size
        self.buffer = Queue()
        self.is_loading_data = False
        self.__get_data_by_ids__ = get_data_by_ids_func
        self.__fill_buffer__()

    def set_batch_size(self, batch_size:Union[int, List[int]]):
        if type(batch_size) != list and batch_size == 0:
            raise
        batch_size = [batch_size] if type(batch_size) != list else batch_size
        self.batch_size = np.array([min(bs, self.sample_count) for bs in batch_size])

    def __get_data_by_ids__(self, ids):
        raise

    def __fill_buffer__(self):
        if self.is_loading_data:
            return
        self.is_loading_data = True 
        def fill_buffer(): 
            while self.buffer.qsize() < self.buffer_size:
                bs = self.rng.choice(self.batch_size)
                tail_i = self.now_buffer_i + bs
                ids = self.select_ids[self.now_buffer_i:tail_i]
                if tail_i >= self.sample_count:
                    self.select_ids = np.array(range(self.sample_count))
                    if self.shuffle:
                        self.rng.shuffle(self.select_ids)
                    if tail_i > self.sample_count and self.drop_last:
                        self.now_buffer_i = 0
                        continue
                    self.now_buffer_i = tail_i - self.sample_count
                    extra_ids = self.select_ids[:self.now_buffer_i]
                    ids = np.concatenate([ids, extra_ids], 0)
                else:
                    self.now_buffer_i = tail_i
                d = self.__get_data_by_ids__(ids)
                self.buffer.put((d, len(ids)))
            self.is_loading_data = False  
        threading.Thread(target = fill_buffer).start() 
    
    def __len__(self): 
        if len(self.batch_size) > 1:
            print('The number of data batches is not accurate as `batch_size` is a list')
        bs = self.batch_size.mean()
        if self.drop_last:
            return int(np.floor(self.sample_count/bs))
        return int(np.ceil(self.sample_count/bs))

    def __iter__(self): 
        self.now_yield_i = 0
        return self

    def __next__(self):
        if self.now_yield_i >= self.sample_count:
            raise StopIteration
        if self.buffer.qsize() <= self.buffer_size/2:
            self.__fill_buffer__() 
        t = 0  
        while self.buffer.qsize() == 0:  
            print('\r', "Waiting data: %d s"%t, end='')
            time.sleep(1) 
            t += 1  
        d, data_n = self.buffer.get()
        self.now_yield_i += data_n
        return d



################################################################################
#    prompts & targets transform to input&output&mask token ids                # 
################################################################################
def prompts_target_to_x_y_mask(tokenizer, prompts:List[str], targets:List[str], device='cuda'):
    '''
    Generate inputs (x) and outputs (y) for training a self-regressive model, 
    along with training masks. Assume that prompts correspond one-to-one with targets.
    return: `input_ids`, `label_ids`, `masks`
    input_ids/label_ids/masks's type, dtype, shape are:  
        torch.Tensor, Long, [batch_size, max_length_of_prompts_and_targets]
    '''
    targets = deepcopy(targets)
    for i, t in enumerate(targets):
        targets[i] = t if t[0] == ' ' else ' ' + t
    input_ids, label_ids, masks = [], [], []
    for p, t in zip(prompts, targets):
        prompt_tok = tokenizer(p)['input_ids']
        input_tok = tokenizer(p + t, return_tensors="pt")['input_ids'][0]
        label_tok = input_tok.clone()[1:] 
        input_tok = input_tok[:-1] 
        mask = torch.ones_like(label_tok)
        mask[:len(prompt_tok)-1] *= 0
        input_ids.append(input_tok)
        label_ids.append(label_tok)
        masks.append(mask)
    input_ids = pad_sequence(input_ids, True, tokenizer.pad_token_id).to(device)
    label_ids = pad_sequence(label_ids, True, tokenizer.pad_token_id).to(device)
    masks = pad_sequence(masks, True, 0).to(device)
    return input_ids, label_ids, masks


################################################################################
#    prompts & predict length to get input&output&mask token ids               #  
################################################################################
def prompts_last_len_to_x_y_mask(tokenizer, prompts:List[str], pre_len:Union[int, float], 
    truncation = 1024, device='cuda'):
    '''
    Use the last pre_len number or proportion of tokens from the prompt for 
    predicting the output. Generate inputs (x) and outputs (y) for training a 
    self-regressive model, along with training masks. 
    `truncation`: to prevent an excessive number of tokens, omitting any surplus tokens.
    input_ids/label_ids/masks's type, dtype, shape are: 
        torch.Tensor, Long, [batch_size, max_length_of_prompts]
    '''
    input_ids, label_ids, masks = [], [], []
    for p in prompts:
        input_tok = tokenizer(p, return_tensors="pt")['input_ids'][0][:truncation]
        label_tok = input_tok.clone()[1:] 
        input_tok = input_tok[:-1] 
        mask = torch.zeros_like(label_tok)
        if type(pre_len) == int:
            mask[-pre_len:] += 1
        elif type(pre_len) == float and pre_len <= 1.:
            pl = int(len(mask) * pre_len)
            mask[-pl:] += 1
        else:
            raise
        input_ids.append(input_tok)
        label_ids.append(label_tok)
        masks.append(mask)
    input_ids = pad_sequence(input_ids, True, tokenizer.pad_token_id).to(device)
    label_ids = pad_sequence(label_ids, True, tokenizer.pad_token_id).to(device)
    masks = pad_sequence(masks, True, 0).to(device)
    return input_ids, label_ids, masks



################################################################################
#              Initialize for training datasets                                #
################################################################################
class TrainDataInit:
    '''
    Functions that preprocess the training datasets and output the `get_data_by_ids_func` 
    for `ParallelDataset` class to generating data.
    '''
    # ZSRE
    def zsre(data_path, tokenizer:AutoTokenizer, device='cuda'):
        assert os.path.isfile(data_path)
        set_tokenizer_pad_id(tokenizer)
        with open(data_path, 'r') as f: 
            data = json.load(f)
            sample_count = len(data)
            prompts = np.array([i['src'] for i in data])
            rep_prompts = np.array([i['rephrase'] for i in data])
            target_new = np.array([i['alt'] for i in data])
            loc_prompts = np.array([i['loc'] for i in data])
            loc_ans = np.array([i['loc_ans'] for i in data])
        def get_data_by_ids(ids:List[int]):
            edit_xym = prompts_target_to_x_y_mask(tokenizer, prompts[ids], target_new[ids], device)
            rep_xym = prompts_target_to_x_y_mask(tokenizer, rep_prompts[ids], target_new[ids], device)
            loc_xym = prompts_target_to_x_y_mask(tokenizer, loc_prompts[ids], loc_ans[ids], device)
            return edit_xym, {'rephrase': rep_xym}, {'normal': (loc_xym[0], loc_xym[2])}
        return sample_count, get_data_by_ids 
    
    # ZSRE + extra
    def zsre_extra(zsre_path = 'data/meta-train/zsre/zsre_mend_train.json', 
                   extra_path = 'data/meta-train/zsre/long_tail_all_two_types_dataset.json', 
                   zsre_n = 30000, extra_n = 30000, tokenizer:AutoTokenizer = None, device='cuda'):
        assert os.path.isfile(zsre_path)
        assert os.path.isfile(extra_path)
        set_tokenizer_pad_id(tokenizer)
        def load_data(data, n):
            sample_count = min(len(data), n)
            prompts = np.array([d['src'] for i, d in zip(range(sample_count), data)])
            rep_prompts = np.array([d['rephrase'] for i, d in zip(range(sample_count), data)])
            target_new = np.array([d['alt'] for i, d in zip(range(sample_count), data)])
            loc_prompts = np.array([d['loc'] for i, d in zip(range(sample_count), data)])
            loc_ans = np.array([d['loc_ans'] for i, d in zip(range(sample_count), data)])
            return sample_count, prompts, rep_prompts, target_new, loc_prompts, loc_ans
        with open(zsre_path, 'r') as f: 
            data = json.load(f)
            sample_count, prompts, rep_prompts, target_new, loc_prompts, loc_ans = load_data(data, zsre_n) 
        with open(extra_path, 'r') as f: 
            data = json.load(f)
            sample_count_1, prompts_1, rep_prompts_1, target_new_1, loc_prompts_1, loc_ans_1 = load_data(data, extra_n)
        sample_count += sample_count_1
        prompts = np.concatenate([prompts, prompts_1], 0)
        rep_prompts = np.concatenate([rep_prompts, rep_prompts_1], 0)
        target_new = np.concatenate([target_new, target_new_1], 0)
        loc_prompts = np.concatenate([loc_prompts, loc_prompts_1], 0)
        loc_ans = np.concatenate([loc_ans, loc_ans_1], 0)
        def get_data_by_ids(ids:List[int]):
            edit_xym = prompts_target_to_x_y_mask(tokenizer, prompts[ids], target_new[ids], device)
            rep_xym = prompts_target_to_x_y_mask(tokenizer, rep_prompts[ids], target_new[ids], device)
            loc_xym = prompts_target_to_x_y_mask(tokenizer, loc_prompts[ids], loc_ans[ids], device)
            return edit_xym, {'rephrase': rep_xym}, {'normal': (loc_xym[0], loc_xym[2])}
        return sample_count, get_data_by_ids 
    
    # CF + extra
    def cf_extra(cf_path = 'data/meta-train/cf/counterfact-train.json', 
                 extra_path = 'data/meta-train/zsre/long_tail_all_two_types_dataset.json', 
                 cf_n = 10000, extra_n = 30000, tokenizer:AutoTokenizer = None, device='cuda'):
        assert os.path.isfile(cf_path)
        assert os.path.isfile(extra_path)
        set_tokenizer_pad_id(tokenizer)
        def load_data_cf(data, n):
            sample_count = min(len(data), n)
            prompts = np.array([d['prompt'] for i, d in zip(range(sample_count), data)])
            rep_prompts = np.array([d['rephrase_prompt'] for i, d in zip(range(sample_count), data)])
            target_new = np.array([d['target_new'] for i, d in zip(range(sample_count), data)])
            loc_prompts = np.array([d['locality_prompt'] for i, d in zip(range(sample_count), data)])
            loc_ans = np.array([d['locality_ground_truth'] for i, d in zip(range(sample_count), data)])
            return sample_count, prompts, rep_prompts, target_new, loc_prompts, loc_ans
        def load_data_extra(data, n):
            sample_count = min(len(data), n)
            prompts = np.array([d['src'] for i, d in zip(range(sample_count), data)])
            rep_prompts = np.array([d['rephrase'] for i, d in zip(range(sample_count), data)])
            target_new = np.array([d['alt'] for i, d in zip(range(sample_count), data)])
            loc_prompts = np.array([d['loc'] for i, d in zip(range(sample_count), data)])
            loc_ans = np.array([d['loc_ans'] for i, d in zip(range(sample_count), data)])
            return sample_count, prompts, rep_prompts, target_new, loc_prompts, loc_ans
        with open(cf_path, 'r') as f: 
            data = json.load(f)
            sample_count, prompts, rep_prompts, target_new, loc_prompts, loc_ans = load_data_cf(data, cf_n) 
        with open(extra_path, 'r') as f: 
            data = json.load(f)
            sample_count_1, prompts_1, rep_prompts_1, target_new_1, loc_prompts_1, loc_ans_1 = load_data_extra(data, extra_n)
        sample_count += sample_count_1
        prompts = np.concatenate([prompts, prompts_1], 0)
        rep_prompts = np.concatenate([rep_prompts, rep_prompts_1], 0)
        target_new = np.concatenate([target_new, target_new_1], 0)
        loc_prompts = np.concatenate([loc_prompts, loc_prompts_1], 0)
        loc_ans = np.concatenate([loc_ans, loc_ans_1], 0)
        def get_data_by_ids(ids:List[int]):
            edit_xym = prompts_target_to_x_y_mask(tokenizer, prompts[ids], target_new[ids], device)
            rep_xym = prompts_target_to_x_y_mask(tokenizer, rep_prompts[ids], target_new[ids], device)
            loc_xym = prompts_target_to_x_y_mask(tokenizer, loc_prompts[ids], loc_ans[ids], device)
            return edit_xym, {'rephrase': rep_xym}, {'normal': (loc_xym[0], loc_xym[2])}
        return sample_count, get_data_by_ids 
    
    def meta_train_data(tokenizer:AutoTokenizer, random_seed:int = None, 
            loss_sample_max_count:int = None, device = 'cuda', 
            data_dir = 'data/meta-train/comprehensive', 
            training_sets = ['zsre', 'cf', 'long_tail', 'popular', 'recent', 'robust', 'wiki_base'],
            train_alternate = True):  
        '''
        loss_sample_max_count: limit loss samples for generality and locality below `loss_sample_max_count`.
        '''
        set_tokenizer_pad_id(tokenizer)
        print('Selected training datasets:', training_sets)
        # load wiki data
        if 'wiki_base' in training_sets:
            wiki_path = os.path.join(data_dir, 'wikitext/wikitext-103-raw-v1')
            wiki_data = load_dataset(wiki_path, split='train')
            print('Load:', wiki_path)
            wiki_data =  np.array([t for t in wiki_data['text'] if len(t.split(' ')) > 20 and not \
                re.search(r'^[\s\n]*=', t) and not re.search(r'=[\s\n]*$', t)]) # String List
        #### load training data
        sample_count = 0
        data_funcs = []
        datas = []
        # load cf
        if 'cf' in training_sets:
            cf_path = os.path.join(data_dir, 'cf_train.json')
            with open(cf_path, 'r') as f:
                cf_data = json.load(f)
                print('Load:', cf_path)
            sample_count += len(cf_data)
            def get_cf_data(i, edit_prompts:List, edit_new_targets:List, 
                        gen_prompts:Dict[str, List], gen_targets:Dict[str, List], 
                        loc_prompts:Dict[str, List], loc_targets:Dict[str, List]):
                d = cf_data[i]
                edit_prompts.append(d['prompt'])
                edit_new_targets.append(d['target_new'])
                gen_prompts['cf_rephrase'].append(d['rephrase_prompt'])
                gen_targets['cf_rephrase'].append(d['target_new'])
                if 'cf_distract' in training_sets:
                    loc_prompts['cf_distract'].append(f"{d['prompt']} {d['target_new']}. " + d['locality_prompt'])
                    loc_targets['cf_distract'].append(d['locality_ground_truth'])
                loc_prompts['cf_original'].append(d['locality_prompt'])
                loc_targets['cf_original'].append(d['locality_ground_truth'])
            data_funcs.append(get_cf_data)
            datas.append(cf_data)
        # load zsre
        def get_zsre_like_data_wrap(the_data, data_name):
            def get_data_(i, edit_prompts:List, edit_new_targets:List, 
                        gen_prompts:Dict[str, List], gen_targets:Dict[str, List], 
                        loc_prompts:Dict[str, List], loc_targets:Dict[str, List]):
                d = the_data[i]
                edit_prompts.append(d['prompt'])
                edit_new_targets.append(d['target_new'])
                gen_prompts[data_name+'_rephrase'].append(d['rephrase_prompt'])
                gen_targets[data_name+'_rephrase'].append(d['target_new'])
                loc_prompts[data_name+'_original'].append(d['locality_prompt'])
                loc_targets[data_name+'_original'].append(d['locality_ground_truth'])
            return get_data_
        if 'zsre' in training_sets:
            zsre_path = os.path.join(data_dir, 'zsre_train.json')
            with open(zsre_path, 'r') as f:
                zsre_data = json.load(f)
                print('Load:', zsre_path)
            sample_count += len(zsre_data)
            get_data = get_zsre_like_data_wrap(zsre_data, 'zsre')
            data_funcs.append(get_data)
            datas.append(zsre_data)
        #load long_tail/popular/recent
        extra_fnames = ['long_tail', 'popular', 'recent']
        for extra_data_name in extra_fnames:
            if extra_data_name in training_sets:
                path = os.path.join(data_dir, extra_data_name+'.json')
                with open(path, 'r') as f:
                    extra_data = json.load(f)
                    print('Load:', path)
                sample_count += len(extra_data)
                zsre_data.extend(extra_data)
        # load robust data
        if 'robust' in training_sets:
            robust_path = os.path.join(data_dir, 'robust.json')
            with open(robust_path, 'r') as f:
                robust_data = json.load(f)
                print('Load:', robust_path)
            sample_count += len(robust_data)
            def get_robust_data(i, edit_prompts:List, edit_new_targets:List, 
                        gen_prompts:Dict[str, List], gen_targets:Dict[str, List], 
                        loc_prompts:Dict[str, List], loc_targets:Dict[str, List]):
                def sample_prefix(prefixs:List[str]):
                    prefix = prefixs[rng.choice(len(prefixs), 1)[0]]
                    prefix = ' '.join(prefix.split(' ')[:30])
                    prefix = prefix + '. ' if prefix[-1] != '.' else prefix + ' '
                    return prefix
                d = robust_data[i]
                edit_prompts.append(d['prompt'])
                edit_new_targets.append(d['target_new'])
                prefix = sample_prefix(d['prefix'])
                gen_prompts['robust'].append(prefix + d['rephrase_prompt'])
                gen_targets['robust'].append(d['target_new'])
                prefix = sample_prefix(d['prefix'])
                loc_prompts['robust'].append(prefix + d['locality_prompt'])
                loc_targets['robust'].append(d['locality_ground_truth'])
            data_funcs.append(get_robust_data)
            datas.append(robust_data)
        # main
        rng = np.random.default_rng(random_seed)
        loss_sample_max_count = 99999 if loss_sample_max_count == None else loss_sample_max_count
        def get_data_by_ids(ids):
            edit_prompts = []
            edit_new_targets = []
            gen_prompts = defaultdict(list) #{'rephrase': []}
            gen_targets = defaultdict(list) #{'rephrase': []}
            loc_prompts = defaultdict(list) #{'original': []}
            loc_targets = defaultdict(list) #{'original': []}
            if train_alternate:
                data_i = rng.choice(len(data_funcs), 1)[0]
            for i in ids:
                if train_alternate:
                    for d in datas[:data_i]:
                        i = i - len(d)
                    i = i % len(datas[data_i])
                    data_funcs[data_i](i, edit_prompts, edit_new_targets, gen_prompts, gen_targets, loc_prompts, loc_targets)
                else:
                    has_data = False
                    for d, df in zip(datas, data_funcs):
                        if i < len(d):
                            df(i, edit_prompts, edit_new_targets, gen_prompts, gen_targets, loc_prompts, loc_targets)
                            has_data = True
                            break
                        i = i - len(d)
                    if not has_data:
                        raise
            # random select generality and locality samples 
            def select_loss_samples(prompts:list, targets:list):
                lsmc = min(len(prompts), loss_sample_max_count)
                idx = rng.choice(len(prompts), lsmc, replace = False)
                return [prompts[i] for i in idx], [targets[i] for i in idx]
            for k in gen_prompts.keys():
                gen_prompts[k], gen_targets[k] = select_loss_samples(gen_prompts[k], gen_targets[k])
            for k in loc_prompts.keys():
                loc_prompts[k], loc_targets[k] = select_loss_samples(loc_prompts[k], loc_targets[k])
            # transform data
            edit_xym = prompts_target_to_x_y_mask(tokenizer, edit_prompts, edit_new_targets, device)
            gen_xym = {}
            for k in gen_prompts.keys():
                gen_xym[k] = prompts_target_to_x_y_mask(tokenizer, gen_prompts[k], gen_targets[k], device)
            loc_xm = {}
            for k in loc_prompts.keys(): 
                xym = prompts_target_to_x_y_mask(tokenizer, loc_prompts[k], loc_targets[k], device)
                loc_xm[k] = (xym[0], xym[2])
            # add wiki locality samples
            if 'wiki_base' in training_sets:
                idx = rng.choice(len(wiki_data), 1)[0]
                xym = prompts_last_len_to_x_y_mask(tokenizer, [wiki_data[idx]], 0.9, 1024, device)
                loc_xm['wiki_base'] = (xym[0], xym[2])
            return edit_xym, gen_xym, loc_xm#, \
                    #   edit_prompts, edit_new_targets, \
                    #   gen_prompts, gen_targets,loc_prompts,loc_targets
        return sample_count, get_data_by_ids
        

    # WIKI
    def wiki(data_path, tokenizer:AutoTokenizer, data_type = 'train', 
        pre_len:Union[int, float] = 0.3, truncation = 1024, device='cuda'):
        '''
        `if_train`: Whether to use the training set or the test set.
        `pre_len`: An integer or a float. If it is an integer, take the last 
            fixed number of tokens for prediction. If it is a float and less than or 
            equal to 1, take the last proportion of tokens for prediction.
        '''
        assert os.path.isdir(data_path) 
        set_tokenizer_pad_id(tokenizer)
        ds = load_dataset(data_path)[data_type]
        ds =  np.array([t for t in ds['text'] if len(t.split(' ')) > 20 and not \
            re.search(r'^[\s\n]*=', t) and not re.search(r'=[\s\n]*$', t)]) # String list
        def get_data_by_ids(ids):
            input_ids, label_ids, masks = prompts_last_len_to_x_y_mask(tokenizer, 
                ds[ids], pre_len, truncation, device)
            return input_ids, label_ids, masks
        sample_count = len(ds)
        return sample_count, get_data_by_ids


################################################################################
#                     get structured test datasets                             #
################################################################################
class TestSampleList:
    '''
    Functions used to read and preprocess various datasets for evaluation,
    which return list with structure like [
        { # test1
            'request': {'prompt': str, 'target_new': str, ...},
            'generality': {
                'gen_1_name':[
                    {'prompt': str, 'target': str, ...},
                    {'prompt': str, 'target': str, ...}, ...
                ],
                'gen_2_name':[
                    {'prompt': str, 'target': str, ...},
                    {'prompt': str, 'target': str, ...}, ...
                ], ...
            },
            'locality': {
                'loc_1_name':[
                    {'prompt': str, 'target': str, ...},
                    {'prompt': str, 'target': str, ...}, ...
                ],
                'loc_2_name':[
                    {'prompt': str, 'target': str, ...},
                    {'prompt': str, 'target': str, ...}, ...
                ], ...
            }
        }, 
        { # test2
            'request':{'prompt': str, 'target_new': str, ...},
            'generality': ...
        }, ...
    ]. 
    '''
    def load_and_select_data(path:str, test_i:Union[List, int], shuffle:bool, seed:int):
        with open(path, 'r') as f:
            data = json.load(f)
        idx = list(range(len(data)))
        if shuffle:
            rng = np.random.default_rng(seed)
            rng.shuffle(idx)
        if test_i == None:
            test_i = idx
        elif type(test_i) == int:
            test_i = idx[:test_i]
        elif type(test_i) == list:
            test_i = [idx[i] for i in test_i]
        else:
            raise
        return [data[i] for i in test_i]
 
    def zsre(path = 'data/evaluation/zsre/zsre_mend_eval.json', 
            test_i:Union[List, int] = None, shuffle = True, seed = 0):
        data = TestSampleList.load_and_select_data(path, test_i, shuffle, seed)
        test_sample_list = []
        for s in data:
            ns = {}
            ns['request'] = {
                'prompt': s['src'], 
                'target_new': s['alt'], 
                'subject': s['subject'],
                'ground_truth': s['answers'][0],
            }
            ns['generality'] = {
                'rephrase': [
                    {'prompt': s['rephrase'], 'target': s['alt']},
                ]
            }
            ns['locality'] = {
                'loc1': [
                    {'prompt': s['loc'], 'target': s['loc_ans']},
                ]
            }
            test_sample_list.append(ns)
        return test_sample_list

    def counterfact(path = 'data/evaluation/cf/counterfact-edit.json', 
            test_i:Union[List, int] = None, shuffle = True, seed = 0):
        # counterfact dataset path
        data = TestSampleList.load_and_select_data(path, test_i, shuffle, seed)
        test_sample_list = []
        for s in data:
            ns = {}
            ns['request'] = {
                'prompt': s['prompt'], 
                'target_new': s['target_new'], 
                'subject': s['subject'],
                'ground_truth': s['ground_truth'],
            }
            ns['generality'] = {
                'rephrase': [
                    {'prompt': s['rephrase_prompt'], 'target': s['target_new']},
                ]
            }
            ns['locality'] = {
                'loc1': [
                    {'prompt': s['locality_prompt'], 'target': s['locality_ground_truth']},
                ]
            }
            test_sample_list.append(ns)
        return test_sample_list
    
    def counterfact_plus(path, test_i:Union[List, int] = None, shuffle = True, seed = 0):
        # counterfact dataset path
        data = TestSampleList.load_and_select_data(path, test_i, shuffle, seed)
        test_sample_list = []
        for s in data:
            ns = {}
            ns['request'] = {
                'prompt': s['prompt'], 
                'target_new': s['target_new'], 
                'subject': s['subject'],
                'ground_truth': s['ground_truth'], 
            }
            ns['generality'] = {
                'rephrase': [
                    {'prompt': s['rephrase_prompt'], 'target': s['target_new']},
                ]
            }
            ns['locality'] = {
                'loc1': [
                    {
                        'prompt': f"{s['prompt']} {s['target_new']}. " + s['locality_prompt'], 
                        'target': s['locality_ground_truth']
                    },
                ]
            }
            test_sample_list.append(ns)
        return test_sample_list

    def ripple_effect(path = 'data/evaluation/ripple_effect/ripple_effect.json', 
                      test_i:Union[List, int] = None, shuffle = True, seed = 0):
        # ripple effect dataset path
        data = TestSampleList.load_and_select_data(path, test_i, shuffle, seed)
        test_sample_list = []
        for s in data:
            ns = {}
            ns['example_type'] = s['example_type']
            ns['request'] = {
                'prompt': s['prompt'], 
                'target_new': s['target_new'], 
                'subject': s['subject'],
            }
            if s['example_type'] == 'recent':
                ns['request']['ground_truth'] = s['target_new']
            else:
                ns['request']['ground_truth'] = s['ground_truth']
            gen_types = ['Logical_Generalization', 'Compositionality_I', 
                            'Compositionality_II', 'Subject_Aliasing']
            ns['generality'] = {}
            for gen_type in gen_types:
                ns['generality'][gen_type] = []
                for i in s[gen_type]:
                    for t in i['targets']:
                        if t != "":
                            ns['generality'][gen_type].append({'prompt': i['prompt'], 'target':t})
                            break
            loc_types = ['Relation_Specificity', 'Forgetfulness']
            ns['locality'] = {}
            for loc_type in loc_types:
                ns['locality'][loc_type] = []
                for i in s[loc_type]:
                    for t in i['targets']:
                        if t != "":
                            ns['locality'][loc_type].append({'prompt': i['prompt'], 'target':t})
                            break
            test_sample_list.append(ns)
        return test_sample_list


    def test_data(path, test_n = None):
        with open(path, 'r') as f:
            data = json.load(f)
        return data










 