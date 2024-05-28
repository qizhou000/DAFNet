def get_attr():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, help = 'Name of model to be edited.', required=True)
    parser.add_argument('--device', type=int, help = 'Device for load model to be edited.', required=True)
    parser.add_argument('--extra_device', type=int, nargs='+', help='Extra CUDA devices for training dafnet.', required=True)
    parser.add_argument('--datasets', type=str, help = 'Datasets used for training.', 
        default = 'zsre-cf-long_tail-popular-recent-robust-wiki_base')
    parser.add_argument('--dafnet_ckpt_path', type=str, help = 'Checkpoint path of DAFNet.', default=None)
    parser.add_argument('--train_name', type=str, help = 'Name of this training.', default=None)
    args = parser.parse_args()
    args.datasets = args.datasets.split('-')
    return args
 
cfg = get_attr()
# get DAFNet
from utils.data import TrainDataInit
from utils.utils import get_editor
dafnet = get_editor('dafnet', cfg.model_name, cfg.device, cfg.extra_device, None)
sample_count, get_data_by_ids = TrainDataInit.meta_train_data(dafnet.tokenizer, None, 
    dafnet.cfg.training.loss_sample_max_count, cfg.device, training_sets = cfg.datasets) 
dafnet.train_init(sample_count, get_data_by_ids, 'train_records', None, cfg.train_name, 
    cfg.dafnet_ckpt_path, 1000, 1, None, False, cfg.extra_device[1:3], cfg.extra_device[3:])
# Train
dafnet.train(epochs = 100)
 

 