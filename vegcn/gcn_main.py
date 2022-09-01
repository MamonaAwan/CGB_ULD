
import os
import torch

from mmcv import Config

from utils_gcn import (set_random_seed, rm_suffix, mkdir_if_no_exists) 
from vegcn.models import build_model
from vegcn import build_handler


class args_fixed():
    def __init__(self,):
        self.config = 'vegcn/configs/cfg_test_gcnv.py'
        self.seed = 42
        self.phase = 'test'
        self.work_dir= 'gcn_outputs'
        self.load_from = 'data/pretrained_models/pretrained_gcn_v_ms1m.pth'
                        
        self.resume_from = None
        self.gpus = 1
        
        self.random_conns = False
        self.distributed = False
        self.eval_interim = True
        self.save_output = True
        self.no_cuda = False 
        self.force = True


def gcn_main(folder, logger):
    args = args_fixed()
    cfg = Config.fromfile(args.config)

    # set cuda
    cfg.cuda = not args.no_cuda and torch.cuda.is_available()

    # set cudnn_benchmark & cudnn_deterministic
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    if cfg.get('cudnn_deterministic', False):
        torch.backends.cudnn.deterministic = True

    # update configs according to args
    args.work_dir = os.path.join(folder, args.work_dir)
    if not hasattr(cfg, 'work_dir'):
        if args.work_dir is not None:
            cfg.work_dir = args.work_dir
        else:
            cfg_name = rm_suffix(os.path.basename(args.config))
            cfg.work_dir = os.path.join('./data/work_dir', cfg_name)
    mkdir_if_no_exists(cfg.work_dir, is_folder=True)

    cfg.load_from = args.load_from
    cfg.resume_from = args.resume_from

    cfg.gpus = args.gpus
    cfg.distributed = args.distributed

    cfg.random_conns = args.random_conns
    cfg.eval_interim = args.eval_interim
    cfg.save_output = args.save_output
    cfg.force = args.force


    for data in ['train_data', 'test_data']:
        if not hasattr(cfg, data):
            continue
        cfg[data].eval_interim = cfg.eval_interim
        if not hasattr(cfg[data], 'knn_graph_path') or not os.path.isfile(
                cfg[data].knn_graph_path):
            cfg[data].prefix = cfg.prefix
            cfg[data].knn = cfg.knn
            cfg[data].knn_method = cfg.knn_method
            name = 'train_name' if data == 'train_data' else 'test_name'
            cfg[data].name = cfg[name]
    
    # set random seeds
    if args.seed is not None:
        logger.info('Set random seed to {}'.format(args.seed))
        set_random_seed(args.seed)

    model = build_model(cfg.model['type'], **cfg.model['kwargs'])
    handler = build_handler(args.phase, cfg.model['type'])

    predictions_dir = handler(model, cfg, logger)
    
    return predictions_dir



if __name__ == '__main__':
    gcn_main()
    
