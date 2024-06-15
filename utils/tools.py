import os
import time
import yaml
import random
import numpy as np
from argparse import Namespace

import torch
import torch.distributed as dist

from PIL import Image



class Logger():
    def __init__(self, path: str):
        """
        Args:
            path (str):     the path to save log file
        """
        self.log_path = path
        self.log_dir = os.path.dirname(path)

        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
    

    def info(self, content: str):
        """
        Record information and write to log file
        Args:
            content (str):  the content to be recorded
        """
        time_stamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        content = '[{0}] {1}\n'.format(time_stamp, content)

        with open(self.log_path, 'a') as f:
            f.write(content)


    def create_config(self, args: Namespace):
        """
        Create config file and save to disk in the form of yaml
        Args:
            opt (Namespace):    options/configs to be wrote
        """
        path = os.path.join(self.log_dir, 'configs.yaml')
        
        with open(path, 'w', encoding='utf-8') as f:
            yaml.dump(vars(args), f, allow_unicode=True)



def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True



worker_logger = None

def set_logger(logger):
    global worker_logger
    worker_logger = logger

def get_logger():
    global worker_logger
    return worker_logger



def reduce_tensor(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= dist.get_world_size()
    return rt

def gather_tensor(tensor):
    gt = [torch.zeros_like(tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(gt, tensor)
    return gt

def broadcast_tensor(tensor, src):
    t = tensor.clone()
    dist.broadcast(t, src)
    return t



def pil_loader(path: str) -> Image.Image:
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")
