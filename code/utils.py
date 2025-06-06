import os, sys
import yaml 
import torch
import torch.nn as nn
import numpy as np
import random

import time
def printDebugInfo(*args):
    print("[DEBUG] [%.3f]"%(time.time()), *args)
    pass

class TeeOutput:
    """
        Why we need this?
        Because we want to save the output of the print 
        automatically to a file
    """
    def __init__(self, filename, mode="a"):
        self.file = open(filename, mode)
        self.stdout = sys.stdout 

    def write(self, message):
        self.stdout.write(message)
        self.file.write(message)

    def flush(self):
        self.stdout.flush()
        self.file.flush()
    
def print_args(args):
    print( "-"*20 + f"{args.prefix}" + "-"*20)
    print("-"*20 + "Arguments" + "-"*20)
    print(yaml.dump(vars(args)))
    print("-"*20 + "End of Arguments" + "-"*20)

def set_save_path(args, extra="", sub_dir=""):
    folder_name = ""
    save_path = args.save_path
    if hasattr(args, 'prefix') and \
        args.prefix !="":
        folder_name = args.prefix
    if extra != "":
        folder_name = folder_name + "_" + extra
    
    folder_name = folder_name + "_" + \
            args.fusion_method + "_" + \
            args.dataset + "_" + \
            str(args.random_seed)
    
    save_path = os.path.join(save_path, folder_name)
    if sub_dir != "":
        save_path = os.path.join(save_path, sub_dir)

    if os.path.exists(save_path):
        print("Clearing directory: {}".format(save_path))
        os.system("rm -rf {}".format(save_path))
    
    print("Creating directory: {}".format(save_path))
    os.makedirs(save_path)
    return save_path

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
