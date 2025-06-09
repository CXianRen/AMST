# MSLR: Modality-specific Learning Rates for Effective Multimodal Additive Late-fusion
# why we use this method is because it support the text modality 
# well due to text modality learning rate is 
# different from the image modality learning 
# rate in a degree of magnitude
# This code is implemented by ourself, based on the paper.
# More details can see the comments in the code. We put the thing we 
# think is important in the comments with Notice! in the code.

import numpy as np
from models.basic_model import  M_TEXT_NAME, M_AUDIO_NAME, \
    M_VISUAL_NAME, KEY_HELPERS, KEY_ENCODERS,\
    KEY_FUSION, KEY_TEXT_TOKENS, KEY_TEXT_PADDING_MASK

from common import BasicTrainer, MAIN_DEVICE_KEY, update_arg

from torch import optim

class cycle_buffer:
    def __init__(self, max_len):
        self.max_len = max_len
        self.buffer = []
    
    def append(self, item):
        if len(self.buffer) >= self.max_len:
            self.buffer.pop(0)
        self.buffer.append(item)
    
    def get_avg(self):
        if len(self.buffer) == 0:
            return 0.0
        return np.mean(self.buffer)

class MSLRTrainer(BasicTrainer):
    def __init__(self, args_str=None):
        super(MSLRTrainer, self).__init__(args_str)
        
    def init_parser(self):
        """
        Override
        """
        parser = super().init_parser()
        update_arg(parser, '--prefix', default='MSLR', type=str,
                            help='prefix for the save path')
        
        parser.add_argument('--strategy', default='dynamic', type=str,
            choices=['dynamic', 'keep'],
            help='dynamic: change learning rate during training,\
            static: fixed learning rate')

        parser.add_argument('--history_length', default=5, type=int,
            help='history length of the acc on validation set\
                used for dynamic strategy')

        # each model part has its own inital learning rate based on the paper
        # and with our experiment, we find that the learning rate 
        # for each model part. And for fusion_lr, we keep it as 
        # what we used in other methods, which is 0.001. The learning 
        # rate of each eoncder will be dynamically updated based on the
        # validation set acc.
        parser.add_argument('--fusion_lr', default=0.001, type=float,)
        parser.add_argument('--audio_lr', default=0.001, type=float)
        parser.add_argument('--visual_lr', default=0.01, type=float)
        parser.add_argument('--text_lr', default=2e-5, type=float)
        
        return parser

    def init_optimizer_scheduler(self):
        """
        Override
        """
        # Notice!
        # Each model part has its own learning rate based on the paper
        lr_dict = {
            M_TEXT_NAME: self.args.text_lr,
            M_AUDIO_NAME: self.args.audio_lr,
            M_VISUAL_NAME: self.args.visual_lr
        }
        
        optimizer_map = {
            KEY_FUSION: optim.SGD(
                self.model[KEY_FUSION].parameters(),
                lr=self.args.fusion_lr, momentum=0.9, weight_decay=1e-4
            )
        }
        
        optimizer_map[KEY_HELPERS] = optim.SGD(
            self.model[KEY_HELPERS].parameters(),
            lr=self.args.fusion_lr, momentum=0.9, weight_decay=1e-4
        )
        
        modality_name_list = list(self.model[KEY_ENCODERS].keys())
        params = []
        self.params_name_id_map = {} # used for update the learning rate
        for name in modality_name_list:
            params.append({
                    'params': self.model[KEY_ENCODERS][name].parameters(),
                    'lr': lr_dict[name],
            })
            self.params_name_id_map[name] = len(params) - 1

        optimizer_map[KEY_ENCODERS] = optim.SGD(
            params, momentum=0.9, weight_decay=1e-4
        )
        
        # assign an individual scheduler for each model part
        scheduler_map = {
            key: optim.lr_scheduler.StepLR(
                optimizer_map[key], self.args.lr_decay_step, self.args.lr_decay_ratio
            ) for key in optimizer_map.keys()
        }
        
        # history of the acc on validation set
        self.val_acc_history_map = {
            name : 0.0 for name in modality_name_list
        }
        
        self.val_acc_history_map = {
            name : cycle_buffer(self.args.history_length) \
                for name in modality_name_list
        }
        
        self.optimizer_map = optimizer_map
        self.scheduler_map = scheduler_map
    
    def adjust_lr(self):
        # Notice!
        # This is the core part of the method.
        # update the history of the acc on validation set
        
        # self.val_m_map updated in print_metric before calling after_valid
        val_m = self.val_m_map
        
        for name in self.modality_name_list:
            self.val_acc_history_map[name].append(
                val_m[name].get_acc()
            )
        
        # update the lr for every N epochs, N is the history length
        if self.args.strategy == 'dynamic' and \
            self.epoch % self.args.history_length == 0 and\
            self.epoch > 0:
                
            for name in self.modality_name_list:
                avg_acc = self.val_acc_history_map[name].get_avg()
                ratio = val_m[name].get_acc() / avg_acc
                group_id = self.params_name_id_map[name]
                self.optimizer_map[KEY_ENCODERS].param_groups[group_id]['lr'] *= ratio
                
                print("Update learning rate for {}: {:.6f}".format(
                    name, self.optimizer_map[KEY_ENCODERS].param_groups[group_id]['lr']))
                self.tsb_writer.add_scalar(
                    "lr/{}".format(name),
                    self.optimizer_map[KEY_ENCODERS].param_groups[group_id]['lr'],
                    self.epoch
                )
        # update end       
    
    def after_validate(self):
        """
        Override.
        This function is called after the validation step.
        It is used to update the learning rate for each model part.
        """
        self.adjust_lr()


if __name__ == "__main__":
    trainer = MSLRTrainer()
    trainer.train_validate()
