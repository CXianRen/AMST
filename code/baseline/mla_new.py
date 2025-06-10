# This method is based on the original MLA method 
# But we fix the bugs (at least we think they are bugs)
# 1. The gradient modulation is actually not worked at all.
#    It cause memory leak, we have submitted the 
#    issue and fixing method to the original repository
#    ref: (the author closed all the issues of this repo without
#    any reason))
#    So in this version, we just use the alternating training and
#    the run time fusion
# 2. The original method computing the entropy is wrong
#    More details can be found in this code wth comment 
#    "Notice!"
#    ref: https://github.com/CXianRen/MLA/blob/ab6c184579f9ff390e9c9b8216c9900c2a43545b/main.py#L66
# 3. The original implementation doesn't use a validation set
#    ref: https://github.com/CXianRen/MLA/blob/ab6c184579f9ff390e9c9b8216c9900c2a43545b/main.py#L898

import torch
import torch.nn.functional as F

from dataset.dataset import DATASET_LIST, \
    TVA_SET_LIST, AV_SET_LIST, TV_SET_LIST, \
    M2DATASET_LIST, M3DATASET_LIST
    
from models.basic_model import  M_TEXT_NAME, M_AUDIO_NAME,\
    M_VISUAL_NAME, KEY_HELPERS, KEY_ENCODERS, \
    KEY_FUSION, KEY_TEXT_TOKENS, KEY_TEXT_PADDING_MASK, \
    forward_fusion
    
from models.fusion_modules import Fusion_List

from common import MAIN_DEVICE_KEY, \
    print_loss_and_acc, BasicTrainer, update_arg

from metrics import performanceMetric
from prettytable import PrettyTable

# helper functions
def calculate_entropy(output):
    # Notice!
    # For a batch, it should use dim = 1, but
    # in the original code, it uses dim = 0
    probabilities = F.softmax(output, dim=1)
    log_probabilities = torch.log(probabilities)
    entropy = -torch.sum(probabilities * log_probabilities, dim=1)
    return entropy

def calculate_gating_weights(logit_map:dict):
    # Notice!
    # We refactor this part, but keep it as described in the paper
    name_list = list(logit_map.keys())
    
    e_map = {}
    w_map = {}
    
    for name in name_list:
        e_map[name] = calculate_entropy(logit_map[name]).unsqueeze(1)

    combined_entropy = torch.cat([e_map[n] for n in name_list], dim=1)
    max_entropy = torch.max(combined_entropy, dim=1)
    max_entropy = max_entropy.values.unsqueeze(1)
    
    for name in name_list:
        w_map[name] = torch.exp(max_entropy - e_map[name])
    
    sum_weights = sum([w_map[n] for n in name_list])
    
    for name in name_list:
        w_map[name] = w_map[name] / sum_weights
    
    return w_map, e_map

class MLATrainer(BasicTrainer):
    def __init__(self, args_str=None):
        super(MLATrainer, self).__init__(args_str)
        
    def init_parser(self):
        parser = super().init_parser()
        update_arg(parser, '--prefix', default='MLA', type=str,
                            help='prefix for the save path')
        
        # We regard the shared head with runtime fusion mechanism
        # as a fusion mothod, called msum, using default setting is ok. 
        update_arg(parser, '--fusion_method', default='msum', type=str,
                            help="now supported: " + str(Fusion_List),
                            choices=["msum"])   

        return parser

    def reinitialize_metrics(self):
        """
        for dynamic fusion, we need extra metrics
        To measure the fusion results with 
        different fusion factors (alpha), like for 2 modality 
        f_55 means 0.5 for each modality
        f_37 means 0.3 for the first modality and 0.7 for the second
        f_111 means 1/3 for each modality
        and also for the entropy based fusion
        """
        super().reinitialize_metrics()

        self.weight_list = []
        if self.args.dataset in AV_SET_LIST:
            self.weight_list = [
                {M_AUDIO_NAME: 0.3,M_VISUAL_NAME: 0.7},
                {M_AUDIO_NAME: 0.4,M_VISUAL_NAME: 0.6},
                # the default alpha of msum fusion
                {M_AUDIO_NAME: 0.5,M_VISUAL_NAME: 0.5},
                {M_AUDIO_NAME: 0.6,M_VISUAL_NAME: 0.4},
                {M_AUDIO_NAME: 0.7,M_VISUAL_NAME: 0.3},
            ]
        elif self.args.dataset in TV_SET_LIST:
            self.weight_list = [
                {M_TEXT_NAME: 0.3,M_VISUAL_NAME: 0.7},
                {M_TEXT_NAME: 0.4,M_VISUAL_NAME: 0.6},
                # the default alpha of msum fusion
                {M_TEXT_NAME: 0.5,M_VISUAL_NAME: 0.5},
                {M_TEXT_NAME: 0.6,M_VISUAL_NAME: 0.4},
                {M_TEXT_NAME: 0.7,M_VISUAL_NAME: 0.3},
            ]
        elif self.args.dataset in TVA_SET_LIST:
            self.weight_list = [
                # MLA default alpha for 3 modality dataset
                {M_TEXT_NAME:0.4, M_VISUAL_NAME:0.25, M_AUDIO_NAME:0.35},
            ]
            
        self.m_f_map ={}
        for weight in self.weight_list:
            name = "_".join([f"{n}{int(weight[n]*100)}" \
                for n in self.modality_name_list])
            self.m_f_map[name] = performanceMetric(
                self.n_classes, name=name)
            
        self.m_f_map["entropy"] = performanceMetric(
            self.n_classes, name="entropy")

    def print_metrics(self, mode = "train"):
        super().print_metrics(mode)
        # print the metrics for prototypical classification
        
        dataloader_map = {
            "train": self.train_dataloader,
            "val": self.val_dataloader,
            "test": self.test_dataloader
        }
        batches = len(dataloader_map[mode])
        
        print("Weighted fusion classification acc:")
        print_loss_and_acc(self.epoch,
                        self.m_f_map, batches,
                        self.tsb_writer, 
                        mode + "_f")
        
        # also save the metrics of each mode
        if mode == "train":
            self.train_m_f_map = self.m_f_map
        elif mode == "val":
            self.val_m_f_map = self.m_f_map
        elif mode == "test":
            self.test_m_f_map = self.m_f_map

    def after_forward_batch(self, embedding_dict, labels_device):
        """
        to collect accuracy of the fusion results
        with different fusion factors (alpha)
        """
        with torch.no_grad():
            # get the output with different fusion factors
            for weight in self.weight_list:
                out_list = []
                m_name =  "_".join([f"{n}{int(weight[n]*100)}" \
                    for n in self.modality_name_list])
                    
                for m in self.modality_name_list:
                    m_out = self.model[KEY_FUSION].get_out_m(m)
                    out_list.append(weight[m] * m_out)
                out = torch.sum(torch.stack(out_list), dim=0)
                pred = self.softmax(out)
                self.m_f_map[m_name].update(
                    pred, labels_device)
            
            # using entropy based fusion    
            w_map, e_map = calculate_gating_weights(embedding_dict)
            out_list = []
            for n in self.modality_name_list:
                m_out = self.model[KEY_FUSION].get_out_m(n)
                out_list.append(w_map[n] * m_out)
            out = torch.sum(torch.stack(out_list), dim=0)
            pred = self.softmax(out)
            self.m_f_map["entropy"].update(pred, labels_device)
 
    def init_best_model_metric(self):
        # for saving the best model
        self.best_acc_info_map = {}
        
    def update_best_model(self):
        for f_name in self.val_m_f_map.keys():
            val_acc = self.val_m_f_map[f_name].get_acc()
            test_acc = self.test_m_f_map[f_name].get_acc()
            
            acc_info = self.best_acc_info_map.get(f_name, {
                "val_acc": val_acc,
                "test_acc": test_acc,
                "epoch": self.epoch
            })
            if val_acc > acc_info["val_acc"]:
                acc_info.update({
                    "val_acc": val_acc,
                    "test_acc": test_acc,
                    "epoch": self.epoch
                })
            self.best_acc_info_map[f_name] = acc_info
        
    def print_best_model(self):
        table = PrettyTable()
        table.field_names = ["F method", "Val Acc", "Test Acc", "Epoch"]

        for m_name, acc_info in self.best_acc_info_map.items():
            table.add_row([
            m_name, 
            f"{acc_info['val_acc']:.4f}", 
            f"{acc_info['test_acc']:.4f}", 
            acc_info['epoch']
            ])
        
        print(table)

    def alter_train(self, embedding_dict, labels_device):
        for modality_name in self.modality_name_list:
            # fill mute input 
            fill_embed = None
            # fill_embed will be updated by the fusion layer itself
            # because different fusion methods have different fill_embed
            # although most of the time, it is just a zero tensor
        
            temp_input_dict = dict()
            for name in self.modality_name_list:
                if name == modality_name:
                    temp_input_dict[name] = embedding_dict[name]
                else:
                    temp_input_dict[name] = fill_embed
            # forward again and get the prediction of the current modality
            out_m = forward_fusion(self.model[KEY_FUSION], temp_input_dict)
            
            # backward
            self.optimizer_map[KEY_ENCODERS].zero_grad()
            self.optimizer_map[KEY_FUSION].zero_grad()
            
            loss = self.criterion(out_m, labels_device) 
            loss.backward()
            
            self.optimizer_map[KEY_FUSION].step()
            self.optimizer_map[KEY_ENCODERS].step()   
            

    def train_method(self, embedding_dict, labels_device):
        self.alter_train(
            embedding_dict, labels_device)
  
if __name__ == "__main__":
    trainer = MLATrainer()
    trainer.train_validate()