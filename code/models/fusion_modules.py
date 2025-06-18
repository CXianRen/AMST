# This file is based on the original MLA code,
# link: https://github.com/CXianRen/MLA/blob/main/models/fusion_modules.py
# We support fusion methods:
# 1. lsum: Late Sum, from the original MLA, OGM-GE code, this is fusion
# is a bit wired, but for consistency, we keep it.
# 2. msum: MLA Sum, it is the shared head fusion. It is only used in MLA method 
# and MLA related experiments.
# 3. concat: Early Concat, the standard definition of early fusion

import torch
import torch.nn as nn
import torch.nn.functional as F

Fusion_List = ['concat', 'lsum', 'msum']

def gen_fusion_v2(args, input_dim, output_dim, name_list):
    fusion_methods = {
        'lsum': LateSum,
        'msum': MLASum,
        'concat': EarlyConcat
    }
    
    fusion_method = args.fusion_method
    if fusion_method in fusion_methods:
        return fusion_methods[fusion_method](input_dim, output_dim, name_list)
    else:
        raise NotImplementedError("Fusion method not implemented: {}".format(fusion_method))
 
# New fusion api

class MLASum(nn.Module):
    # MLA
    """
        the forward is using equal weights for all modalities by default.
        if we want different weights, we need to call get_out_m function
        for each modality, and then sum the output with the weights.
        
        Structure:
        out = w1*FC(e1) + w2*FC(e2) + w3*FC(e3)
    """
    
    def __init__(self, input_dim, output_dim, modality_name_list):
        super(MLASum, self).__init__()
        # W[e1]+ b + W[e2] +b + W[e3] + b = W[e1 + e2 + e3] + c
        self.fc = nn.Linear(input_dim, output_dim, bias=True)
        self.n_modalities = len(modality_name_list)
        self.out_dict = dict()
    
    def forward(self, embeddings_dict: dict):
        # This might look a bit complicated, but just 
        # for compatibility with our code. 
        # it can be regared as the fusion with same fixed 
        # weights for all modalities.
        # And for the other weights, is computed separately.
        # by using get_out_m function, which returns the
        # the output of the specific modality.
        
        self.out_dict = dict()
        # embeddings_dict: {modality: tensor}
        # sum the embeddings
        for k, v in embeddings_dict.items():
            if v is None:
                # This part should only be used when 
                # using alternating training.
                self.out_dict[k] = None
            else:
                self.out_dict[k] = self.fc(v)
        
        # sum the output
        valid_values = [v for v in self.out_dict.values() if v is not None]
        if not valid_values:
            raise ValueError("All values in out_dict are None.")

        # compute the mean of the valid values
        # this equals to :  out = (out1 + out2 + out3) / n
        out = torch.mean(torch.stack(valid_values), dim=0)
        
        return out
    
    def get_out_m(self, modality_name):
        if modality_name in self.out_dict:
            return self.out_dict[modality_name]
        else:
            raise ValueError("Modality name not found in the output dict")
 
class EarlyConcat(nn.Module):
    """
        This is the standard definition of concatenation.
        Structure:
        out = W[e1, e2, e3] + b = W1[e1] + b/n + W2[e2]+ b/n + W3[e3] + b/n
        where n is the number of modalities.
    """
    def __init__(self, intput_dim, output_dim, modality_name_list):
        super(EarlyConcat, self).__init__()
        self.n_modalities = len(modality_name_list)
        #  W[e1, e2, e3] + b = W1[e1] + b/n + W2[e2]+ b/n + W3[e3] + b/n
        self.bias = nn.Parameter(torch.zeros(output_dim))
        
        self.out_layers = nn.ModuleDict()
        for m in modality_name_list:
            self.out_layers[m] = nn.Linear(intput_dim, output_dim, bias=False)
        
        self.out_dict = dict()
        self.modality_name_list = modality_name_list
        self.modality_name_list.sort()
        
    def forward(self, embeddings_dict: dict):
        self.out_dict = dict()
        # embeddings_dict: {modality: tensor}
        # concatenate the embeddings
        #  W[e1, e2, e3] + b = W1[e1] + b/n + W2[e2]+ b/n + W3[e3] + b/n
        for k, v in embeddings_dict.items():
            if v is None:
                # This part should only be used when
                # using alternating training.
                continue
                
            self.out_dict[k] = self.out_layers[k](v) + self.bias/self.n_modalities
    
        # compute the sum of the valid values
        # this equals to :  out = (out1 + out2 + out3)
        out = torch.sum(torch.stack(list(self.out_dict.values())), dim=0)
        return out
    
    def get_out_m(self, modality_name):
        if modality_name in self.out_dict:
            return self.out_dict[modality_name]
        else:
            raise ValueError("Modality name not found in the output dict")

class LateSum(nn.Module):
    """
    Late Sum fusion module.
    Structure:
    out = W[e1] + b1 + W[e2] + b2 + W[e3] + b3
    where n is the number of modalities.
    
    This is what MLA, OGM-GE code is using.
    This is a bit wired, but for consistency, we keep it.
    """
    # late fusion
    # logit = W1[e1] + b1 + W2[e2] + b2 + W3[e3] + b3
    def __init__(self, input_dim, output_dim, modality_name_list: list):
        super(LateSum, self).__init__()
        self.n_modalities = len(modality_name_list)
        self.out_layers = nn.ModuleDict()
        self.out_dict = dict()
        
        for m in modality_name_list:
            self.out_layers[m] = nn.Linear(input_dim, output_dim)
    
    def forward(self, embeddings_dict: dict):
        self.out_dict = dict()
        # embeddings_dict: {modality: tensor}
        for k, v in embeddings_dict.items():
            if v is None:
                # This part should only be used when
                # using alternating training.
                continue
            self.out_dict[k] = self.out_layers[k](v)
        
        out = torch.sum(torch.stack(list(self.out_dict.values())), dim=0)
        return out

    def get_out_m(self, modality_name):
        if modality_name in self.out_dict:
            return self.out_dict[modality_name]
        else:
            raise ValueError("Modality name not found in the output dict") 
