# This file is based on the original MLA code,
# link: https://github.com/CXianRen/MLA/blob/main/models/fusion_modules.py
# We support fusion methods:
# 1. esum: Early Sum, the standard definition of early fusion
# 2. lsum: Late Sum, from the original MLA, OGM-GE code, this is fusion
# is a bit wired, but for consistency, we keep it.
# 3. msum: MLA Sum, it is the shared head fusion. It is only used in MLA method 
# and MLA related experiments.
# 4. concat: Early Concat, the standard definition of early fusion
# 5. gated: Gated Fusion. It actually is bi-gated fusion, in our experiments, 
# we always use audio as the control modality (gate or film).
# 6. film: FiLM, same as gated fusion


import torch
import torch.nn as nn
import torch.nn.functional as F

Fusion_List = ['sum', 'sumN', 'concat', 'gated', 'gate','film', 'mtmm', 'esum', 'lsum', 'msum']

def gen_fusion_v2(args, input_dim, output_dim, name_list):
    fusion_methods = {
        'esum': EarlySum,
        'lsum': LateSum,
        'msum': MLASum,
        'concat': EarlyConcat,
        'gated': newGatedFusion,
        'gate': newGatedFusion,
        'film': newFiLM,
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
 
class EarlySum(nn.Module):
    """
    Early Sum fusion module.
    Structure:
    out = W[e1 + e2 + e3] + b = W[e1] + b/n + W[e2] + b/n + W[e3] + b/n
    where n is the number of modalities.
    This is the standard definition of early fusion.
    """
    def __init__(self, intput_dim, output_dim, modality_name_list):
        super(EarlySum, self).__init__()
        #  W[e1 + e2 + e3] + b = W[e1] + b/n + W[e2] + b/n + W[e3] + b/n
        self.fc = nn.Linear(intput_dim, output_dim, bias=False)
        self.bias = nn.Parameter(torch.zeros(output_dim))
        
        self.n_modalities = len(modality_name_list)
        self.out_dict = dict()
        
    def forward(self, embeddings_dict: dict):
        # embeddings_dict: {modality: tensor}
        # sum the embeddings
        #  W[e1 + e2 + e3] + b = W[e1] + b/n + W[e2] + b/n + W[e3] + b/n
        non_zero_values = [v for v in embeddings_dict.values() if v is not None]
        
        for k, v in embeddings_dict.items():
            if v is None:
                # This part should only be used when
                # using alternating training.
                v = torch.zeros_like(non_zero_values[0])

            self.out_dict[k] = self.fc(v) + self.bias/self.n_modalities
        
        # compute the sum of the valid values
        # this equals to :  out = (out1 + out2 + out3)
        out = torch.sum(torch.stack(list(self.out_dict.values())), dim=0)
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
        # embeddings_dict: {modality: tensor}
        # concatenate the embeddings
        #  W[e1, e2, e3] + b = W1[e1] + b/n + W2[e2]+ b/n + W3[e3] + b/n
        non_zero_values = [v for v in embeddings_dict.values() if v is not None]
        for k, v in embeddings_dict.items():
            if v is None:
                # This part should only be used when
                # using alternating training.
                v = torch.zeros_like(non_zero_values[0])
            self.out_dict[k] = self.out_layers[k](v) + self.bias/self.n_modalities
            # require the internal gradient
            if v.requires_grad == True:
                # this is only for monitoring the graident.
                # it is ok to remove this line.
                self.out_dict[k].retain_grad()
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
            self.out_layers[m] = nn.Linear(input_dim, output_dim, bias=True)
    
    def forward(self, embeddings_dict: dict):
        # embeddings_dict: {modality: tensor}
        non_zero_values = [v for v in embeddings_dict.values() if v is not None]
        for k, v in embeddings_dict.items():
            if v is None:
                # This part should only be used when
                # using alternating training.
                v = torch.zeros_like(non_zero_values[0])
            self.out_dict[k] = self.out_layers[k](v)
        
        out = torch.sum(torch.stack(list(self.out_dict.values())), dim=0)
        return out

    def get_out_m(self, modality_name):
        if modality_name in self.out_dict:
            return self.out_dict[modality_name]
        else:
            raise ValueError("Modality name not found in the output dict") 
        
class newGatedFusion(nn.Module):
    """
    Efficient Large-Scale Multi-Modal Classification,
    https://arxiv.org/pdf/1802.02892.pdf.
    
    This is based on the original code from MLA, OGM-GE
    """

    def __init__(self, input_dim, output_dim, modality_name_list: list, dim=512,):
        super(newGatedFusion, self).__init__()

        self.modality_name_list = modality_name_list
        
        # default
        self.gated_modality = modality_name_list[0]
        
        self.fc_x = nn.Linear(input_dim, dim)
        self.fc_y = nn.Linear(input_dim, dim)
        self.fc_out = nn.Linear(dim, output_dim)

        self.sigmoid = nn.Sigmoid()
    
    def forward(self, embeddings_dict: dict):
        x_key = self.modality_name_list[0]
        y_key = self.modality_name_list[1]

        ebed_x = embeddings_dict[x_key]
        ebed_y = embeddings_dict[y_key]

        x_gate = self.gated_modality == x_key
        if ebed_x is None or ebed_y is None:
            # Handle alternative training cases
            # when x is None, we set it to zero and used it as the gate
            # because sigmoid(Fc(0)) ~ 0.5, so then it conveted to a normal 
            # y = 0.5 * y, then it is equal to a simple linear layer
            ebed_x = ebed_x if ebed_x is not None else torch.zeros_like(ebed_y).to(ebed_y.device)
            ebed_y = ebed_y if ebed_y is not None else torch.zeros_like(ebed_x).to(ebed_x.device)
            x_gate = ebed_x is None

        out_x = self.fc_x(ebed_x)
        out_y = self.fc_y(ebed_y)
            
        if x_gate:
            gate = self.sigmoid(out_x)
            # print(gate)
            output = self.fc_out(torch.mul(gate, out_y))
        else:
            gate = self.sigmoid(out_y)
            output = self.fc_out(torch.mul(gate, out_x))
            
        return output
    
    def set_control_modality(self, modality_name):
        if modality_name in self.modality_name_list:
            self.gated_modality = modality_name
        else:
            raise ValueError("Modality name not found in the list")
    
    def get_out_m(self, modality_name):
        raise NotImplementedError("get_out_m not implemented for newGatedFusion")
   
class newFiLM(nn.Module):
    """
    FiLM: Visual Reasoning with a General Conditioning Layer,
    https://arxiv.org/pdf/1709.07871.pdf.
    
    This is based on the original code from MLA, OGM-GE
    """

    def __init__(self, input_dim, output_dim, modality_name_list: list):
        super(newFiLM, self).__init__()
        self.modality_name_list = modality_name_list
        
        self.dim = input_dim
        self.fc = nn.Linear(input_dim, 2 * self.dim)
        self.fc_out = nn.Linear(self.dim, output_dim)
        
        # default
        self.film_modality = modality_name_list[0]

    def forward(self, embeddings_dict: dict, x_film=True):
        x_key = self.modality_name_list[0]
        y_key = self.modality_name_list[1]
        
        ebed_x = embeddings_dict[x_key]
        ebed_y = embeddings_dict[y_key]

        x_film = self.film_modality == x_key
        if ebed_x is None or ebed_y is None:
            # Handle alternative training cases
            # when x is None, we set it to one and used it as the film
            # then is like introducing random noise to the other modality
            # How to eliminate the effect of the other modality is the core
            # of the method, maybe, using protoype is a btter way ? (Todo?) 
            ebed_x = ebed_x if ebed_x is not None else torch.ones_like(ebed_y).to(ebed_y.device)
            ebed_y = ebed_y if ebed_y is not None else torch.ones_like(ebed_x).to(ebed_x.device)
            x_film = ebed_x is None

        if x_film:
            film = ebed_x
            to_be_film = ebed_y
        else:
            film = ebed_y
            to_be_film = ebed_x

        gamma, beta = torch.split(self.fc(film), self.dim, 1)

        output = gamma * to_be_film + beta
        output = self.fc_out(output)

        return output
    
    def set_control_modality(self, modality_name):
        if modality_name in self.modality_name_list:
            self.film_modality = modality_name
        else:
            raise ValueError("Modality name not found in the list")
    
    def get_out_m(self, modality_name):
        raise NotImplementedError("get_out_m not implemented for newFiLM")
        
