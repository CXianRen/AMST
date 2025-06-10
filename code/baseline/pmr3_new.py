# This file is based on the original code 
# from PMR official repo. We refactor the 
# code to make it more compact with our project structure
# It does't support more than 2 modalities due to the way 
# of computing beta and lambada which are the main hyperparameters
# for each modality.Refer link:
# https://github.com/fanyunfeng-bit/Modal-Imbalance-PMR/blob/3615e48983555089b0e693380087617c3797a641/main_Prototype_grad.py#L169

# and the result they got might incorrect,
# they saved the best model by monitoring the result of the 
# testing set. We use the validation set to save the best model
# and the test set evaluate the model.

import torch

from metrics import performanceMetric

from dataset.dataset import M2DATASET_LIST

from models.basic_model import  M_TEXT_NAME, M_AUDIO_NAME, \
    M_VISUAL_NAME, KEY_HELPERS, KEY_ENCODERS, \
    KEY_FUSION, KEY_TEXT_TOKENS, KEY_TEXT_PADDING_MASK

from models.basic_model import forward_fusion
    
from common import print_loss_and_acc

from common import BasicTrainer, update_arg

def EU_dist(x1, x2):
    """
        x1: [batch_size, dim]
        x2: [num_classes, dim]
        
        d_matrix: [batch_size, num_classes]
    """
    # d_matrix = torch.zeros(x1.shape[0], x2.shape[0]).to(x1.device)
    # for i in range(x1.shape[0]):
    #     for j in range(x2.shape[0]):
    #         d = torch.sqrt(torch.dot((x1[i] - x2[j]), (x1[i] - x2[j])))
    #         d_matrix[i, j] = d
    
    # a faster way to compute the distance
    d_matrix = torch.cdist(x1, x2, p=2)
    return d_matrix

class PMRTrainer(BasicTrainer):
    def __init__(self, args_str=None):
        super(PMRTrainer, self).__init__(args_str)
        
        # for the prototype of each modality
        self.prototype_dict = {}
        
        # the metric for the prototypical classification
        self.m_p_map = {}
        
    def init_parser(self):
        parser = super().init_parser()
        update_arg(parser, '--prefix', default='PMR', type=str,
                            help='prefix for the save path')

        # only support 2 modalities
        update_arg(parser,'--dataset', default='CREMAD', type=str,
                            choices=M2DATASET_LIST,
                            help='dataset name, choose from {}'.format(
                                M2DATASET_LIST))
        
        parser.add_argument('--modulation_starts', default=0, type=int, help='where modulation begins')
        parser.add_argument('--modulation_ends', default=70, type=int, help='where modulation ends')
        parser.add_argument('--alpha', default=1.0, type=float, help='alpha in Proto')
        parser.add_argument('--momentum_coef', default=0.2, type=float)
        
        #
        parser.add_argument('--factor', default=0.1, type=float,
                            help='the percentage of dataset to calculate the prototype,' +
                            'for AVE, it ishould be 1, from the original code')
    
        return parser
 
    def reinitialize_metrics(self):
        """
            generate the metrics for prototypical classification
        """
        super().reinitialize_metrics()
        
        # prototypical classification accuracy
        self.m_p_map = {
            modality_name : performanceMetric(
                self.n_classes, name= modality_name +'_p'
                ) for modality_name in self.modality_name_list
        } 
    
    def print_metrics(self, mode = "train"):
        super().print_metrics(mode)
        # print the metrics for prototypical classification
        
        dataloader_map = {
            "train": self.train_dataloader,
            "val": self.val_dataloader,
            "test": self.test_dataloader
        }
        batches = len(dataloader_map[mode])
        
        print("Prototypical classification acc:")
        print_loss_and_acc(self.epoch,
                        self.m_p_map, batches,
                        self.tsb_writer, 
                        mode + "_p")
        
        # also save the metrics of each mode
        if mode == "train":
            self.train_m_p_map = self.m_p_map
        elif mode == "val":
            self.val_m_p_map = self.m_p_map
        elif mode == "test":
            self.test_m_p_map = self.m_p_map

    def valid(self, dataloader):
        self.reinitialize_metrics()         
        self.model.eval()
        
        with torch.no_grad():
            for step, data_packet in enumerate(dataloader):
                embedding_dict, _, labels_device = \
                    self.forward( data_packet)

                for modality_name in self.modality_name_list:
                    # similarity to the prototype
                    x_sim = -EU_dist(
                        embedding_dict[modality_name],
                        self.prototype_dict[modality_name])
                    
                    x_sim_pred = self.softmax(x_sim)
                    x_sim_loss = self.criterion(x_sim, labels_device)
                    self.m_p_map[modality_name].update(
                        x_sim_pred, labels_device, 
                        loss=x_sim_loss)
    
    def train_method(self, embedding_dict, labels_device):
        
        # training process
        # compute EU distance (prototyical classification)
        # need to rerun the fusion again because 
        # in forward, the embedding is detached when run
        # the fusion. We need a full graph to compute the loss
        out_f = forward_fusion(
                    self.model[KEY_FUSION], 
                    embedding_dict)
        
        out_f_loss = self.criterion(out_f, labels_device)

        proto_loss_dict = {} 
        proto_score_dict = {}
        for modality_name in self.modality_name_list:
            # similarity to the prototype
            x_sim = -EU_dist(
                embedding_dict[modality_name],
                self.prototype_dict[modality_name])
            
            x_sim_pred = self.softmax(x_sim)
            x_sim_loss = self.criterion(x_sim, labels_device)
            self.m_p_map[modality_name].update(
                x_sim_pred, labels_device, 
                loss=x_sim_loss)
            
            proto_loss_dict[modality_name] = x_sim_loss
            
            # prototypical classification score
            # the code is from the original repo:
            # main_Prototype_grad.py: line 162 
            m_p_score = sum(
                [self.softmax(x_sim)[i][labels_device[i]]
                 for i in range(x_sim.size(0))])
            proto_score_dict[modality_name] = m_p_score

        # compute the factor and proto loss
        # because we only have two modalities
        # so we keep this part as the original code
        m1_name = self.modality_name_list[0]
        m2_name = self.modality_name_list[1]
        
        score_m1 = proto_score_dict[m1_name]
        score_m2 = proto_score_dict[m2_name]
        
        ratio_m1_m2 = score_m1 / score_m2
        
        # Determine beta and lambda based 
        # on the ratio of scores. In the original code,
        # beta is audio coef and lambda is visual coef
        beta, lam = 0, 0  # Default values
        if ratio_m1_m2 > 1:
            lam = 1 * self.args.alpha
        elif ratio_m1_m2 < 1:
            beta = 1* self.args.alpha
            
        # whole model loss
        loss = out_f_loss 
        if self.args.modulation_starts <= self.epoch \
            <= self.args.modulation_ends:
            loss = loss + \
                beta * proto_loss_dict[m1_name] + \
                lam * proto_loss_dict[m2_name]

        # clean the gradient
        for opt in self.optimizer_map.values():
            opt.zero_grad()
        
        # backward
        loss.backward() 
        
        for opt in self.optimizer_map.values():
            opt.step()

    def calculate_prototype(self, dataloader, momentum_coef,
                        proto_dict:dict, factor = 0.1):
        """
            Calculate the prototype for each class
            Because we only have two modalities
            so we keep this part as the original code
            
            Notice: the original code has specific 
            processing for AVE dataset, which is confused. 
        """
        
        count_class = [0 for _ in range(self.n_classes)]
            
        proto_dict_new = {}
        
        # bugfix: forward always requires the metric
        # walk around: just reinitialize the metric, but
        # not use it
        self.reinitialize_metrics()
        
        # calculate prototype
        self.model.eval()
        with torch.no_grad():
            # this actually is the number of batches
            sample_count = 0 
            all_num = len(dataloader) 
            
            for step, data_packet in enumerate(dataloader):
                ### forward ###
                embedding_dict, _, labels_device = \
                    self.forward(data_packet)
                labels = labels_device.cpu()
                
                for modality_name in self.modality_name_list:
                    modality_embedding = embedding_dict[modality_name]
                    
                    if modality_name not in proto_dict_new:
                        proto_dict_new[modality_name] = \
                            torch.zeros(self.n_classes, 
                                modality_embedding.shape[1]
                            ).to(modality_embedding.device)
                
                for idx, label in enumerate(labels):
                    label = label.long()
                    count_class[label] += 1
                    
                    # sum the embedding for each class
                    for modality_name in self.modality_name_list:
                        modality_embedding = embedding_dict[modality_name]
                        proto_dict_new[modality_name][label, :] += \
                            modality_embedding[idx, :]
                
                sample_count += 1   
                if sample_count / all_num >= factor:
                    break
                
        # average the prototype
        for c in range(self.n_classes):
            for modality_name in self.modality_name_list:
                proto_dict_new[modality_name][c, :] /= count_class[c]


        # if the prototype dict is not empty
        if proto_dict != {}:
            for modality_name in self.modality_name_list:
                # new = (1 - momentum_coef) * new + momentum_coef * old
                proto_dict_new[modality_name] = \
                    (1 - momentum_coef) * proto_dict_new[modality_name] + \
                    momentum_coef * proto_dict[modality_name]
                    
        return proto_dict_new

    def before_train_epoch(self):
        # Notice: the original code has specific
        # processing for AVE dataset, which is confused.
        # We use the same way as the original code,
        # and they used the whole validation set, which is 
        # a tricky way to do it. 

        # only use 10% of the training data to calculate the prototype
        # self.args.factor = 0.1 (default)
        if self.args.dataset == 'AVE':
            self.prototype_dict = self.calculate_prototype( 
                                self.val_dataloader,  
                                self.args.momentum_coef, 
                                self.prototype_dict, 
                                factor=self.args.factor)
        else:
            self.prototype_dict = self.calculate_prototype( 
                                self.train_dataloader,  
                                self.args.momentum_coef, 
                                self.prototype_dict, 
                                factor=self.args.factor)   
        print("Prototypes are updated")
        # print(self.prototype_dict)    
  
if __name__ == "__main__":
    trainer = PMRTrainer()
    trainer.train_validate()