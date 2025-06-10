# This is our method

import torch
from models.basic_model import  M_TEXT_NAME, M_AUDIO_NAME, M_VISUAL_NAME, \
    KEY_HELPERS, KEY_ENCODERS, KEY_FUSION, KEY_TEXT_TOKENS, KEY_TEXT_PADDING_MASK

from common import MAIN_DEVICE_KEY, BasicTrainer, update_arg, forward_fusion

class AMST_J_Trainer(BasicTrainer):
    def __init__(self, args_str=None):
        super(AMST_J_Trainer, self).__init__(args_str)
        
        self.m_skip_factor_map = {
            M_AUDIO_NAME: self.args.a_skip_factor,
            M_VISUAL_NAME: self.args.v_skip_factor,
            M_TEXT_NAME: self.args.t_skip_factor
        }
        
        self.skip_list_map = {
        }

    def init_parser(self):
        parser = super().init_parser()
        update_arg(parser, '--prefix', default='AMST-AJ', type=str,
                            help='prefix for the save path')
        
        # for joint training, it is using the concat fusion 
        update_arg(parser, '--fusion_method', default='concat', type=str,
                            choices=["concat"]) 
        
        # skip a modality or not, 1 means no skip
        parser.add_argument('--a_skip_factor', default=1, type=int,
                            help='skip factor for audio')
        parser.add_argument('--v_skip_factor', default=1, type=int,
                            help='skip factor for visual')
        parser.add_argument('--t_skip_factor', default=1, type=int,
                            help='skip factor for text')
        return parser
    
    def before_train_epoch(self):
        for modality_name in self.modality_name_list:
            if self.epoch % int(self.m_skip_factor_map[modality_name]) != 0:
                # skip the gradient update
                if modality_name not in self.skip_list_map:
                    self.skip_list_map[modality_name] = []
                self.skip_list_map[modality_name].append(self.epoch)
                print(f"skip {modality_name} at epoch {self.epoch}")

    def after_summary(self):
        print("SKIP INFO:")
        for modality_name in self.skip_list_map.keys():
            skip_list = self.skip_list_map.get(modality_name)
            len_skip = len(skip_list)
            print(f"{modality_name} skip {len_skip} times:\n{skip_list}")
        print("END SKIP INFO")
            
    def joint_train(self, embedding_dict, labels_device):
        temp_input_dict = dict()
        # fill mute input 
        # zero_embed = torch.zeros_like(embedding_dict[modality_name]).to(
        #     device_map[MAIN_DEVICE_KEY])
        fill_embed = None
        # fill_embed will be updated by the fusion layer itself
        # because different fusion methods have different fill_embed
        # although most of the time, it is just a zero tensor
            
        for modality_name in self.modality_name_list:
            if self.epoch % int(self.m_skip_factor_map[modality_name]) != 0:
                # skip the gradient update
                temp_input_dict[modality_name] = fill_embed
            else:
                temp_input_dict[modality_name] = embedding_dict[modality_name]     
                    
        # if all modalities are skipped, we will not do the forward pass
        if all([temp_input_dict[modality_name] is None for modality_name in self.modality_name_list]):
            return
        
        out_m = forward_fusion(self.model[KEY_FUSION], temp_input_dict)
            # out_m = model[KEY_FUSION].get_out_m(modality_name)
        loss = self.criterion(out_m, labels_device) 
            
        self.optimizer_map[KEY_FUSION].zero_grad()
        self.optimizer_map[KEY_ENCODERS].zero_grad()
        loss.backward()
        self.optimizer_map[KEY_FUSION].step()
        self.optimizer_map[KEY_ENCODERS].step()

    def train_method(self, embedding_dict, labels_device):
        self.joint_train(
            embedding_dict, labels_device)

if __name__ == "__main__":
    trainer = AMST_J_Trainer()
    trainer.train_validate()