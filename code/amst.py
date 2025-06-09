# This is our method

import torch
from models.basic_model import  M_TEXT_NAME, M_AUDIO_NAME, M_VISUAL_NAME, \
    KEY_HELPERS, KEY_ENCODERS, KEY_FUSION, KEY_TEXT_TOKENS, KEY_TEXT_PADDING_MASK

from common import MAIN_DEVICE_KEY, BasicTrainer, update_arg, forward_fusion

class AMSTTrainer(BasicTrainer):
    def __init__(self, args_str=None):
        super(AMSTTrainer, self).__init__(args_str)
        
        self.m_skip_factor_map = {
            M_AUDIO_NAME: self.args.a_skip_factor,
            M_VISUAL_NAME: self.args.v_skip_factor,
            M_TEXT_NAME: self.args.t_skip_factor
        }
        
        self.skip_list_map = {
        }

    def init_parser(self):
        parser = super().init_parser()
        update_arg(parser, '--prefix', default='AMST', type=str,
                            help='prefix for the save path')
        # skip a modality or not, 1 means no skip
        parser.add_argument('--a_skip_factor', default=1, type=int,
                            help='skip factor for audio')
        parser.add_argument('--v_skip_factor', default=1, type=int,
                            help='skip factor for visual')
        parser.add_argument('--t_skip_factor', default=1, type=int,
                            help='skip factor for text')
        return parser
    
    def prepare_filled_input(self, modality_name, embedding_dict):
        # fill mute input 
        # zero_embed = torch.zeros_like(embedding_dict[modality_name]).to(
        #     device_map[MAIN_DEVICE_KEY])
        fill_embed = None
        # fill_embed will be updated by the fusion layer itself
        # because different fusion methods have different fill_embed
        # although most of the time, it is just a zero tensor
    
        temp_input_dict = dict()
        for name in self.modality_name_list:
            if name == modality_name:
                temp_input_dict[name] = embedding_dict[name].detach().to(
                    self.device_map[MAIN_DEVICE_KEY])   
                temp_input_dict[name].requires_grad = True
            else:
                temp_input_dict[name] = fill_embed
        return temp_input_dict    

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
            

    def alter_train_with_stream(self, embedding_dict, labels_device):
        """
        use multi-streams to speed up the training
        """
        temp_grad_dict = {}
        with torch.profiler.record_function("fusion layer"):
            for modality_name in self.modality_name_list:

                temp_input_dict = self.prepare_filled_input(modality_name, embedding_dict)
                
                out_m = forward_fusion(self.model[KEY_FUSION], temp_input_dict)
                # out_m = model[KEY_FUSION].get_out_m(modality_name)
                loss = self.criterion(out_m, labels_device) 
                
                if self.epoch % int(self.m_skip_factor_map[modality_name]) != 0:
                    # skip the gradient update
                    continue
                
                self.optimizer_map[KEY_FUSION].zero_grad()
                loss.backward()
                temp_grad_dict[modality_name] = \
                    temp_input_dict[modality_name].grad
                self.optimizer_map[KEY_FUSION].step()

        with torch.profiler.record_function("encoder"):   
            self.optimizer_map[KEY_ENCODERS].zero_grad()
            device_used = set()
            
            for modality_name in temp_grad_dict.keys():
                # we only need to update the encoder recorded in the 
                # temp_grad_dict
                e_device = self.device_map[modality_name]
                device_used.add(e_device)
                
                if self.backward_stream_map.get(modality_name) is None:
                    self.backward_stream_map[modality_name] = \
                        torch.cuda.Stream(device=e_device)
                with torch.cuda.stream(self.backward_stream_map[modality_name]):
                    embedding_dict[modality_name].backward(
                        temp_grad_dict[modality_name].to(e_device))
            # sync all the devices
            for e_device in device_used:
                torch.cuda.synchronize(e_device)
            self.optimizer_map[KEY_ENCODERS].step()

    def train_method(self, embedding_dict, labels_device):
        self.alter_train_with_stream(
            embedding_dict, labels_device)

if __name__ == "__main__":
    trainer = AMSTTrainer()
    trainer.train_validate()