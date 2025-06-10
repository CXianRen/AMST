# The original code is from OGM-GE official repo
# We refactored the code by removing the unrelated parts
# only keep the core parts for the OGM-GE

import torch
import torch.nn as nn

from common import BasicTrainer, MAIN_DEVICE_KEY, update_arg

from models.basic_model import  M_TEXT_NAME, M_AUDIO_NAME, \
    M_VISUAL_NAME, KEY_HELPERS, KEY_ENCODERS, KEY_FUSION

from models.basic_model import \
    forward_encoders, forward_fusion, forward_helper


class OGMTrainer(BasicTrainer):
    def __init__(self, args_str=None):
        super(OGMTrainer, self).__init__(args_str)

        self.relu = nn.ReLU(inplace=True)
        self.tanh = nn.Tanh()
        
    def init_parser(self):
        parser = super().init_parser()
        update_arg(parser, '--prefix', default='OGM', type=str,
                            help='prefix for the save path')
        
        # 'gated', 'film' are not supported in the original code:
        #  https://github.com/CXianRen/OGM-GE_CVPR2022/blob/0da3fe76f1023657c354520c3cdab5aa8185b40b/main.py#L86
        #  and the definition of sum fusion is confused
        #  so we use lsum for the original sum fusion defined in the original code
        #  and use esum for the general accepted sum fusion
        #  can check the class implementation for details
        update_arg(parser, '--fusion_method', default='concat', type=str,
                            choices=['esum', 'lsum', 'concat'])
        
        parser.add_argument('--modulation_starts', default=0,
                            type=int, help='where modulation begins')
        parser.add_argument('--modulation_ends', default=50,
                            type=int, help='where modulation ends')

        # the alpha is a hyper-parameter in the OGM-GE paper
        # they said for vggsound is 0.1, and for CREMAD is 0.8
        # more details can be found in the paper and the official repo
        parser.add_argument('--alpha', default=0.1,
                            type=float, 
                            help='alpha in OGM-GE default is 0.1')
        return parser

    def train_method(self, embedding_dict, labels_device):
                
        # compute the the score for each modality after the forward fusion
        score_dict = {}
        for modality_name in self.modality_name_list:
            out_x = self.model[KEY_FUSION].get_out_m(modality_name)
            out_x_pred = self.softmax(out_x)
            # computing the score for modulation
            score_dict[modality_name] = \
                sum([out_x_pred[i][labels_device[i]] \
                        for i in range(len(labels_device))])
        
        # backward                
        out_f = forward_fusion(self.model[KEY_FUSION], embedding_dict)
            
        loss = self.criterion(out_f, labels_device) 
        self.optimizer_map[KEY_FUSION].zero_grad()
        self.optimizer_map[KEY_ENCODERS].zero_grad()
        loss.backward()
              
        # apply modulation       
        def apply_modulation(score_dict):
            """
                This part is the core of the OGM-GE
                
                Notice! 
                1. just modulate the modality with the highest ratio
                And the original code is only supporting 2 modality
                but in the paper based on it like AGM, making it 
                support 3 modality. And when the modality is 2,
                the result is the same as the original code.
                
                the original code:
                    ratio_v = score_v / score_a
                    ratio_a = 1 / ratio_v
                    
                    if ratio_v > 1:
                        coeff_v = 1 - tanh(args.alpha * relu(ratio_v))
                        coeff_a = 1
                    else:
                        coeff_a = 1 - tanh(args.alpha * relu(ratio_a))
                        coeff_v = 1
                    
                2. And only modulating the 4D tensor (the original code), 
                and ignore the 2D tensor is confusing, 
                let me know if you have any idea
                
                3. Apparently, the fusion layer is not modulated,
                and the learning rate is set as the initial learning rate
            """

            # compute ratio
            ratio_dict = {}
            
            for m in self.modality_name_list:
                m_score = score_dict[m]
                
                # sum the scores of all other modalities
                # except the current one
                # this is the key point of the modulation
                sum_score_wo_m = sum([score_dict[other_m] for other_m in \
                    self.modality_name_list if other_m != m])
                
                ratio_dict[m] = m_score / sum_score_wo_m

            # compute the coeff based on the ratio
            coeff_dict = {m: torch.tensor(1) for m in self.modality_name_list}
            highest_ratio_modality = max(ratio_dict, key=ratio_dict.get)
            highest_ratio = ratio_dict[highest_ratio_modality]
            
            coeff_dict[highest_ratio_modality] = \
                1 - self.tanh(self.args.alpha * self.relu(highest_ratio))
                
            if self.args.modulation_starts <= self.epoch <= self.args.modulation_ends:
                for modality_name in self.modality_name_list:
                    # only the encoders are modulated, from the original code
                    encoder = self.model[KEY_ENCODERS][modality_name]
                    coeff = coeff_dict[modality_name]
                    coeff = coeff.to(self.device_map[modality_name])
                    for name, parms in encoder.named_parameters():
                        if parms.requires_grad == False:
                            # for pretrained encoders, we only train
                            # part of the model
                            continue
                        
                        if len(parms.grad.size()) == 4:
                            # Notice!
                            # modulating the gradient and plus a noise
                            nosie = torch.zeros_like(parms.grad).normal_(
                                0, parms.grad.std().item() + 1e-8).to(
                                parms.device)        
                            parms.grad = parms.grad * coeff + nosie

            
            return ratio_dict, coeff_dict
        
        apply_modulation(score_dict) 
        
        # update the fusion layer and encoders
        self.optimizer_map[KEY_FUSION].step()
        self.optimizer_map[KEY_ENCODERS].step()
        
if __name__ == "__main__":
    trainer = OGMTrainer()
    trainer.train_validate()