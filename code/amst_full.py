# This is our method
import os 

import torch
import torch.nn as nn
import torch.optim as optim

from models.basic_model import  \
    M_TEXT_NAME, M_AUDIO_NAME, M_VISUAL_NAME, \
    KEY_HELPERS, KEY_ENCODERS, KEY_FUSION, \
    KEY_TEXT_TOKENS, KEY_TEXT_PADDING_MASK, \
    forward_encoders, forward_fusion, forward_helper, \
    gen_model
    

from common import MAIN_DEVICE_KEY, BasicTrainer, \
    update_arg, forward_fusion, MAIN_DEVICE_KEY, gen_model
    
from utils import print_args, set_save_path, TeeOutput, \
    printDebugInfo, setup_seed

from metrics import performanceMetric
    
# for full strcuture, it bascally using 2 independent mmodels with different
# fusion methods, which means there are 4 encoders and 2 fusion layers for 2 modalities.


class AMST_F_Trainer(BasicTrainer):
    def __init__(self, args_str=None):
        self.parser = self.init_parser()
        self.args = self.init_parser().parse_args(args_str)
        self.init_logging()
        print_args(self.args)
        self.init_multi_gpu_env()
        self.init_env()
        self.model, self.device_map = self.init_model()
        self.modality_name_list = list(self.model["A"][KEY_ENCODERS].keys())
        self.modality_name_list.sort()

        self.m_skip_factor_map = {
            M_AUDIO_NAME: self.args.a_skip_factor,
            M_VISUAL_NAME: self.args.v_skip_factor,
            M_TEXT_NAME: self.args.t_skip_factor
        }
        
        self.skip_list_map = {
        }
        
        try:
            self.init_dataloader(self.args.using_ploader)
            self.init_optimizer_scheduler()
        except Exception as e:
            print(f"Error in init_dataloader: {e}")
            self.release()
            raise e
        
        self.softmax=nn.Softmax(dim=1)
        self.criterion=nn.CrossEntropyLoss()
        self.train_val_epoch_time_list=[]

    def init_parser(self):
        parser = super().init_parser()
        update_arg(parser, '--prefix', default='AMST-FULL', type=str,
                            help='prefix for the save path')
        # skip a modality or not, 1 means no skip
        parser.add_argument('--a_skip_factor', default=1, type=int,
                            help='skip factor for audio')
        parser.add_argument('--v_skip_factor', default=1, type=int,
                            help='skip factor for visual')
        parser.add_argument('--t_skip_factor', default=1, type=int,
                            help='skip factor for text')
        return parser

    def init_model(self):
        device_map = {
            M_TEXT_NAME: 0,
            M_AUDIO_NAME: 0,
            M_VISUAL_NAME: 0,
            MAIN_DEVICE_KEY: 0
        }  
            
        self.args.fusion_method = 'msum'
        self.alt_model = gen_model(self.args)
        self.args.fusion_method = 'concat'
        self.joint_model = gen_model(self.args)
  
        model = nn.ModuleDict()
        model["J"]= self.joint_model
        model["A"] = self.alt_model
        
        print(model, 
        file=open(os.path.join(
            self.save_path, 
            'model.txt'), 'w'))
        
        model.to(device_map[MAIN_DEVICE_KEY])
            
        return model, device_map
    
    def init_optimizer_scheduler(self):
        # assign an individual optimizer for each model part
        
        def init_optimizer_scheduler(model):
            optimizer_map = {
                key: optim.SGD(
                    model[key].parameters(),
                    lr=self.args.learning_rate, 
                    momentum=0.9, 
                    weight_decay=1e-4
                ) for key in [KEY_ENCODERS, KEY_FUSION, KEY_HELPERS]
            }
            # assign an individual scheduler for each model part
            scheduler_map = {
                key: optim.lr_scheduler.StepLR(
                    optimizer_map[key], 
                    self.args.lr_decay_step,
                    self.args.lr_decay_ratio
                ) for key in [KEY_ENCODERS, KEY_FUSION, KEY_HELPERS]
            }
            return optimizer_map, scheduler_map
        
        self.J_optimizer_map, self.J_scheduler_map = \
            init_optimizer_scheduler(self.model["J"])
        self.A_optimizer_map, self.A_scheduler_map = \
            init_optimizer_scheduler(self.model["A"])
                
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

    def reinitialize_metrics(self):
        """
        Reinitialize the metrics，
        these metrics are common for train, val and test
        and all methods.
        """ 
        # record the fusion results, and the results of each modality
        self.m_map = {}
        self.m_map["f"] = performanceMetric(self.n_classes, name="f")
        self.m_map["alt"] = performanceMetric(self.n_classes, name="alt")
        self.m_map["joint"] = performanceMetric(self.n_classes, name="joint")
        
        for modality_name in self.modality_name_list:
            name="alt_"+modality_name
            self.m_map[name] = performanceMetric(
                self.n_classes, name)
            name="joint_"+modality_name
            self.m_map[name] = performanceMetric(
                self.n_classes, name)
            
        # record the helper results of each modality
        self.m_h_map = {}
        for modality_name in self.modality_name_list:
            name = f"alt_{modality_name}_h"
            self.m_h_map[name] = performanceMetric(
                self.n_classes, name)
            self.m_h_map[name] = performanceMetric(
                self.n_classes, name)
            
    def forward(self, data_packet):
        """
        Forward pass for the model.
        forward the encoders, fusion layer and helper
        """
                 
        device_map = self.device_map
        modality_name_list = self.modality_name_list
        softmax= self.softmax
        criterion= self.criterion
        m_map = self.m_map
        m_h_map = self.m_h_map
        
        input_dict, labels, infos = \
            self.prepare_input_dict(self.args.dataset, 
                                    data_packet)
                
        labels_device = labels.to(device_map[MAIN_DEVICE_KEY])
        
      
        alt_embedding_dict = forward_encoders(self.model["A"][KEY_ENCODERS], input_dict)
        joint_embedding_dict = forward_encoders(self.model["J"][KEY_ENCODERS], input_dict)
        
        
        local_alt_embedding_dict = {}
        loacl_joint_embedding_dict = {}
        for modality_name in modality_name_list:
            # it is just for BL parallelism (move data to the main device)
            # for single GPU, DP parallelism (it is not needed, but for consistency,
            # all data is moved automatically to the main device)
            # for DDP parallelism, main device is same as the rank of the process
            local_alt_embedding_dict[modality_name] = \
                alt_embedding_dict[modality_name].detach().to(
                device_map[MAIN_DEVICE_KEY])
                
            loacl_joint_embedding_dict[modality_name] = \
                joint_embedding_dict[modality_name].detach().to(
                device_map[MAIN_DEVICE_KEY])
            
            
        # fusion step
        with torch.no_grad():
            alt_out_f = forward_fusion(self.model["A"][KEY_FUSION],
                local_alt_embedding_dict)
            joint_out_f = forward_fusion(self.model["J"][KEY_FUSION],
                loacl_joint_embedding_dict)    
            final_out_f = alt_out_f + joint_out_f
        
            final_out_f_pred = softmax(final_out_f)
            final_out_f_loss = criterion(final_out_f, labels_device)
            if m_map is not None and "f" in m_map:
                m_map["f"].update(final_out_f_pred, labels_device, loss=final_out_f_loss)
                
            alt_out_f_pred = softmax(alt_out_f)
            alt_out_f_loss = criterion(alt_out_f, labels_device)
            if m_map is not None and "alt" in m_map:
                m_map["alt"].update(alt_out_f_pred, labels_device, loss=alt_out_f_loss)
            
            joint_out_f_pred = softmax(joint_out_f)
            joint_out_f_loss = criterion(joint_out_f, labels_device)
            if m_map is not None and "joint" in m_map:
                m_map["joint"].update(joint_out_f_pred, labels_device, loss=joint_out_f_loss)
            
            for modality_name in modality_name_list:
                mp_name = f"alt_{modality_name}"
                if m_map is not None and mp_name in m_map:
                    out_x = self.model["A"][KEY_FUSION].get_out_m(modality_name)
                    out_x_pred = softmax(out_x)
                    out_x_loss = criterion(out_x, labels_device)
                    m_map[mp_name].update(out_x_pred, labels_device, loss=out_x_loss)
            
                mp_name = f"joint_{modality_name}"
                if m_map is not None and mp_name in m_map:
                    out_x = self.model["J"][KEY_FUSION].get_out_m(modality_name)
                    out_x_pred = softmax(out_x)
                    out_x_loss = criterion(out_x, labels_device)
                    m_map[mp_name].update(out_x_pred, labels_device, loss=out_x_loss)
            
                    
        # helper metric
        alt_helper_out_dict = forward_helper(self.model["A"][KEY_HELPERS], 
            local_alt_embedding_dict)
        joint_helper_out_dict = forward_helper(self.model["J"][KEY_HELPERS],
            loacl_joint_embedding_dict)
        
        
        for modality_name in modality_name_list:
            mp_name=f"alt_{modality_name}_h"
            if m_h_map is not None and mp_name in m_h_map:
                out_h_x = alt_helper_out_dict[modality_name]
                out_h_x_pred = softmax(out_h_x)
                out_h_x_loss = criterion(out_h_x, labels_device)
                m_h_map[mp_name].update(
                    out_h_x_pred, labels_device, loss=out_h_x_loss)
            
            mp_name=f"joint_{modality_name}_h"
            if m_h_map is not None and mp_name in m_h_map:
                out_h_x = joint_helper_out_dict[modality_name]
                out_h_x_pred = softmax(out_h_x)
                out_h_x_loss = criterion(out_h_x, labels_device)
                m_h_map[mp_name].update(
                    out_h_x_pred, labels_device, loss=out_h_x_loss)
                    
        return alt_embedding_dict, joint_embedding_dict, \
            alt_helper_out_dict, joint_helper_out_dict , labels_device

    def valid(self, dataloader): 
        self.reinitialize_metrics()         
        self.model.eval()
        
        with torch.no_grad():
            for step, data_packet in enumerate(dataloader):
                ### forward ###
                self.forward(data_packet)
                
    def alt_train(self, embedding_dict, labels_device, model, optimizer_map):
        for modality_name in self.modality_name_list:
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
                    temp_input_dict[name] = embedding_dict[name] 
                else:
                    temp_input_dict[name] = fill_embed
                  
            out_m = forward_fusion(model[KEY_FUSION], temp_input_dict)
            # out_m = model[KEY_FUSION].get_out_m(modality_name)
            loss = self.criterion(out_m, labels_device) 
            
            if self.epoch % int(self.m_skip_factor_map[modality_name]) != 0:
                # skip the gradient update
                continue
            
            optimizer_map[KEY_FUSION].zero_grad()
            optimizer_map[KEY_ENCODERS].zero_grad()
            loss.backward()
            optimizer_map[KEY_FUSION].step()
            optimizer_map[KEY_ENCODERS].step()

    def joint_train(self, embedding_dict, labels_device, model, optimizer_map):
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
        
        out_m = forward_fusion(model[KEY_FUSION], temp_input_dict)
            # out_m = model[KEY_FUSION].get_out_m(modality_name)
        loss = self.criterion(out_m, labels_device) 
            
        optimizer_map[KEY_FUSION].zero_grad()
        optimizer_map[KEY_ENCODERS].zero_grad()
        loss.backward()
        optimizer_map[KEY_FUSION].step()
        optimizer_map[KEY_ENCODERS].step()
        
    def train_epoch(self, dataloader): 
        
        self.reinitialize_metrics()        
        self.model.train() 
    
        for step, data_packet in enumerate(dataloader):

            ### forward ###
            alt_embedding_dict, joint_embedding_dict, \
            alt_helper_out_dict, joint_helper_out_dict , labels_device = \
                self.forward(data_packet)
           
                
            # train each modality alternatively
            self.alt_train(alt_embedding_dict, labels_device,
                self.model["A"], self.A_optimizer_map)
            
            self.alt_train(joint_embedding_dict, labels_device,
                self.model["J"], self.J_optimizer_map)
          
            ### backward ###
            # backward helper, we don't update the backbone, just update the helper
            
            self.A_optimizer_map[KEY_HELPERS].zero_grad()
            self.J_optimizer_map[KEY_HELPERS].zero_grad()
            for modality_name in self.modality_name_list:
                alt_loss = self.criterion(
                    alt_helper_out_dict[modality_name], labels_device)
                alt_loss.backward()
                
                joint_loss = self.criterion(
                    joint_helper_out_dict[modality_name], labels_device)
                joint_loss.backward()
            self.A_optimizer_map[KEY_HELPERS].step()
            self.J_optimizer_map[KEY_HELPERS].step()

        for sch in self.J_scheduler_map.values():
            sch.step()
        for sch in self.A_scheduler_map.values():
            sch.step()


if __name__ == "__main__":
    trainer = AMST_F_Trainer()
    trainer.train_validate()