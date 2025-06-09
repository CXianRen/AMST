import argparse
import os,sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset.dataset import DATASET_LIST, \
    TVA_SET_LIST, AV_SET_LIST, TV_SET_LIST, \
    get_num_classes, build_train_val_test_datasets

from models.basic_model import  \
    M_TEXT_NAME, M_AUDIO_NAME, M_VISUAL_NAME, \
    KEY_HELPERS, KEY_ENCODERS, KEY_FUSION, \
    KEY_TEXT_TOKENS, KEY_TEXT_PADDING_MASK, \
    forward_encoders, forward_fusion, forward_helper, \
    gen_model
    

from metrics import performanceMetric


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--model_path', required=True, type=str, help='path to load trained models')
    return parser.parse_args()

args = get_arguments()

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
gpu_ids = list (range(torch.cuda.device_count()))
device = torch.device("cuda:0")


# load single modality model
model = gen_model(args)

# load model weights
model_path = args.model_path
if os.path.isfile(model_path):
    print("Loading model from {}".format(model_path))
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model'])
else:
    print("Model file does not exist: {}".format(model_path))
    sys.exit(1)    

model.to(device)


_, _, test_dataset\
      = build_train_val_test_datasets(args)

test_dataloader = DataLoader(
    test_dataset, batch_size=args.batch_size,
    shuffle=False, num_workers=4, pin_memory=True)

modality_name_list = list(model[KEY_ENCODERS].keys())
   
m = performanceMetric(get_num_classes(args.dataset))
    
    
def prepare_input_dict(dataset, data_packet):
    """
    Prepare the input dictionary for the model based on the dataset and data packet.
    Args:
        dataset (str): The name of the dataset.
        data_packet (tuple): The data packet containing the input features and labels.
        # device_map (dict): A dictionary mapping device names to device objects.
    Returns:
        input_dict (dict): A dictionary containing the input features for the model.
        labels (torch.Tensor): The labels for the input data.
        extra_infos (list): Additional information about the input data (id).
    """

    if dataset in TVA_SET_LIST:
        tokenizers, padding_masks, images, audio_features, \
        labels, extra_infos = data_packet
        tokenizers = tokenizers.to(device)
        padding_masks = padding_masks.to(device)
        audio_features = audio_features.to(device).unsqueeze(1).float()
        images = images.to(device).float()
        input_dict = {
            M_TEXT_NAME: {KEY_TEXT_TOKENS: tokenizers, 
                        KEY_TEXT_PADDING_MASK: padding_masks},
            M_AUDIO_NAME: audio_features,
            M_VISUAL_NAME: images
        }
    elif dataset in AV_SET_LIST:
        audio_features, images, labels, extra_infos = data_packet
        audio_features = audio_features.to(device).unsqueeze(1).float()
        images = images.to(device).float()
    
        input_dict = {
            M_AUDIO_NAME: audio_features,
            M_VISUAL_NAME: images
        }
    elif dataset in TV_SET_LIST:
        tokenizers, padding_masks, images, labels, extra_infos = data_packet
        tokenizers = tokenizers.to(device)
        padding_masks = padding_masks.to(device)
        images = images.to(device).float()
        input_dict = {
            M_TEXT_NAME: {KEY_TEXT_TOKENS: tokenizers, 
                        KEY_TEXT_PADDING_MASK: padding_masks},
            M_VISUAL_NAME: images
        }
    else:
        raise NotImplementedError(f"Dataset not supported: {dataset}")
    return input_dict, labels, extra_infos


def forward(model, data_packet):
    """
    Forward pass for the model.
    forward the encoders, fusion layer and helper
    """  
    softmax = nn.Softmax(dim=1)
    
    input_dict, labels, infos = \
       prepare_input_dict(args.dataset, data_packet)
            
    labels_device = labels.to(device)
    
    embedding_dict = forward_encoders(model[KEY_ENCODERS], input_dict)
    
    out_f = forward_fusion(model[KEY_FUSION], embedding_dict)
    out_f_pred = softmax(out_f)
    m.update(out_f_pred, labels_device)

for step, data_packet in enumerate(test_dataloader):
    ### forward ###
    forward(data_packet)



print("Test results:")
print("Accuracy: {:.4f}".format(m.get_acc()))
print("Class Acc: %s" % m.get_class_acc())