import argparse
import os,sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset.dataset import \
    get_num_classes, build_train_val_test_datasets
    
    
from models.basic_model import UniR18, UniBERT

from metrics import performanceMetric


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--model_path', required=True, type=str, help='path to load trained models')
    parser.add_argument('--modality', type=str, choices=['audio', 'visual', 'text'])
    return parser.parse_args()

args = get_arguments()

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
gpu_ids = list (range(torch.cuda.device_count()))
device = torch.device("cuda:0")


# load single modality model
if args.modality == 'visual' or args.modality == 'audio':
    print("Using ResNet18 model")
    model = UniR18(args)
if args.modality == 'text':
    print("Using BERT model")
    model = UniBERT(args)

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


def batch_forward(args, model, data_packet, device="cuda", 
                  criterion = nn.CrossEntropyLoss(), softmax = nn.Softmax(dim=1)):
    if args.dataset in ["URFUNNY", "IEMOCAP3"]:
        # 3 modalities: text, audio, visual
        # TVADataset instance
        tokenizers, padding_masks, images, audio_features, labels, extra_infos = data_packet
    elif args.dataset in ["CREMAD", "AVE", "SSW60"]:
        # 2 modalities: audio, visual
        # AVDataset instance
        audio_features, images, labels, extra_infos = data_packet
    elif args.dataset in ["MVSA"]:
        # 2 modalities: text, visual
        # TVDataset instance
        tokenizers, padding_masks, images, labels, extra_infos = data_packet
    else:
        print("Dataset not supported: {}".format(args.dataset))
        raise NotImplementedError
        
    if args.modality == "text":
        tokenizers = tokenizers.to(device)
        padding_masks = padding_masks.to(device)
        out = model(tokenizers, padding_masks)
    elif args.modality == "audio":
        input = audio_features.to(device).unsqueeze(1).float()
        out = model(input)
    elif args.modality == "visual":
        input = images.to(device).float()
        out = model(input)
    else:
        print("Modality not supported: {}".format(args.modality))
        raise NotImplementedError
        
    loss = criterion(out, labels.to(device))
    predictions = softmax(out)    
    return predictions, loss, labels, extra_infos

def test(args, model, dataloader, device) -> performanceMetric:
    n_classes = get_num_classes(args.dataset)
    x_m = performanceMetric(n_classes)
    criterion = nn.CrossEntropyLoss()
    softmax = nn.Softmax(dim=1)
    
    with torch.no_grad():
        model.eval()
        for _, (data_packet) in enumerate(dataloader):
            predictions, loss, labels, extra_infos = \
                batch_forward(args, model, data_packet, device=device,
                    criterion=criterion, softmax=softmax)
                
            x_m.update(predictions, labels, extra_infos, loss)

    return x_m


test_m = test(args, model, test_dataloader, device)

print("Test results:")
print("Accuracy: {:.4f}".format(test_m.get_acc()))
print("Class Acc: %s" % test_m.get_class_acc())