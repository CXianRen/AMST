import argparse
import os,sys, time
import yaml

import numpy as np

from dataset.dataset import DATASET_LIST, \
    TVA_SET_LIST, AV_SET_LIST, TV_SET_LIST, \
    get_num_classes, build_train_val_test_datasets
    
    
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from models.basic_model import UniR18, UniBERT, weight_init
from utils import setup_seed, TeeOutput

from metrics import performanceMetric

from torch.utils.tensorboard import SummaryWriter

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--optimizer', default='sgd', type=str, choices=['sgd', 'adam'])
    parser.add_argument('--learning_rate', default=0.001, type=float, help='initial learning rate')
    parser.add_argument('--lr_decay_step', default=70, type=int, help='where learning rate decays')
    parser.add_argument('--lr_decay_ratio', default=0.1, type=float, help='decay coefficient')
    parser.add_argument('--random_seed', default=0, type=int)
    parser.add_argument('--save_path', required=True, type=str, help='path to save trained models')
    parser.add_argument('--modality', default='audio', type=str, choices=['audio', 'visual', 'text'])
    parser.add_argument('--use_audio_augm', default=False, type=bool)
    parser.add_argument('--name_prefix', default='', type=str)
    return parser.parse_args()

args = get_arguments()

# make sure the save path exists
save_path = os.path.join(args.save_path,
        "{}Single_Moddality_{}_{}_{}_lr{}".format(
            args.name_prefix,
            args.dataset, args.modality, 
            args.random_seed, 
            args.learning_rate)
)

if os.path.exists(save_path):
    print("Clearing directory: {}".format(save_path))
    os.system("rm -rf {}".format(save_path))

print("Creating directory: {}".format(save_path))
os.makedirs(save_path)

sys.stdout = TeeOutput(os.path.join(save_path, 'output.log'))
    

print("--------Single Modality Model --------")
print("------- Using args --------")
print(yaml.dump(vars(args)))
print("--------------------------")


# set environment
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
gpu_ids = list (range(torch.cuda.device_count()))
device = torch.device("cuda:0")


print("matmul.allow_tf32 = " + str(torch.backends.cuda.matmul.allow_tf32))
print("cudnn.allow_tf32 = " + str(torch.backends.cudnn.allow_tf32))

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
print("Setting matmul.allow_tf32 and cudnn.allow_tf32 to True")
print("matmul.allow_tf32 = " + str(torch.backends.cuda.matmul.allow_tf32))
print("cudnn.allow_tf32 = " + str(torch.backends.cudnn.allow_tf32))

setup_seed(args.random_seed)

# load single modality model
if args.modality == 'visual' or args.modality == 'audio':
    print("Using ResNet18 model")
    model = UniR18(args)
    model.apply(weight_init)
if args.modality == 'text':
    print("Using BERT model")
    model = UniBERT(args)

model.to(device)


# load dataset
train_dataset, val_dataset, test_dataset\
      = build_train_val_test_datasets(args)

train_dataloader = DataLoader(
    train_dataset, batch_size=args.batch_size,
    shuffle=True, num_workers=16, pin_memory=True)

val_dataloader = DataLoader(
    val_dataset, batch_size=args.batch_size,
    shuffle=False, num_workers=16, pin_memory=True,
    persistent_workers=True)

test_dataloader = DataLoader(
    test_dataset, batch_size=args.batch_size,
    shuffle=False, num_workers=16, pin_memory=True,
    persistent_workers=True)

# define optimizer and scheduler
optimizer = optim.SGD(model.parameters(), 
                      lr=args.learning_rate, 
                      momentum=0.9, 
                      weight_decay=1e-4)

scheduler = optim.lr_scheduler.StepLR(optimizer, 
                      args.lr_decay_step,
                      args.lr_decay_ratio)

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


train_time_list = []

def train(args, epoch, model, device, dataloader, optimizer, scheduler) -> performanceMetric:
    n_classes = get_num_classes(args.dataset)
    x_m = performanceMetric(n_classes)
    model.train()
    criterion = nn.CrossEntropyLoss()
    softmax = nn.Softmax(dim=1)
    global train_time_list
    
    for step, data_packet in enumerate(dataloader):
        # do forward
        if step == 0:
            start_time = time.time()
        # with autocast(dtype=torch.bfloat16): 
        predictions, loss, labels, extra_infos = \
            batch_forward(args, model, data_packet, device=device, \
                criterion=criterion, softmax=softmax)
        # do backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # update performance metric
        x_m.update(predictions, labels, extra_infos, loss)
    
    end_time = time.time()
    train_time_list.append(end_time - start_time)
    print("Epoch: {}, step: {}, time: {:.3f}".format(epoch, step, end_time - start_time))
        
    scheduler.step()

    return x_m

# test code
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


# main training loop
best_acc = 0
model_path = os.path.join(save_path, "best_model.pth")

tsb_writer = SummaryWriter(save_path)

def print_loss_and_acc(epoch, x_m, dataloader, writer=None, name="train"):
    loss = x_m.loss/len(dataloader)

    acc = x_m.get_acc()

    print("Loss: {:.4f}".format(loss))
    print("Accuracy: {:.4f}".format(acc))

    if writer is not None:
        writer.add_scalars("{}/loss".format(name), {
            'loss': loss
        }, epoch)

        writer.add_scalars("{}/acc".format(name), {
            'acc': acc
        }, epoch)


for epoch in range(args.epochs):
    s_time = time.time()
    print("\n######## Start  epoch: {} ##########\n".format(epoch))

    print("\n-------- Training -----------\n")
    train_m = train(args, epoch, model, device,
                            train_dataloader, optimizer, scheduler)
    print_loss_and_acc(epoch, train_m, train_dataloader, tsb_writer, "train")

    print("\n-------- Validation -----------\n")
    val_m = test(args, model, val_dataloader, device)
    print_loss_and_acc(epoch, val_m, val_dataloader, tsb_writer, "val")

  
    print("\n--------- Testing  -----------\n")
    test_m = test(args, model, test_dataloader, device)
    print_loss_and_acc(epoch, test_m, test_dataloader, tsb_writer, "test")

    acc = val_m.get_acc()

    if acc > best_acc:
        best_acc = acc
        # save model and args to ckpt
        save_dict = {
            "model": model.state_dict(),
            "args": args,
            "epoch": epoch,
            "best_acc": best_acc,
        }
        torch.save(save_dict, model_path)
        print("Model saved: {}".format(model_path))

    print("\n-------- Time info --------\n")
    print("Best accuracy: {:.4f}".format(best_acc))
    print("modality: {}".format(args.modality))
    print("Time: {:.2f}, remaining time: {:.2f}".format(
            time.time() - s_time, 
            (args.epochs - epoch) * (time.time() - s_time)))
    
    # save time to tensorboard
    tsb_writer.add_scalar("time", time.time() - s_time, epoch)



print("\n######### Training finished ##########")
print("Best accuracy: {:.4f}".format(best_acc))

print("Average training time: {:.4f}".format(
    np.mean(train_time_list)))