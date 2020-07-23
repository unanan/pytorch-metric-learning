# Origin: KevibMusgrave/pytorch-metric-learning
# Modify: unanan
# *REMARKS: So Far Not Supported Distributed Training (07/23/2020)

import os
import argparse
import logging

import torch
from torch import optim

from pytorch_metric_learning import losses as tripletloss
from pytorch_metric_learning import miners as tripletminer

import models
from utils.calc_metrics import calculate_class_accuracy



def train(args):

    ### Prepare Dataset
    if args.data_format=="coco":  #TODO
        from datasets.cocodataset import COCODetection
        trainset = COCODetection(image_path=args.train_images,
                            info_file=args.train_info,
                            transform=SSDAugmentation(MEANS))
        valset = None
    else:
        from datasets.customdataset import CustomDataset
        trainset = CustomDataset()
        valset = None

    trainloader = None
    valloader = None
    # batch_num = len(trainloader)


    ### Init Model
    model = getattr(models, args.backbone)(pretrained = args.backbone_pretrained and not args.pretrained_model)
    if args.cuda:
        model = model.cuda()


    ### Pretrained Model
    if args.pretrained_model:
        model.load_state_dict(torch.load(args.pretrained_model))


    ### Data parallel
    IS_DP_AVAILABLE = False
    try:
        devices = list(map(int,args.devices.strip().split(",")))
        if len(devices)>=2:
            IS_DP_AVAILABLE = True
    except:
        logging.warning(f"Format of args.devices is invalid. {args.devices}")

    if IS_DP_AVAILABLE:
        pass


    ### Init Optimizer
    # optimizer = optim.SGD(model.parameters(), lr=args.start_lr,
    #                       momentum=args.momentum, weight_decay=args.weight_decay)
    optimizer = optim.RMSprop(model.parameters(), lr=args.start_lr, alpha = 0.9, eps=1e-08,
                         momentum=args.momentum, weight_decay=args.weight_decay)


    ### Init Triplet Loss
    loss_func = tripletloss.TripletMarginLoss(triplets_per_anchor=args.triplets_per_anchor, margin=args.triplet_margin)
    if args.mining:
        miner = tripletminer.MultiSimilarityMiner(epsilon=args.mining_epsilon)


    interval = -1
    for epoch in range(args.max_epoch):
        for batch_idx, batch_data in enumerate(trainloader):
            interval+=1

            imgs, labels = batch_data

            optimizer.zero_grad()
            embeddings = model(imgs)

            # Metric Learning
            if args.mining:
                hard_pairs = miner(embeddings, labels)
                loss = loss_func(embeddings, labels, hard_pairs)
            else:
                loss = loss_func(embeddings, labels)
            loss.backward()
            optimizer.step()

            # Print Loss
            if args.print_interval % interval == 0:
                logging.info(f"[{epoch:%4d}/{args.max_epoch:%4d}] {interval:%7d} Triplet Loss: {loss}")

            # Validation
            if args.val_interval% interval == 0 and interval!=0:
                logging.info(f"[{epoch}/{args.max_epoch}] Starting Validating..")
                for valbatch_idx, valbatch_data in enumerate(valloader):
                    val_imgs, val_labels = valbatch_data

                    val_embeddings = model(val_imgs)

                    # Metric Learning
                    val_loss = loss_func(val_embeddings, val_labels)

                    # softmax
                    #...
                    cls_acc = calculate_class_accuracy()

                    if args.val_print_interval % valbatch_idx == 0:
                        logging.info(f"[{epoch:%4d}/{args.max_epoch:%4d}] {interval:%7d} Triplet Loss: {val_loss} Cls Acc: {cls_acc}")



def get_args():
    parser = argparse.ArgumentParser(description='Train Metric Learning')

    # Train
    parser.add_argument('--max_epoch', default=800, help='epoch to train the network')
    parser.add_argument('--backbone', default="resnet50", help='resnet50 | resnet101 | shufflenetv2 | squeezenet | vgg | xception')
    parser.add_argument('--backbone_pretrained', default=True, type=bool, help='if the backbone uses official pretrained model')
    parser.add_argument('--train_batch_size', default=128, help='training batch size ')
    parser.add_argument('--start_lr', default=0.005, help='base learning rate at the beginning')
    parser.add_argument('--momentum', default=0.9, help='base learning rate at the beginning')
    parser.add_argument('--weight_decay', default=2e-5, help='base learning rate at the beginning')
    parser.add_argument('--print_interval', default=100, help='interval of printing the training loss')

    # Triplet Loss
    parser.add_argument('--triplet_margin', default=0.1, type=float, help='')
    parser.add_argument('--triplets_per_anchor', default="all", type=str, help='')
    parser.add_argument('--mining', default=True, type=bool, help='if do mining before doing triplet loss')
    parser.add_argument('--mining_epsilon', default=0.1, type=float, help='')

    # Validate
    parser.add_argument('--val_batch_size', default=120, help='validating batch size')
    parser.add_argument('--val_interval', default=1000, help='interval of validation')
    parser.add_argument('--val_print_interval', default=20, help='interval of printing the training loss')

    # Dataset
    parser.add_argument('--data_format', default="custom", help='coco | custom')
    parser.add_argument('--train_imgdir', default="", help='the train images path')
    parser.add_argument('--val_imgdir', default="", help='the test images path')
    parser.add_argument('--img_maxsize', default=512, help='the image size')
    parser.add_argument('--num_workers', default=4, type=int, help='Number of workers used in dataloading')

    # Misc
    parser.add_argument('--cuda', default=True, type=bool, help='')
    parser.add_argument('--devices', default="0", type=str, help='set as like "0" or "1,2", use comma to split the device ids')
    parser.add_argument('--save_folder', default="weights/", type=str, help='')
    parser.add_argument('--pretrained_model', default='', help='pretrained base model')


    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    if torch.cuda.is_available() and args.cuda:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.devices.strip()
        logging.info(f"Visible devices: {args.devices.strip()}")

    train(args)
