# Origin: KevibMusgrave/pytorch-metric-learning
# Modify: unanan
# *REMARKS: So Far Not Supported Distributed Training (07/23/2020)

import os
import argparse
import torch
import logging
import models
from utils.calc_metrics import calc_class_accuracy

from pytorch_metric_learning import losses
loss_func = losses.TripletMarginLoss(margin=0.1)
loss = loss_func(embeddings, labels) # in your training loop



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
    model = getattr(models, args.backbone)()
    if args.cuda:
        model = model.cuda()


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

    interval = -1
    for epoch in range(args.max_epoch):
        for batch_idx, batch_data in enumerate(trainloader):
            interval+=1

            imgs, labels = batch_data

            embeds = model(imgs)

            # Metric Learning
            # ...
            losses = None

            # Print Loss
            if args.print_interval % interval == 0 :
                logging.info(f"[{epoch:%4d}/{args.max_epoch:%4d}] {interval:%7d} Triplet Loss: {losses}")

            # Validation
            if args.val_interval% interval == 0 and interval!=0:
                logging.info(f"[{epoch}/{args.max_epoch}] Starting Validating..")
                for valbatch_idx, valbatch_data in enumerate(valloader):
                    valimgs, vallabels = valbatch_data

                    valembeds = model(valimgs)

                    # Metric Learning
                    # ...
                    vallosses = None

                    # softmax
                    #...
                    cls_acc = calc_class_accuracy()
                    logging.info(f"[{epoch:%4d}/{args.max_epoch:%4d}] {interval:%7d} Triplet Loss: {vallosses} Cls Acc: {cls_acc}")



def get_args():
    parser = argparse.ArgumentParser(description='Train Metric Learning')

    # Train
    parser.add_argument('--max_epoch', default=800, help='epoch to train the network')
    parser.add_argument('--backbone', default="resnet50", help='resnet50 | resnet101 | shufflenetv2 | squeezenet | vgg | xception')
    parser.add_argument('--train_batch_size', default=64, help='training batch size ')
    parser.add_argument('--start_lr', default=0.005, help='base learning rate at the beginning')

    # Validate
    parser.add_argument('--val_batch_size', default=60, help='validating batch size')
    parser.add_argument('--val_interval', default=1000, help='interval of validation')

    # Dataset
    parser.add_argument('--data_format', default="custom", help='coco | custom')
    parser.add_argument('--img_maxsize', default=512, help='the image size')
    parser.add_argument('--train_imgdir', default="", help='the train images path')
    parser.add_argument('--val_imgdir', default="", help='the test images path')
    parser.add_argument('--num_workers', default=4, type=int, help='Number of workers used in dataloading')

    # Misc
    parser.add_argument('--cuda', default=True, type=bool, help='')
    parser.add_argument('--devices', default="0", type=str, help='set as like "0" or "1,2", use comma to split the device ids')
    parser.add_argument('--print_interval', default=100, help='interval of printing the training loss')
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
