# Origin: KevibMusgrave/pytorch-metric-learning
# Modify: unanan

import argparse

from pytorch_metric_learning import losses
loss_func = losses.TripletMarginLoss(margin=0.1)
loss = loss_func(embeddings, labels) # in your training loop



def train(args):
    pass



def get_args():
    parser = argparse.ArgumentParser(description='parameters to train net')
    parser.add_argument('--max_epoch', default=800, help='epoch to train the network')
    parser.add_argument('--img_size', default=(256, 64), help='the image size')  # 376, 96, #282, 72 #256, 64  #94, 24  96, 24
    parser.add_argument('--train_img_dirs',
                        default="/media/data4T2/meterdataset/digital_dataset/lmdbdataset/train/images",
                        help='the train images path')
    parser.add_argument('--test_img_dirs',
                        default="/media/data4T2/meterdataset/digital_dataset/lmdbdataset/val/images",
                        help='the test images path')
    parser.add_argument('--dropout_rate', default=0.5, help='dropout rate.')
    parser.add_argument('--learning_rate', default=0.005, help='base value of learning rate.')
    parser.add_argument('--lpr_max_len', default=10, help='license plate number max length.')
    parser.add_argument('--train_batch_size', default=64, help='training batch size.')
    parser.add_argument('--test_batch_size', default=60, help='testing batch size.')
    parser.add_argument('--phase_train', default=True, type=bool, help='train or test phase flag.')
    parser.add_argument('--num_workers', default=4, type=int, help='Number of workers used in dataloading')
    parser.add_argument('--cuda', default=True, type=bool, help='Use cuda to train model')
    parser.add_argument('--resume_epoch', default=0, type=int, help='resume iter for retraining')
    parser.add_argument('--save_interval', default=5000, type=int, help='interval for save model state dict')
    parser.add_argument('--test_interval', default=2000, type=int, help='interval for evaluate')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--weight_decay', default=2e-5, type=float, help='Weight decay for SGD')
    parser.add_argument('--lr_schedule', default=[60, 250, 500, 700, 780], help='schedule for learning rate.')
    parser.add_argument('--save_folder', default='weights/', help='Location to save checkpoint models')
    # parser.add_argument('--pretrained_model', default='./weights/Final_LPRNet_model.pth', help='pretrained base model')
    parser.add_argument('--pretrained_model', default='', help='pretrained base model')

    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = get_args()

    train(args)
