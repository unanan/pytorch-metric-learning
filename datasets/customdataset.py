import os
import os.path as osp
import sys
import torch
import torch.utils.data as data
import torch.nn.functional as F
import cv2
from PIL import Image
import numpy as np
import datasets.transforms

transforms= {
    "train": datasets.transforms.Compose([
        datasets.transforms.RandomHorizontalFlip(p=0.5),
        datasets.transforms.RandomResizedCrop(224, (0.8,1.0)),

        datasets.transforms.ColorJitter(brightness=(0.65, 1.35), contrast=(0.5, 1.5)),
        datasets.transforms.RandomChoice([
            datasets.transforms.ColorJitter(saturation=(0, 2), hue=0.3),
            datasets.transforms.HEDJitter(theta=0.05)]),
        datasets.transforms.ToTensor(),
        datasets.transforms.Normalize([0.6270, 0.5013, 0.7519], [0.1627, 0.1682, 0.0977])
    ]),
    "val":datasets.transforms.Compose([
        datasets.transforms.Resize(224),
        datasets.transforms.ToTensor(),
        datasets.transforms.Normalize([0.6270, 0.5013, 0.7519], [0.1627, 0.1682, 0.0977])
    ]),
    "test":datasets.transforms.Compose([
        datasets.transforms.Resize(224),
        datasets.transforms.ToTensor(),
        datasets.transforms.Normalize([0.6270, 0.5013, 0.7519], [0.1627, 0.1682, 0.0977])
    ]),
}


def read_info_file(info_file):
    '''
    line: /path/to/img.jpg;452,462,33,55,0;356,742,34,67,2
    '''
    imgpaths = {};box_labels= {};labels=[]
    with open(info_file) as f:
        lines = f.readlines()
        box_id=0
        for img_id, line in enumerate(lines):
            imgpath_, box_label_ = line.split(";")[0],line.split(";")[1:]
            imgpaths[img_id] = imgpath_

            for bl in box_label_:
                try:
                    x,y,w,h,label = list(map(int, bl.split(",")))
                except:
                    continue
                box_labels[box_id]= {"x":x,"y":y,"w":w,"h":h,"label":label,"imgid":img_id}
                box_id+=1

                if label not in labels:
                    labels.append(label)

    return imgpaths, box_labels, labels


class BaseDataset(data.Dataset):
    """`MS Coco Detection <http://mscoco.org/dataset/#detections-challenge2016>`_ Dataset.
    Args:
        root (string): Root directory where images are downloaded to.
        set_name (string): Name of the specific set of COCO images.
        transform (callable, optional): A function/transform that augments the
                                        raw images`
        target_transform (callable, optional): A function/transform that takes
        in the target (bbox) and transforms it.
        prep_crowds (bool): Whether or not to prepare crowds for the evaluation step.
    """

    def __init__(self, image_path, info_file, num_class=12, transform=None, background_ratio=0.0):
        self.imgpaths, self.box_labels, self.labels = read_info_file(info_file)
        self.ids = self.box_labels.keys()
        self.background_ratio = background_ratio
        self.transform = transform
        self.num_class = num_class

    def __getitem__(self, index):
        raise NotImplementedError()

    def __len__(self):
        return int(len(self.ids)*(1+self.background_ratio))



class CustomDataset(BaseDataset):
    def __init__(self, image_path, info_file, num_class=12, transform=None, background_ratio=0.0):
        super(CustomDataset, self).__init__(image_path, info_file, num_class=num_class, transform=transform, background_ratio=background_ratio)
        self._BACKGROUND_ = None
        if self.background_ratio>0:
            self._BACKGROUND_ = sorted(self.labels)[-1]+1
            self.labels.append(self._BACKGROUND_)

    def __getitem__(self, index):
        imgpath_ = self.imgpaths[self.box_labels[index]["imgid"]]
        pil_img = Image.open(imgpath_)
        width, height = pil_img.size

        left_ = self.box_labels[index]["x"]
        top_ = self.box_labels[index]["y"]
        right_ = width-1-self.box_labels[index]["x"]-self.box_labels[index]["w"]
        bottom_ = height-1-self.box_labels[index]["y"]-self.box_labels[index]["h"]
        pil_img = pil_img.crop((left_,top_,right_,bottom_))

        img_tensor = self.transform(pil_img)


        label = self.box_labels[index]["label"]
        label_tensor = torch.IntTensor(label)

        return img_tensor, label_tensor