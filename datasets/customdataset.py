import os
import os.path as osp
import sys
import torch
import torch.utils.data as data
import torch.nn.functional as F
import cv2
import numpy as np

transforms= {
    "train": {},
    "val":{},
    "test":{}
}

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

    def __init__(self, image_path, info_file, transform=None,
                 target_transform=None):
        with open(info_file)
        self.ids

    def __getitem__(self, index):
        raise NotImplementedError()

    def __len__(self):
        return len(self.ids)

    def pull_item(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: Tuple (image, target, masks, height, width, crowd).
                   target is the object returned by ``coco.loadAnns``.
            Note that if no crowd annotations exist, crowd will be None
        """
        img_id = self.ids[index]

        if self.has_gt:
            ann_ids = self.coco.getAnnIds(imgIds=img_id)

            # Target has {'segmentation', 'area', iscrowd', 'image_id', 'bbox', 'category_id'}
            target = [x for x in self.coco.loadAnns(ann_ids) if x['image_id'] == img_id]
        else:
            target = []

        # Separate out crowd annotations. These are annotations that signify a large crowd of
        # objects of said class, where there is no annotation for each individual object. Both
        # during testing and training, consider these crowds as neutral.
        crowd = [x for x in target if ('iscrowd' in x and x['iscrowd'])]
        target = [x for x in target if not ('iscrowd' in x and x['iscrowd'])]
        num_crowds = len(crowd)

        for x in crowd:
            x['category_id'] = -1

        # This is so we ensure that all crowd annotations are at the end of the array
        target += crowd

        # The split here is to have compatibility with both COCO2014 and 2017 annotations.
        # In 2014, images have the pattern COCO_{train/val}2014_%012d.jpg, while in 2017 it's %012d.jpg.
        # Our script downloads the images as %012d.jpg so convert accordingly.
        file_name = self.coco.loadImgs(img_id)[0]['file_name']

        if file_name.startswith('COCO'):
            file_name = file_name.split('_')[-1]

        path = osp.join(self.root, file_name)
        assert osp.exists(path), 'Image path does not exist: {}'.format(path)

        img = cv2.imread(path)
        height, width, _ = img.shape

        if len(target) > 0:
            # Pool all the masks for this image into one [num_objects,height,width] matrix
            masks = [self.coco.annToMask(obj).reshape(-1) for obj in target]
            masks = np.vstack(masks)
            masks = masks.reshape(-1, height, width)

        if self.target_transform is not None and len(target) > 0:
            target = self.target_transform(target, width, height)

        if self.transform is not None:
            if len(target) > 0:
                target = np.array(target)
                img, masks, boxes, labels = self.transform(img, masks, target[:, :4],
                                                           {'num_crowds': num_crowds, 'labels': target[:, 4]})

                # I stored num_crowds in labels so I didn't have to modify the entirety of augmentations
                num_crowds = labels['num_crowds']
                labels = labels['labels']

                target = np.hstack((boxes, np.expand_dims(labels, axis=1)))
            else:
                img, _, _, _ = self.transform(img, np.zeros((1, height, width), dtype=np.float),
                                              np.array([[0, 0, 1, 1]]),
                                              {'num_crowds': 0, 'labels': np.array([0])})
                masks = None
                target = None

        if target.shape[0] == 0:
            print('Warning: Augmentation output an example with no ground truth. Resampling...')
            return self.pull_item(random.randint(0, len(self.ids) - 1))

        return torch.from_numpy(img).permute(2, 0, 1), target, masks, height, width, num_crowds

    def pull_image(self, index):
        '''Returns the original image object at index in PIL form

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to show
        Return:
            cv2 img
        '''
        img_id = self.ids[index]
        path = self.coco.loadImgs(img_id)[0]['file_name']
        return cv2.imread(osp.join(self.root, path), cv2.IMREAD_COLOR)

    def pull_anno(self, index):
        '''Returns the original annotation of image at index

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to get annotation of
        Return:
            list:  [img_id, [(label, bbox coords),...]]
                eg: ('001718', [('dog', (96, 13, 438, 332))])
        '''
        img_id = self.ids[index]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        return self.coco.loadAnns(ann_ids)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str


class CustomDataset(BaseDataset):
    def __init__(self,image_path, info_file, transform=None,target_transform=None):
        super(CustomDataset, self).__init__(image_path, info_file, transform, target_transform)
