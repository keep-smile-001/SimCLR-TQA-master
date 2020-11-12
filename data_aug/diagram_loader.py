# -----------------------------------------------
# !/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time:2020/9/1 18:47
# @Author:
# -----------------------------------------------
import torch.utils.data as Data
import os
import cv2 as cv
from tqdm import tqdm
import torchvision.transforms as transforms
from data_aug.gaussian_blur import GaussianBlur
from PIL import Image


class DiagramDataset(Data.Dataset):
    """
    build diagram dataset like STL, CIFA
    """

    def __init__(self, data_path, splits, input_shape, s):
        self.data_path = data_path
        self.splits = splits
        self.img_size = input_shape
        self.s = s
        self.img_name_to_ix = {}
        self.transform = self._get_simclr_pipeline_transform()
        self.dataset = self._get_tqa_data()

    def __len__(self):
        print('Dataset size: {}'.format(len(self.dataset)))
        return len(self.dataset)

    def __getitem__(self, id):
        return self.dataset[id]

    def _get_simclr_pipeline_transform(self):
        # get a set of data augmentation transformations as described in the SimCLR paper.
        color_jitter = transforms.ColorJitter(0.8 * self.s, 0.8 * self.s, 0.8 * self.s, 0.2 * self.s)
        data_transforms = transforms.Compose([transforms.RandomResizedCrop(size=self.img_size[0]),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.RandomApply([color_jitter], p=0.8),
                                              transforms.RandomGrayscale(p=0.2),
                                              GaussianBlur(kernel_size=int(0.1 * self.img_size[0]) - 1),
                                              transforms.ToTensor()]) # ensure the kernel size % 2 == 1
        return data_transforms

    def _get_img_dir_list(self, dir_path):
        """
        get the file list of tqa like textbook_images/, abc_question_images/
        :param dir_path: img file path
        :return: img dir list
        """
        img_dirs = [name for name in os.listdir(dir_path) if name.endswith('images')]
        img_dirs.sort()
        return img_dirs

    def _get_tqa_data(self):
        """
        load tqa data like torchvision.datasets.STL10.
        :return: dataset
        """

        dataset = []
        print('processing TQA images ...')

        for split in self.splits:
            img_folder_path = os.path.join(self.data_path, split)
            img_folder_list = self._get_img_dir_list(img_folder_path)

            for img_f in img_folder_list:
                imgs = os.listdir(os.path.join(img_folder_path, img_f))
                imgs.sort()
                for img_name in tqdm(imgs):
                    img = Image.open(os.path.join(img_folder_path, img_f, img_name)).convert('RGB') #PGBA -> RGB
                    # img = img.resize((self.img_size[0], self.img_size[0]), Image.BILINEAR)

                    if img_name not in self.img_name_to_ix:
                        self.img_name_to_ix[img_name] = len(self.img_name_to_ix)
                        dataset.append(SimCLRDataTransform(self.transform)(img))
        return dataset

class SimCLRDataTransform(object):
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, sample):
        xi = self.transform(sample)
        xj = self.transform(sample)
        return xi, xj