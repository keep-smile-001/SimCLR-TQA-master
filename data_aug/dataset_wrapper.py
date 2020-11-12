import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision.transforms as transforms
from data_aug.gaussian_blur import GaussianBlur
from torchvision import datasets
from data_aug.diagram_loader import DiagramDataset

np.random.seed(0)


class DataSetWrapper(object):

    def __init__(self, batch_size, num_workers, valid_size, input_shape, s, data_path, splits):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.valid_size = valid_size
        self.s = s
        self.input_shape = eval(input_shape)
        self.data_path = data_path
        self.splits = splits

    def get_data_loaders(self):
        # data_augment = self._get_simclr_pipeline_transform()

        # train_dataset = datasets.STL10('./data', split='train+unlabeled', download=True,
        #                                transform=SimCLRDataTransform(data_augment))
        train_dataset = DiagramDataset(self.data_path, self.splits, self.input_shape, self.s)

        train_loader, valid_loader, train_datasize, valid_datasize = self.get_train_validation_data_loaders(
            train_dataset)

        return train_loader, valid_loader, train_datasize, valid_datasize

    def get_train_validation_data_loaders(self, train_dataset):
        # obtain training indices that will be used for validation
        num_train = len(train_dataset)
        indices = list(range(num_train))
        np.random.shuffle(indices)

        split = int(np.floor(self.valid_size * num_train))
        train_idx, valid_idx = indices[split:], indices[:split]

        # define samplers for obtaining training and validation batches
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, sampler=train_sampler,
                                  num_workers=self.num_workers, drop_last=True, shuffle=False)
        train_datasize = train_sampler.__len__()

        valid_loader = DataLoader(train_dataset, batch_size=self.batch_size, sampler=valid_sampler,
                                  num_workers=self.num_workers, drop_last=True)
        valid_datasize = valid_sampler.__len__()

        return train_loader, valid_loader, train_datasize, valid_datasize

    # def _get_simclr_pipeline_transform(self):
    #     # get a set of data augmentation transformations as described in the SimCLR paper.
    #     color_jitter = transforms.ColorJitter(0.8 * self.s, 0.8 * self.s, 0.8 * self.s, 0.2 * self.s)
    #     data_transforms = transforms.Compose([transforms.RandomResizedCrop(size=self.input_shape[0]),
    #                                           transforms.RandomHorizontalFlip(),
    #                                           transforms.RandomApply([color_jitter], p=0.8),
    #                                           transforms.RandomGrayscale(p=0.2),
    #                                           GaussianBlur(kernel_size=int(0.1 * self.input_shape[0]) - 1),
    #                                           transforms.ToTensor()]) # modify the kernel size
    #     return data_transforms

# class SimCLRDataTransform(object):
#     def __init__(self, transform):
#         self.transform = transform
#
#     def __call__(self, sample):
#         xi = self.transform(sample)
#         xj = self.transform(sample)
#         return xi, xj
