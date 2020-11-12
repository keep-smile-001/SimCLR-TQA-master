#-----------------------------------------------
#!/usr/bin/env python
# _*_ coding: utf-8 _*_
#@Time:2020/9/8 21:41
#@Author:
#-----------------------------------------------

import os
import yaml
import torch
import torchvision.transforms as transforms
import torch.utils.data as Data
import importlib.util
import numpy as np
from tqdm import tqdm
from PIL import Image

def obtain_img_representations():
    folder_name = '/data/kf/majie/codehub/simclr/runs/Sep08_21-33-04_lthpc'
    checkpoints_folder = os.path.join(folder_name, 'checkpoints')
    config = yaml.load(open(os.path.join(checkpoints_folder, 'config.yaml'), 'r'), Loader=yaml.FullLoader)
    print(config)

    device = 'cuda: {}'.format(config['gpu']) if torch.cuda.is_available() else 'cpu'
    print('Using device {}'.format(device))

    dataset = DiagramDataset(config['dataset']['data_path'],
                                 config['dataset']['splits'],
                                 (224, 224, 3))
    dataloader = Data.DataLoader(
        dataset=dataset,
        batch_size=config['batch_size'],
        num_workers=config['dataset']['num_workers'],
        shuffle=False,
        drop_last=False
    )

    model = load_resnet_model(checkpoints_folder, config, device)
    ix = 0
    for img_iter in dataloader:
        img_iter = img_iter.to(device)
        features, _ = model(img_iter)
        feat_array = features.cpu().detach().numpy()

        for i in range(feat_array.shape[0]):
            np.savez(file=os.path.join('/data/kf/majie/codehub/simclr/data/tqa_diagram/', dataset.img_name_to_ix[ix]))
            ix = ix + 1
    assert ix == dataset.__len__(), 'error: data saving.'

def load_resnet_model(checkpoints_folder, config, device):
  # Load the neural net module
  spec = importlib.util.spec_from_file_location("model", os.path.join(checkpoints_folder, 'resnet_simclr.py'))
  resnet_module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(resnet_module)

  model = resnet_module.ResNetSimCLR('resnet50', 128)
  model.eval()

  state_dict = torch.load(os.path.join(checkpoints_folder, 'model.pth'), map_location=torch.device('cpu'))
  model.load_state_dict(state_dict)
  model = model.to(device)
  return model

class DiagramDataset(Data.Dataset):
    """
    build diagram dataset like STL, CIFA
    """

    def __init__(self, data_path, splits, input_shape):
        self.data_path = data_path
        self.splits = splits
        self.img_size = input_shape
        self.img_name_to_ix = {}
        self.transforms = transforms.ToTensor()
        self.dataset = self._get_tqa_data()

    def __len__(self):
        print('Dataset size: {}'.format(len(self.dataset)))
        return len(self.dataset)

    def __getitem__(self, id):
        return self.transforms(self.dataset[id])

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
        load tqa images.
        :return: dataset
        """

        dataset = []
        counter = 0
        print('processing TQA images ...')

        for split in self.splits:
            img_folder_path = os.path.join(self.data_path, split)
            img_folder_list = self._get_img_dir_list(img_folder_path)

            for img_f in img_folder_list:
                imgs = os.listdir(os.path.join(img_folder_path, img_f))
                imgs.sort()
                for img_name in tqdm(imgs):
                    img = Image.open(os.path.join(img_folder_path, img_f, img_name)).convert('RGB') #PGBA -> RGB
                    img = img.resize((self.img_size[0], self.img_size[0]), Image.BILINEAR)

                    self.img_name_to_ix[counter] = img_name
                    counter += 1
                    dataset.append(img)
        return dataset

if __name__ == '__main__':
    obtain_img_representations()
