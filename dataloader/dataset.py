import torch
from PIL import Image
from torch.utils.data.dataset import Dataset


class FLDataset(torch.utils.data.Dataset):
    """
    ------------------------------------------
    CUSTOM DATASET FOR FLOWER CLASSIFICATOR
    :return image and label
    ------------------------------------------
    """
    def __init__(self, image_paths, flag, transform, idx_to_class):
        self.image_paths = image_paths
        self.flag = flag
        self.transform = transform
        self.idx_to_class = idx_to_class

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        data_dict = {}
        image_filepath = self.image_paths[idx]
        label = int(self.image_paths[idx].split('\\')[-2]) - 1
        image = Image.open(image_filepath).convert('RGB')
        if self.flag == 'train':
            image = self.transform['train'](image)
        elif self.flag == 'valid':
            image = self.transform['valid'](image)
        else:
            print('Choose train or valid')

        data_dict['image'] = image
        data_dict['label'] = label

        return data_dict