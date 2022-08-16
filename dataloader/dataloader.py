import os
import warnings
from dataloader.dataset import FLDataset
from scripts.utils import get_images_paths
from torch.utils.data import DataLoader
from dataloader.transforms import image_transformatiom
warnings.filterwarnings("ignore")

def get_dataloader(args, idx_to_class: dict) -> dict:
    """
    :param args: return the value of predefined parameters
    :param idx_to_class: dictionary of correspondence between the number of the class and its writing
    :return: dict of train/valid data loaders
    """
    # GET TRAIN/VALID PATHS
    data_folder = args.data_path
    data_folder_train = os.path.join(data_folder, 'train')
    data_folder_valid = os.path.join(data_folder, 'valid')

    # FUNCTION RETURNS LIST OF IMAGES PATHS
    train_image_paths = get_images_paths(data_folder_train)
    valid_image_paths = get_images_paths(data_folder_valid)

    # LOAD IMAGE TRANSFORMER
    data_transforms = image_transformatiom()

    # INITIALIZE TRAIN/VALID DATASETS
    train_dataset = FLDataset(train_image_paths, 'train', data_transforms,  idx_to_class)
    valid_dataset = FLDataset(valid_image_paths, 'valid', data_transforms,  idx_to_class)

    print(f'TRAIN DATASET LENGTH - {train_dataset.__len__()}')
    print(f'VALID DATASET LENGTH - {valid_dataset.__len__()}')
    print(40*'--')

    # INITIALIZE TRAIN/VALID DATA LOADERS
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    print("TRAIN LOADER LENGTH - %r" % (len(train_loader)))
    print("VALID LOADER LENGTH - %r" % (len(valid_loader)))
    print(40 * '--')

    # INITIALIZE DATA LOADERS DICT
    dataloaders = {'train': train_loader,
                   'valid': valid_loader
                   }

    return dataloaders