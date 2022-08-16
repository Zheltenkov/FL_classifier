import os
import json
import torch
from torch import nn
from PIL import Image
import pretrainedmodels
from torch.autograd import Variable
from config.config import parse_args
from dataloader.transforms import image_transformatiom

def check_paths(args):
    ''' Check (create if they don't exist) experiment directories.
    :param args: Runtime arguments as passed by the user.
    :return: List containing result_dir_path, model_dir_path, train_log_dir_path, val_log_dir_path.
    '''
    folders = [args.result_dir_name, args.model_dir_name]
    paths = list()
    for folder in folders:
        folder_path = os.path.join(args.experiment_path, folder)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        paths.append(folder_path)

    log_folders = ['train', 'valid']
    for folder in log_folders:
        folder_path = os.path.join(args.experiment_path, args.log_dir_name, folder)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        paths.append(folder_path)
    return paths


def get_images_paths(data_path: str) -> list:
    """
    :param data_path: The path for storing image classes. Must correspond to the structure dir -> class -> image.
    :return: List of paths of all images in the directory.
    """
    try:
        images_paths = [os.path.join(*[data_path, class_, path]) for class_ in os.listdir(data_path)
                        for path in os.listdir(os.path.join(data_path, class_))]
        return images_paths
    except NotADirectoryError:
        print(f'For {data_path} сheck the data storage structure')


def prepare_model(model_name, num_classes, model_path):
    """
    :param model_path: Path to save model
    :param model_name: Model architecture
    :param num_classes: Number of classes in model
    :return:
    """
    model_args = {}
    model = getattr(pretrainedmodels.models, model_name)(**model_args)
    model.last_linear = nn.Linear(in_features=model.last_linear.in_features, out_features=num_classes)
    model = model.cpu()

    torch.save(model, os.path.join(model_path, f'{model_name}_model' + '.pth'))
    return model


def prepare_image(image_path: str):
    """
    :param image_path: Path to image directory
    :return:
    """
    data_transforms = image_transformatiom()

    img = Image.open(image_path).convert('RGB')
    img = data_transforms['test'](img)
    image = img.expand(1, 3, 224, 224)
    return image


