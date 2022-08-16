from torchvision import transforms


def image_transformatiom() -> dict:
    """
    :return: Transformation dictionary for train, valid and test data.
    """
    data_transforms = {'train': transforms.Compose([transforms.Resize(256),
                                                    transforms.CenterCrop(224),
                                                    transforms.RandomHorizontalFlip(),
                                                    transforms.RandomVerticalFlip(),
                                                    transforms.ColorJitter(brightness=0.2, contrast=0.2,
                                                                           saturation=0.1, hue=0.1),
                                                    transforms.RandomAffine(degrees=40, translate=None, scale=(1, 2),
                                                                            shear=15, resample=False, fillcolor=0),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                         std=[0.229, 0.224, 0.225])]),

                       'valid': transforms.Compose([transforms.Resize(256),
                                                    transforms.CenterCrop(224),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                         std=[0.229, 0.224, 0.225])]),

                       'test': transforms.Compose([transforms.Resize(256),
                                                   transforms.CenterCrop(224),
                                                   transforms.ToTensor(),
                                                   transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                        std=[0.229, 0.224, 0.225])])}

    return data_transforms