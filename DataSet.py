import torchvision
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import SimpleITK as sitk
from skimage.io import imread
from torch.utils.data import Dataset, DataLoader, random_split
from Transforms_v2 import *
import warnings
import random

warnings.filterwarnings("ignore")


def load_sitk(path):
    return sitk.GetArrayFromImage(sitk.ReadImage(path))


class IDRiDDataset(Dataset):
    def __init__(self, mode='train', root_dir='./idrid',
                 transform=None, tasks=None, data_augmentation=True):

        super(IDRiDDataset, self).__init__()
        # After resize image
        if tasks is None:
            tasks = ['EX', 'HE', 'MA', 'SE']

        if mode == 'train':
            mask_file, image_file = './train_masks/', './train_images/'

        elif mode == 'val':
            mask_file, image_file = './test_masks/', './test_images/'

        else:
            raise EnvironmentError('You should put a valid mode to generate the dataset')

        self.mode = mode
        self.transform = transform
        self.mask_file = mask_file
        self.image_file = image_file
        self.root_dir = root_dir
        self.tasks = tasks
        self.data_augmentation = data_augmentation
        self.name_list = self.get_list()
        self.process_image = False

    def __len__(self):
        task = self.tasks[0]  # Assuming all the masks folders have the same length
        mask_path = os.path.join(self.root_dir, self.mask_file + task)
        return len(self.name_list)

    def __getitem__(self, idx):
        """Generate one batch of data"""
        sample = self.load(idx)
        return sample

    def get_list(self):
        if self.mode == 'val':
            return pd.read_csv(os.path.join(self.root_dir, 'EX_test.csv'), header=None).iloc[:, 0].values
        return pd.read_csv(os.path.join(self.root_dir, 'EX_train.csv'), header=None).iloc[:, 0].values

    def load(self, idx):
        # Get masks from a particular idx
        masks = [0]
        # 读取每个类的mask 0-BG(背景) 1-EX 2-HE 3-MA 4-SE
        bg = np.ones((2848, 4288))
        for task in self.tasks:
            suffix = '.tif'
            mask_name = 'IDRiD_{:02d}_'.format(self.name_list[idx]) + task + suffix
            mask_path = os.path.join(self.root_dir, self.mask_file + task + '/' + mask_name)
            if os.path.exists(mask_path):
                mask = load_sitk(mask_path)
                mask = mask[:, :, 0] / 255
            else:
                mask = np.zeros((2848, 4288))
            bg[mask == 1] = 0
            masks.append(mask)
        masks[0] = bg
        masks = np.stack(masks, axis=0)
        # Get original images
        image_name = 'IDRiD_{:02d}'.format(self.name_list[idx]) + '.jpg'
        image_path = os.path.join(self.root_dir, self.image_file + image_name)
        image = load_sitk(image_path)
        masks = masks.astype(np.int16)  # Define output sample
        # 去黑边5.2添加 135 136行
        sample = {'image': image, 'masks': masks}
        # If transform apply transformation
        if self.transform:
            sample = self.transform(sample)
        return sample


def load_train_val_data(tasks=None, data_path='./data/idrid/', batch_size=8, green=False):
    # The data is store in the folder 'idrid/'
    if tasks is None:
        tasks = ['EX', 'HE', 'MA', 'SE']
    transforms_train = [
        # CenterCrop((2848, 3500)),

        # Resize(700),  # resize to 520x782
        # RandomCrop((640, 786)),

        # Resize(1030),  # resize to 520x782
        # RandomCrop(1024),

        Resize(656),  # resize to 520x782
        RandomCrop(640),

        RandomRotate90(),
        RandomHorizontalFlip(flip_prob=random.random()),
        RandomVerticalFlip(flip_prob=random.random()),
        ApplyCLAHE(green=green),
        ToTensor(green=green)
    ]
    transforms_val = [
        Resize(642),  # resize to 520x782
        CenterCrop(640),
        ApplyCLAHE(green=green),
        ToTensor(green=green)
    ]

    transformation_train = torchvision.transforms.Compose(transforms_train)
    transformation_val = torchvision.transforms.Compose(transforms_val)

    print('Loading Train and Validation Datasets... \n')
    print("Whole Image Train Mode")
    train_data = IDRiDDataset(mode='train', transform=transformation_train, root_dir=data_path)
    val_data = IDRiDDataset(mode='val', transform=transformation_val, root_dir=data_path)
    train_loader = DataLoader(train_data,
                              batch_size=batch_size,
                              shuffle=True,
                              drop_last=False,
                              num_workers=2,
                              pin_memory=False)


    val_loader = DataLoader(val_data,
                                batch_size=batch_size,
                                shuffle=False,
                                drop_last=False,
                                num_workers=2,
                                pin_memory=False)

    print('Length of train dataset: ', len(train_loader.dataset))
    print('Length of val dataset: ', len(val_loader.dataset))
    print('Shape of image :', train_loader.dataset[10]['image'].shape)
    print('Shape of mask : ', train_loader.dataset[10]['masks'].shape)
    print('-' * 20)
    print('\n')
    return train_loader, val_loader


class DDRDataSet(Dataset):
    def __init__(self, mode='train', root_dir='../CAUNet/data/ddr/',
                 transform=None, data_augmentation=True):

        super(DDRDataSet, self).__init__()
        # After resize image
        if mode == 'train':
            image_file, mask_file = 'train/image/', 'train/label/'

        elif mode == 'val':
            image_file, mask_file = 'valid/image/', 'valid/label/'

        elif mode == 'test':
            image_file, mask_file = 'test/image/', 'test/label/'

        self.mode = mode
        self.tasks = ['EX', 'HE', 'MA', 'SE']
        self.transform = transform
        self.mask_file = mask_file
        self.image_file = image_file
        self.root_dir = root_dir
        self.data_augmentation = data_augmentation
        self.name_list = self.get_list()

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, idx):
        """Generate one batch of data"""
        sample = self.load(idx)
        return sample

    def get_list(self):
        return os.listdir(os.path.join(self.root_dir, self.image_file))

    def load(self, idx):
        # Get masks from a particular idx
        masks = [0]

        # load image
        image_name = self.name_list[idx]
        image_path = os.path.join(self.root_dir, self.image_file + image_name)
        image = load_sitk(image_path)
        # load masks
        # 读取每个类的mask 0-BG(背景) 1-EX 2-HE 3-MA 4-SE
        bg = np.ones((image.shape[0], image.shape[1]))
        for task in self.tasks:
            suffix = '.tif'
            mask_name = self.name_list[idx][:-4] + suffix
            mask_path = os.path.join(self.root_dir, self.mask_file, task, mask_name)
            mask = load_sitk(mask_path)
            mask = mask / 255
            bg[mask == 1] = 0
            masks.append(mask)
        masks[0] = bg
        masks = np.stack(masks, axis=0)
        masks = masks.astype(np.int16)  # Define output sample
        # 去黑边5.2添加 135 136行
        sample = {'image': image, 'masks': masks}
        # If transform apply transformation
        if self.transform:
            sample = self.transform(sample)
        return sample


def load_ddr_train_val(data_path='../CAUNet/data/ddr/', batch_size=8):
    transforms_train = [
        Resize(656),
        RandomCrop(640),
        RandomRotate90(),
        RandomHorizontalFlip(flip_prob=random.random()),
        RandomVerticalFlip(flip_prob=random.random()),
        ApplyCLAHE(green=False),
        ToTensor(green=False)
    ]
    transforms_val = [
        Resize(642),  # resize to 520x782
        CenterCrop(640),
        ApplyCLAHE(green=False),
        ToTensor(green=False)
    ]
    transformation_train = torchvision.transforms.Compose(transforms_train)
    transformation_val = torchvision.transforms.Compose(transforms_val)

    print('Loading Train and Validation Datasets... \n')
    print("Whole Image Train Mode")
    train_data = DDRDataSet(mode='train', transform=transformation_train, root_dir=data_path)
    val_data = DDRDataSet(mode='val', transform=transformation_val, root_dir=data_path)
    train_loader = DataLoader(train_data,
                              batch_size=batch_size,
                              shuffle=True,
                              drop_last=False,
                              num_workers=2,
                              pin_memory=False)
    val_loader = DataLoader(val_data,
                            batch_size=batch_size,
                            shuffle=False,
                            drop_last=False,
                            num_workers=2,
                            pin_memory=False)

    print('Length of train dataset: ', len(train_loader.dataset))
    print('Length of val dataset: ', len(val_loader.dataset))
    print('Shape of image :', train_loader.dataset[10]['image'].shape)
    print('Shape of mask : ', train_loader.dataset[10]['masks'].shape)
    print('-' * 20)
    print('\n')
    return train_loader, val_loader


if __name__ == '__main__':
    train_loader, val_loader = load_ddr_train_val(batch_size=2)
