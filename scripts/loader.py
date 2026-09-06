import os
import pandas as pd
from PIL import Image
from matplotlib import pyplot as plt

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torch.utils.data import Subset
from torchvision import transforms as T
from torchvision.transforms import ToTensor, InterpolationMode
from torchvision.transforms.functional import adjust_contrast, adjust_brightness

SPLIT_FILE = '../data/split_fixed_v2.csv'

# helper class for random distortion
class RandomDistortion(torch.nn.Module):
    def __init__(self, probability=0.25, grid_width=2, grid_height=2, magnitude=8):
        super().__init__() 
        self.probability = probability
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.magnitude = magnitude

    def forward(self, img):
        if torch.rand(1).item() < self.probability:
            return T.functional.affine(img, 0, [0, 0], 1, [self.magnitude, self.magnitude], interpolation=T.InterpolationMode.NEAREST, fill=[0, 0, 0])
        else:
            return img

# helper class for random adjust contrast
class RandomAdjustContrast(torch.nn.Module):
    def __init__(self, probability=.5, min_factor=0.8, max_factor=1.2):
        super().__init__()
        self.probability = probability
        self.min_factor = min_factor
        self.max_factor = max_factor

    def forward(self, img):
        if torch.rand(1).item() < self.probability:
            factor = torch.rand(1).item() * (self.max_factor - self.min_factor) + self.min_factor
            return adjust_contrast(img, factor)
        else:
            return img


def random_adjust_brightness(img):
    factor = torch.rand(1).item() + 0.5
    return adjust_brightness(img, factor)   

# transforms for data augmentation
augmentation_transforms = T.Compose([
    T.RandomRotation(5),
    T.RandomHorizontalFlip(p=0.5),
    RandomDistortion(probability=0.25, grid_width=2, grid_height=2, magnitude=8),
    T.RandomApply([T.ColorJitter(brightness=(0.5, 1.5), contrast=(0.8, 1.2), saturation=(0.8, 1.2), hue=(0.0, 0.1))], p=1),
    RandomAdjustContrast(probability=0.5, min_factor=0.8, max_factor=1.2),
    T.Lambda(random_adjust_brightness)
])

# transforms for vit
vit_transforms = T.Compose([
    T.Resize([518], interpolation=InterpolationMode.BICUBIC),
    T.CenterCrop([518]),
    T.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])



# the original dataset
class BMIDataset(Dataset):
    def __init__(self, csv_path, image_folder, y_col_name, transform=None):
        self.csv = pd.read_csv(csv_path)
        self.image_folder = image_folder

        # Drop the rows where the image does not exist
        images = os.listdir(image_folder)
        self.csv = self.csv[self.csv['name'].isin(images)]
        self.csv.reset_index(drop=True, inplace=True)

        self.y_col_name = y_col_name
        self.transform = transform

    def __len__(self):
        return len(self.csv)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_folder, self.csv.iloc[idx]['name'])

        image = Image.open(image_path)

        # check the channel number
        if image.mode != 'RGB':
            image = image.convert('RGB')

        y = self.csv.loc[idx, self.y_col_name]

        if self.transform:
            image = self.transform(image)

        return image, y



# the augmented dataset
class AugmentedBMIDataset(Dataset):
    def __init__(self, original_dataset, transforms=None):
        self.original_dataset = original_dataset
        self.transforms = transforms

    def __len__(self):
        return 5 * len(self.original_dataset)

    def __getitem__(self, idx):
        image, y = self.original_dataset[idx // 5]

        if self.transforms and (idx % 5 != 0):
            image = self.transforms(image)

        return image, y



# the dataset transformed for vit inputs
class VitTransformedDataset(Dataset):
    def __init__(self, original_dataset, transforms=vit_transforms):
        self.original_dataset = original_dataset
        self.transforms = transforms

    def __len__(self):
        return len(self.original_dataset)

    def __getitem__(self, idx):
        image, y = self.original_dataset[idx]

        if self.transforms:
            image = self.transforms(image)

        return image, y


# show 5 sample images
def show_sample_image(dataset):
    fig, axs = plt.subplots(nrows=1, ncols=5, figsize=(15, 3))
    for i, ax in enumerate(axs):
        image, label = dataset.__getitem__(i)
        ax.imshow(image.detach().cpu().permute(1, 2, 0), cmap='gray')
        ax.set_title('gt BMI: ' + str(label))
        ax.axis('off')  # Hide axes
    plt.show()


# split dataset and (optionally) augment and/ or transform it for vit
def train_val_test_split(dataset, augmented=True, vit_transformed=True):
    _df = pd.read_csv(SPLIT_FILE)

    # dataset ที่รับเข้ามาคือ BMIDataset ดิบ ซึ่งมี attribute .csv 
    # ที่ผ่านการ isin(images) + reset_index มาแล้ว
    dataset_names = dataset.csv['name'].reset_index(drop=True)
    name_to_idx = {name: idx for idx, name in enumerate(dataset_names)}

    def get_indices(split_name):
        names = _df.loc[_df['split'] == split_name, 'name']
        idx_list = [name_to_idx[n] for n in names if n in name_to_idx]
        missing = [n for n in names if n not in name_to_idx]
        if missing:
            print(f'[WARNING] split={split_name}: {len(missing)} ภาพใน split_fixed.csv ไม่พบใน dataset จริง (ถูกกรองออกไปแล้ว)')
        return idx_list

    train_dataset = Subset(dataset, get_indices('train'))
    val_dataset   = Subset(dataset, get_indices('val'))
    test_dataset  = Subset(dataset, get_indices('test'))

    if augmented:
        train_dataset = AugmentedBMIDataset(train_dataset, augmentation_transforms)

    if vit_transformed:
        train_dataset = VitTransformedDataset(train_dataset)
        val_dataset   = VitTransformedDataset(val_dataset)
        test_dataset  = VitTransformedDataset(test_dataset)

    return train_dataset, val_dataset, test_dataset



# get dataloaders
def get_dataloaders(batch_size=16, augmented=True, vit_transformed=True, show_sample=False):
    bmi_dataset = BMIDataset(SPLIT_FILE, '../data/Images', 'bmi', ToTensor())
    if show_sample:
        train_dataset, val_dataset, test_dataset = train_val_test_split(bmi_dataset, augmented, vit_transformed=False)
        show_sample_image(train_dataset)
    train_dataset, val_dataset, test_dataset = train_val_test_split(bmi_dataset, augmented, vit_transformed)

    train_loader = DataLoader(train_dataset, batch_size=batch_size,  shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,  shuffle=False, num_workers=2, pin_memory=True)

    return train_loader, val_loader, test_loader



# for test
# if __name__ == "__main__":
#     get_dataloaders(augmented=False, show_sample=True)
#     get_dataloaders(augmented=True, show_sample=True)