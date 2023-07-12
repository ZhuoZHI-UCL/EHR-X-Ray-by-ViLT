import os
import numpy as np
from PIL import Image
import pandas as pd 

import torch
from torch.utils.data import Dataset
# import 
import glob
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os
from randaug import RandAugment
img_aug = RandAugment(n=3,m=9)

class MIMICCXR(Dataset):
    def __init__(self, paths, args, transform=None, split='train'):
        self.data_dir = args.cxr_data_dir
        self.args = args
        self.split = split
        self.CLASSES  = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema',
       'Enlarged Cardiomediastinum', 'Fracture', 'Lung Lesion',
       'Lung Opacity', 'No Finding', 'Pleural Effusion', 'Pleural Other',
       'Pneumonia', 'Pneumothorax', 'Support Devices']
        self.filenames_to_path = {path.split('/')[-1].split('.')[0]: path for path in paths}

        metadata = pd.read_csv(f'{self.data_dir}/mimic-cxr-2.0.0-metadata.csv')
        labels = pd.read_csv(f'{self.data_dir}/mimic-cxr-2.0.0-chexpert.csv')
        labels[self.CLASSES] = labels[self.CLASSES].fillna(0)
        labels = labels.replace(-1.0, 0.0)
        
        splits = pd.read_csv(f'{self.data_dir}/mimic-cxr-ehr-split.csv')


        metadata_with_labels = metadata.merge(labels[self.CLASSES+['study_id'] ], how='inner', on='study_id')


        self.filesnames_to_labels = dict(zip(metadata_with_labels['dicom_id'].values, metadata_with_labels[self.CLASSES].values))
        self.filenames_loaded = splits.loc[splits.split==split]['dicom_id'].values
        self.transform = transform
        self.filenames_loaded = [filename  for filename in self.filenames_loaded if filename in self.filesnames_to_labels]

    def __getitem__(self, index):
        if isinstance(index, str):
            #草，这个图像本身就是灰度的，不是rgb的，我还转成灰度图像了
            img = Image.open(self.filenames_to_path[index]).convert('RGB')

            labels = torch.tensor(self.filesnames_to_labels[index]).float()
            #在这里面进行图像增强哈
            if self.split == 'train':
                img = img_aug(img) #使用随机增强
                # print('randaug applied')
            
            img = self.transform(img)
            return img, labels
        
        filename = self.filenames_loaded[index]
        
        # img = Image.open(self.filenames_to_path[filename]).convert('L')
        img = Image.open(self.filenames_to_path[filename]).convert('RGB')

        labels = torch.tensor(self.filesnames_to_labels[filename]).float()

        # if self.transform is not None:
        #     img = self.transform(img)
        if self.split == 'train':
            img = img_aug(img) #使用随机增强
            # print('randaug applied')
        img = self.transform(img)    
        return img, labels
    
    def __len__(self):
        return len(self.filenames_loaded)
    
def get_transforms(args):
    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    train_transforms = []

    # train_transforms.append(transforms.Resize(384))#之前的
    # train_transforms.append(transforms.RandomHorizontalFlip())
    # train_transforms.append(transforms.RandomAffine(degrees=45, scale=(.85, 1.15), shear=0, translate=(0.15, 0.15)))
    # train_transforms.append(transforms.CenterCrop(384))
    # train_transforms.append(transforms.ToTensor())
    # train_transforms.append(normalize)    
    
    train_transforms.append(transforms.ToTensor())
    train_transforms.append(transforms.Resize(384))#之前的
    train_transforms.append(transforms.CenterCrop(384))
    train_transforms.append(normalize)    


    #test和val上面不会加上这些处理，因为我们希望处理更加真实的图像
    test_transforms = []
    test_transforms.append(transforms.ToTensor())
    test_transforms.append(transforms.Resize(384))
    test_transforms.append(transforms.CenterCrop(384))
    test_transforms.append(normalize)


    return train_transforms, test_transforms
def get_cxr_datasets(args):
    train_transforms, test_transforms = get_transforms(args)

    data_dir = args.cxr_data_dir
    
    # filepath = f'{args.cxr_data_dir}/paths.npy'
    # if os.path.exists(filepath):
    #     paths = np.load(filepath)
    # else:
    paths = glob.glob(f'{data_dir}/resized/**/*.jpg', recursive = True)
    # np.save(filepath, paths)
    
    #test_transforms只进行裁剪不进行其他的操作
    dataset_train = MIMICCXR(paths, args, split='train', transform=transforms.Compose(test_transforms))
    dataset_validate = MIMICCXR(paths, args, split='validate', transform=transforms.Compose(test_transforms))
    dataset_test = MIMICCXR(paths, args, split='test', transform=transforms.Compose(test_transforms))

    return dataset_train, dataset_validate, dataset_test

