from torch.utils.data import Dataset
import os
import pandas as pd
import numpy as np
from PIL import Image


class TinyImageData(Dataset):

    def __init__(self, root_path, stage, transform=None, select_train=None):

        self.stage = stage
        self.path = os.path.join(root_path, stage)

        self.transform = transform

        self.labelsNotations = pd.read_csv(
            os.path.join(root_path, 'wnids.txt'), 
            header=None).reset_index().set_index(0).to_dict()['index']
        
        if self.stage == 'train':

            if select_train:
                self.images = np.array([
                os.listdir(os.path.join(self.path, image_path, 'images')) 
                for image_path in select_train if os.path.exists(os.path.join(self.path, image_path, 'images'))
                ]).flatten()

            else:
                self.images = np.array([
                os.listdir(os.path.join(self.path, image_path, 'images')) 
                for image_path in os.listdir(self.path) if os.path.exists(os.path.join(self.path, image_path, 'images'))
                ]).flatten()

        elif self.stage == 'val':
            self.images = pd.read_csv(
                os.path.join(self.path, 'val_annotations.txt'),
                header=None,
                sep='\t',
                usecols = [0, 1]
                ).values
        else:
            self.images = sorted(os.listdir("./data/ml3-aim-2023-hw2/hw2/tiny-imagenet-200/test/images"), 
                                 key=lambda x: int(x.split('_')[1].split('.')[0]))

    def __getitem__(self, index):
        
        if self.stage == 'train':
            image_name = self.images[index]
            image_folder = image_name.split('_')[0]
            image_path = os.path.join(self.path, image_folder, 'images', image_name) 

            label  = self.labelsNotations[image_folder]

            image = Image.open(image_path)



        elif self.stage == 'val':  
            image_name, label = self.images[index]
            image_path = os.path.join(self.path, 'images', image_name) 

            label  = self.labelsNotations[label]
            image = Image.open(image_path)

        else:
            image_name = self.images[index]
            image_path = os.path.join(self.path, 'images', image_name)   
            image = Image.open(image_path) 




        if self.transform:
            image = self.transform(image)

        if self.stage != 'test':
            return image, label
        else:
            return image, image_name
    
    def __len__(self):
        return len(self.images)