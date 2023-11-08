import xml.etree.ElementTree as ET
import PIL.Image             as pilimg
import torch.utils.data      as data
import numpy                 as np

from torchvision.transforms import v2

import torch
import cv2
import os

VOC_CLASSES = ('background','license_plate','car')

default_transform = v2.Compose([
        v2.ToImage(),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

class VOCDetection(data.Dataset):
    def __init__(self, voc_root, annotation_filename, sample_transform=default_transform):
        self.annotation_filename = annotation_filename
        self.sample_transform    = sample_transform
        self.root                = voc_root  
        
        self.ids = []
        
        with open(self.annotation_filename, 'r') as f:
            for line in f.readlines():
                line = line.strip()
                ano_path = os.path.join(self.root, 'Annotations', line + '.xml')
                img_path = os.path.join(self.root, 'JPEGImages' , line + '.jpg')
                self.ids.append((img_path, ano_path))
    
    def get_annotations(self, path):
        tree = ET.parse(path)

        boxes, labels  = [], []
        for child in tree.getroot():
            if child.tag != 'object':
                continue

            bndbox = child.find('bndbox')
            box =[
                float(bndbox.find(t).text) - 1
                for t in ['xmin', 'ymin', 'xmax', 'ymax']
            ]
            label = VOC_CLASSES.index(child.find('name').text) 
            
            labels.append(label)
            boxes .append(box  )

        return np.array(boxes), np.array(labels)

    def __getitem__(self, index):
        img_path, ano_path = self.ids[index]

        image         = cv2.imread(img_path, cv2.IMREAD_COLOR)
        boxes, labels = self.get_annotations(ano_path)
        
        image = image.astype(np.float32) / 255.
        image = self.sample_transform(image)
        
        _, height, width = image.shape
        boxes[:, 0::2] /=  width
        boxes[:, 1::2] /= height
        
        return image, torch.from_numpy(boxes), torch.from_numpy(labels)
    
    def __len__(self):
        return len(self.ids)
 
if __name__ == '__main__':
    voc_root = "dataset"
    
    train_annotation_filename = os.path.join( voc_root, "ImageSets/Main/trainval.txt" )
    test_annotation_filename  = os.path.join( voc_root, "ImageSets/Main/test.txt"     )
    
    train_dataset = VOCDetection( voc_root, train_annotation_filename )
    
    for train_sample in train_dataset:
        print(train_sample)
    
    test_dataset  = VOCDetection( voc_root, test_annotation_filename  )

    for test_sample in test_dataset:
        print(test_sample)

