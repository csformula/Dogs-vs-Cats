import os
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms

class TestDataset(Dataset):
    ''' Subclass of Dataset to hold test images.'''
    
    def __init__(self, fivecrop=False, test_dir='./test/'):
        '''
        Args:
            fivecrop(bool): If true, fivecrop input data.
            test_dir(str): Root directory saving test images.
        '''
        
        super(TestDataset, self).__init__()
        self.fivecrop = fivecrop
        if not self.fivecrop:
            self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.FiveCrop(224),
                transforms.Lambda(
                    lambda crops: torch.stack([
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(
                            transforms.ToTensor()(crop)) 
                        for crop in crops]))                
            ])
        self.test_dir = test_dir
        self.image_names = os.listdir(self.test_dir)
        self.image_dirs = [os.path.join(self.test_dir, name) for name in self.image_names]
    
    def __len__(self):
        return len(self.image_names)
    
    def __getitem__(self, idx):
        ''' return a tuple in which 
            the first entry is name of requested image and second entry is transformed requested image'''
        
        img_dir = self.image_dirs[idx] 
        img = Image.open(img_dir)
        img = img.convert('RGB')
        
        img_tensor = self.transform(img)
        img_name = self.image_names[idx]
        
        return (int(img_name[:img_name.find('.jpg')]), img_tensor)
    
