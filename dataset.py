from io import BytesIO
import os

from PIL import Image
from torch.utils.data import Dataset


class FFHQ_Dataset(Dataset):
    '''
    Usage:
        Self-coded class for loading the FFHQ data
    '''
    
    def __init__(self, image_folder, transform = None):
        images_list = os.listdir(image_folder)
        self.images_list = sorted([os.path.join(image_folder, image) for image in images_list])
        self.transform = transform
    
    def __getitem__(self, index):
        img_id = self.images_list[index]
        img = Image.open(img_id).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        
        return img

    def __len__(self):
        return len(self.images_list)

