from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os


class SaliconDataset(Dataset):
    def __init__(self, root_dir, source,  transform=None):
        self.__root_dir = root_dir
        self.__image_source = source
        self.__images_path = "images"
        self.__transforms = transform
        self.__images_files = os.listdir(os.path.join(root_dir, self.__images_path, self.__image_source))      
  

    def __len__(self):
        return len(self.__images_files)

    
    def __getitem__(self, idx):
        image_file = self.__images_files[idx]
        
        img = Image.open(os.path.join(self.__root_dir, self.__images_path, self.__image_source, image_file)).convert('RGB')
    
        if self.__transforms is not None:
            img = self.__transforms(img)
            
        return img