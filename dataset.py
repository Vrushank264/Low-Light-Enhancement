import torch
from torch.utils.data import dataset
import torchvision.transforms as T
from PIL import Image
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

class LoLDataset(dataset.Dataset):
    
    def __init__(self, low_light_root, target_root, img_size = 64):
        
        super().__init__()
        self.lol_fnames = [os.path.join(low_light_root, file) for file in os.listdir(low_light_root)]
        self.target_fnames = [os.path.join(target_root, file) for file in os.listdir(target_root)]
        
        self.transform = T.Compose([T.CenterCrop((img_size, img_size)),
                                    T.ToTensor(),
                                    T.Normalize([0.0,0.0,0.0], [1.0,1.0,1.0])])
        
    def __getitem__(self, idx):
        
        lol = Image.open(self.lol_fnames[idx]).convert('RGB')
        target = Image.open(self.target_fnames[idx]).convert('RGB')
        lol = self.transform(lol)
        target = self.transform(target)
        
        return lol, target
    
    def __len__(self):
        
        return len(self.lol_fnames)
    
