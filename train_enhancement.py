import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.utils as vutils
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from dataset import LoLDataset
from model import MIRNet
from utils import *


class Loss_fn(nn.Module):
    
    def __init__(self, eps = 1e-3):
        
        super().__init__()
        self.eps = eps
        
    def forward(self, ip, target):
        
        diff = ip - target
        loss = torch.mean(torch.sqrt((diff * diff) + (self.eps * self.eps)))
        return loss


def train(loader, model, criterion, opt, writer, step, epoch, mixup = True, device = torch.device('cuda')):
    
    train_loss, train_psnr = [], []
    epoch_psnr, epoch_loss = 0.0, 0.0
    loop = tqdm(loader, position = 0, leave = True)
    model.train()
    
    for idx, (lol, target) in enumerate(loop):
        
        lol = lol.to(device)
        target = target.to(device)
        
        if epoch > 5 and mixup is True:
            lol, target = mixup.augment(lol, target)
            
        opt.zero_grad()
        
        gen = model(lol)
        loss = criterion(gen, target)
        train_loss.append(loss.item())
        train_psnr.append(utils.batch_psnr(gen, target, 1.0))
        loss.backward()
        opt.step()
        
        epoch_psnr = sum(train_psnr)/len(train_psnr)
        epoch_loss = sum(train_loss)/len(train_loss)
        model.eval()
        
        writer.add_scalar("Training/Loss", epoch_loss, global_step = step)
        writer.add_scalar("Training/PSNR", epoch_psnr, global_step = step)
        
        if idx % 25:
          gen_img = model(fixed_ip_train)
          writer.add_image("Observations/Train_img", gen_img.squeeze(0), global_step = step)
      
        torch.cuda.empty_cache()
        step += 1
    
    print(f'Epoch: {epoch}, Loss: {epoch_loss} and PSNR: {epoch_psnr}. \n')
    return step
    

def validate(loader, model, criterion, writer, step, epoch, device = torch.device('cuda')):
    
    val_loss, val_psnr = [], []
    loss, psnr = 0.0, 0.0
    loop = tqdm(loader, position = 0, leave = True)
    model.eval()
    print("Validating...")
    for idx, (lol, target) in enumerate(loop):
        
        lol = lol.to(device)
        target = target.to(device)
        
        gen = model(lol)
        loss = criterion(gen, target)
        val_loss.append(loss.item())
        val_psnr.append(utils.batch_psnr(gen, target, 1.0))
        step += 1 
        
    loss = sum(val_loss)/len(val_loss)
    psnr = sum(val_psnr)/len(val_psnr)
    with torch.no_grad():
        gen_img = model(fixed_ip_valid)
        writer.add_image("Observation/Valid_img", gen_img.squeeze(0), global_step = step)
    
    writer.add_scalar("Validation/Loss", loss, global_step = step)
    writer.add_scalar("Validation/PSNR", psnr, global_step = step)
    return step
        

def main():
    
    TRAIN_DATA_IP = '/content/Data/Loldataset/our485/low'
    TRAIN_DATA_TARGET = '/content/Data/Loldataset/our485/high'
    VALID_DATA_IP = '/content/Data/Loldataset/eval15/low'
    VALID_DATA_TARGET = '/content/Data/Loldataset/eval15/high'
    SAVE_DIR = '/content/drive/MyDrive/MIRNet/Enhance'
    LOG_DIR = '/content/drive/MyDrive/MIRNet/Enhance'
    FIXED_IP_PATH = '/content/Data/Loldataset/our485/low/100.png'
    FIXED_IP_TARGET = '/content/Data/Loldataset/our485/high/100.png'
    FIXED_IP1_PATH = '/content/Data/Loldataset/eval15/low/111.png'
    
    device = torch.device('cuda')
    model = MIRNet().to(device)
    criterion = Loss_fn()
    opt = torch.optim.Adam(model.parameters(), lr = 1e-4, betas = (0.9,0.999), weight_decay = 1e-8, eps = 1e-8)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 100, 5e-5, verbose = True)
    writer = SummaryWriter(LOG_DIR)
    
    train_data = LoLDataset(low_light_root = TRAIN_DATA_IP, target_root = TRAIN_DATA_TARGET)
    val_data = LoLDataset(low_light_root = VALID_DATA_IP, target_root = VALID_DATA_TARGET)
    train_loader = DataLoader(train_data, batch_size = 4, shuffle = True, num_workers = 2)
    valid_loader = DataLoader(val_data, batch_size = 3, shuffle = False)
    
    global fixed_ip_train
    fixed_ip_train = T.Compose([T.Resize((128,128)), T.ToTensor()])(Image.open(FIXED_IP_PATH)).unsqueeze(0)
    fixed_ip_train = fixed_ip_train.to(device)
    
    global fixed_ip_valid
    fixed_ip_valid = T.Compose([T.Resize((128,128)), T.ToTensor()])(Image.open(FIXED_IP1_PATH)).unsqueeze(0)
    fixed_ip_valid = fixed_ip_valid.to(device)
    
    target_img = T.Compose([T.Resize((128,128)), T.ToTensor()])(Image.open(FIXED_IP_TARGET))
    target_img = target_img.to(device)
    step, val_step = 0, 0
    
    for epoch in range(100):
        
        with torch.no_grad():
          gen_img1 = model(fixed_ip_valid)
          vutils.save_image(gen_img1, open( SAVE_DIR + f'img{epoch}.png', 'wb'), normalize = True)
        
        writer.add_image("Images/Input", fixed_ip_valid.squeeze(0).cpu(), global_step = step)
        writer.add_image("Images/Generated", gen_img1.squeeze(0).cpu(), global_step = step)
        writer.add_image("Images/Target", target_img.cpu(), global_step = step)
          
        step = train(train_loader, model, criterion, opt, writer, step, epoch)
        scheduler.step()
        torch.save(model.state_dict(), open(SAVE_DIR + 'Mirnet_enhance.pth', 'wb'))
        val_step = validate(valid_loader, model, criterion, writer, val_step, epoch)
        

if __name__ == '__main__':
    
    main()
    