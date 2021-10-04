import torch
import numpy as np
import cv2

class MixUp:
    
    def __init__(self):
        
        self.distribution = torch.distributions.beta.Beta(torch.tensor([1.2]), torch.tensor([1.2]))
        
    def augment(self, gt, ip):
        
        batch_size = gt.size(0)
        idx = torch.randperm(batch_size)
        gt1 = gt[idx]
        ip1 = ip[idx]
        
        lam = self.distribution.rsample((batch_size, 1)).view(-1,1,1,1).cuda()
        gt = lam * gt + (1 - lam) * gt1
        ip = lam * ip + (1 - lam) * ip1
        
        return gt, ip


def calc_PSNR(target, pred):
    
    diff = torch.clamp(pred, 0, 1) - torch.clamp(target, 0, 1)
    rmse = (diff ** 2).mean().sqrt()
    res = 20 * torch.log10(1 / rmse)
    return res


def batch_psnr(img1, img2, data_range = None):
    
    psnr = []
    for i, j in zip(img1, img2):
        
        res = calc_PSNR(img1, img2)
        psnr.append(res)
    
    return sum(psnr)/len(psnr)


def load_img(path):
    
    img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32)
    img /= 255.0
    return img


def save_img(img, path):
    
    cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    
    