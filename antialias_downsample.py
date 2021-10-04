import torch
import torch.nn as nn
import torch.nn.functional as fun
import numpy as np

class DownSample(nn.Module):
    
    def __init__(self, pad_type = 'reflect', filter_size = 3, stride = 2, channels = None, pad_off = 0):
        
        super().__init__()
        self.filter_size = filter_size
        self.stride = stride
        self.pad_off = pad_off
        self.channels = channels
        self.pad_sizes = [int(1.0 * (filter_size - 1) / 2),
                          int(np.ceil(1.0 * (filter_size - 1) / 2)),
                          int(1.0 * (filter_size - 1) / 2),
                          int(np.ceil(1.0 * (filter_size - 1) / 2))]
    
        self.pad_sizes = [pad_size + pad_off for pad_size in self.pad_sizes]
        self.off = int((self.stride - 1) / 2.0)
        
        if self.filter_size == 1:
            a = np.array([1.0])
        elif self.filter_size == 2:
            a = np.array([1.0, 1.0])
        elif self.filter_size == 3:
            a = np.array([1.0, 2.0, 1.0])
        elif self.filter_size == 4:
            a = np.array([1.0, 3.0, 3.0, 1.0])
        elif self.filter_size == 5:
            a = np.array([1.0, 4.0, 6.0, 4.0, 1.0])
        elif self.filter_size == 6:
            a = np.array([1.0, 5.0, 10.0, 10.0, 5.0, 1.0])
        elif self.filter_size == 7:
            a = np.array([1.0, 6.0, 15.0, 20.0, 15.0, 6.0, 1.0])
            
        filt = torch.Tensor(a[:, None] * a[None, :])
        filt = filt / torch.sum(filt)
        self.register_buffer('filt', filt[None, None, :, :].repeat((self.channels, 1, 1, 1)))
        self.pad = get_pad_layer(pad_type)(self.pad_sizes)
        
    def forward(self, x):
        
        if self.filter_size == 1:
            if self.pad_off == 0:
                return x[:, :, ::self.stride, ::self.stride]
            else:
                return self.pad(x)[:, :, ::self.stride, ::self.stride]
        
        else:
            return fun.conv2d(self.pad(x), self.filt, stride = self.stride, groups = x.shape[1])
        

def get_pad_layer(pad_type):
    
    if pad_type == 'reflect':
        pad_layer = nn.ReflectionPad2d
    elif pad_type == 'replication':
        pad_layer = nn.ReplicationPad2d
    else:
        print('Pad Type [%s] not recognized' % pad_type)
    
    return pad_layer