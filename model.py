import torch
import torch.nn as nn
import numpy as np
from antialias_downsample import DownSample 


class CompressChannels(nn.Module):
        
    """
        Compresses the input channels to 2 by concatenating the results of
        Global Average Pooling(GAP) and Global Max Pooling(GMP).
        HxWxC => HxWx1

    """
    
    def forward(self, x):
        
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim = 1)

class SpatialAttention(nn.Module):
    
    '''

    Spatial Attention: 

                    HxWxC
                      |
                  ---------
                  |       |
                 GAP     GMP
                  |       |
                  ----C---
                      |
                    HxWx2
                      |
                    Conv
                     |
                  Sigmoid
                     |
                   HxWx1
                   
    Multiplying HxWx1 with input again gives output -> HxWxC

    '''
    
    def __init__(self):
        
        super().__init__()
        self.compress_channels = CompressChannels()
        self.conv = nn.Conv2d(in_channels = 2, out_channels = 1, kernel_size = 5,
                              stride = 1, padding = 2)
        
    def forward(self, x):
        
        compress_x = self.compress_channels(x)
        x_out = self.conv(compress_x)
        scale = torch.sigmoid(x_out)
        return x * scale
    

class ChannelAttention(nn.Module):
    
    '''

    Channel Attention(Squeeze and Excitation Operation): 

                    HxWxC
                      |       
                     GAP     
                      |  
                    1x1xC
                      |
                 Conv + PReLU 
                      |
                    1x1xC/r (r = reduction ratio)
                      |
                    Conv
                     |
                   1x1xC
                     |
                  Sigmoid
                   
    Multiplying 1x1xC with input again gives output -> HxWxC

    '''
    
    def __init__(self, channels, r = 8, bias = True):
        
        super().__init__()
        #Squeeze
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        #excitation
        self.excite = nn.Sequential(nn.Conv2d(channels, channels // r, kernel_size = 1,
                                              padding = 0, bias = bias),
                                    nn.PReLU(),
                                    nn.Conv2d(channels // r, channels, kernel_size = 1,
                                              padding = 0, bias = bias),
                                    nn.Sigmoid()
                                    )
        
    def forward(self, x):
        
        out = self.squeeze(x)
        out = self.excite(out)
        return out * x


class DAU(nn.Module):
    
    '''

    Dual Attention Unit(DAU) :

          --------- HxWxC
          '            |
          '    Conv + PReLU + Conv
          '            |
          '        -------- 
          '        |      |
          '       SA     CA
          '       |      |
          '       -------
          '          |
          '       Concate
          '          |
          '        Conv
          '         |
          '---------+
                    |
                 Output
                   

    '''
    def __init__(self, channels, kernel_size = 3, r = 8, bias = False):
        
        super().__init__()
        self.block = nn.Sequential(nn.Conv2d(channels, channels, kernel_size, padding = 1, bias = bias),
                                   nn.PReLU(),
                                   nn.Conv2d(channels, channels, kernel_size, padding = 1, bias = bias))
        self.SA = SpatialAttention()
        self.CA = ChannelAttention(channels, r, bias = bias)
        self.conv1x1 = nn.Conv2d(channels*2, channels, kernel_size = 1, bias = bias)
        
    def forward(self, x):
        
        res = self.block(x)
        _sa = self.SA(res)
        _ca = self.CA(res)
        res = torch.cat([_sa, _ca], dim = 1)
        res = self.conv1x1(res)
        res += x
        return res
    

class UpsampleBlock(nn.Module):
    
    '''
    
                HxWxC----------------
                  |                 '
          Conv1x1 + PReLU           |
                  |           Bilinear Upsampling
          Conv3x3 + PReLU           |
                 |               Conv1x1
         Bilinear Upsampling        |
                 |                  ' 
              Conv1x1               '
                 |                  '
                 +-------------------
                 |
            2H x 2W x C/2
                 
    '''
    
    def __init__(self, channels, bias = False):
        
        super().__init__()
        
        self.left = nn.Sequential(nn.Conv2d(channels, channels, kernel_size = 1, padding = 0, bias = bias),
                                  nn.PReLU(),
                                  nn.Conv2d(channels, channels, kernel_size = 3, padding = 1, bias = bias),
                                  nn.PReLU(),
                                  nn.Upsample(scale_factor = 2, mode = 'bilinear', align_corners = bias),
                                  nn.Conv2d(channels, channels//2, kernel_size = 1, padding = 0, bias = bias))
        
        self.right = nn.Sequential(nn.Upsample(scale_factor = 2, mode = 'bilinear', align_corners = bias),
                                  nn.Conv2d(channels, channels//2, kernel_size = 1, padding = 0, bias = bias))
        
    def forward(self, x):
        
        left = self.left(x)
        right = self.right(x)
        out = left + right
        return out

    
class UpSample(nn.Module):
    
    def __init__(self, channels, scale_factor, stride = 2):
        
        super().__init__()
        self.scale_factor = int(np.log2(scale_factor))
        modules = []
        
        for i in range(self.scale_factor):
            modules.append(UpsampleBlock(channels))
            channels = int(channels // 2)
        
        self.block = nn.Sequential(*modules)
        
    def forward(self, x):
        
        return self.block(x)

    
class DownSampleBlock(nn.Module):
    
    '''
    
                HxWxC----------------
                  |                 '
          Conv1x1 + PReLU           |
                  |         Antialias Downsampling
          Conv3x3 + PReLU           |
                 |               Conv1x1
         Antialias Downsampling     |
                 |                  ' 
              Conv1x1               '
                 |                  '
                 +-------------------
                 |
            H/2 x W/2 x 2C
                 
    '''
    
    def __init__(self, channels, bias = False):
        
        super().__init__()
        self.left = nn.Sequential(nn.Conv2d(channels, channels, kernel_size = 1, padding = 0, bias = bias),
                                  nn.PReLU(),
                                  nn.Conv2d(channels, channels, kernel_size = 3, padding = 1, bias = bias),
                                  nn.PReLU(),
                                  DownSample(channels = channels, filter_size = 3, stride = 2),
                                  nn.Conv2d(channels, channels*2, kernel_size = 1, padding = 0, bias = bias))
        
        self.right = nn.Sequential(DownSample(channels = channels, filter_size = 3, stride = 2),
                                  nn.Conv2d(channels, channels*2, kernel_size = 1, padding = 0, bias = bias))
        
    def forward(self, x):
        
        left = self.left(x)
        right = self.right(x)
        out = left + right
        return out
 
       
class DownSamp(nn.Module):
    
    def __init__(self, channels, scale_factor, stride = 2):
        
        super().__init__()
        self.scale_factor = int(np.log2(scale_factor))
        modules = []
        
        for i in range(self.scale_factor):
            modules.append(DownSampleBlock(channels))
            channels = int(channels * stride)
        
        self.block = nn.Sequential(*modules)
        
    def forward(self, x):
        
        return self.block(x)        


class SKFF(nn.Module):
    
    def __init__(self, in_c, r, bias = False):
        
        super().__init__()
        d = max(int(in_c/r), 4)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Sequential(nn.Conv2d(in_c, d, kernel_size = 1, padding = 0, bias = bias),
                                  nn.PReLU())
        self.attention_fcs = nn.ModuleList([])
        
        for i in range(3):
            self.attention_fcs.append(nn.Conv2d(d, in_c, kernel_size = 1, stride = 1, bias = bias))
        
        self.softmax = nn.Softmax(dim = 1)
        
    def forward(self, in_features):
        
        batch_size = in_features[0].shape[0]
        num_features = in_features[0].shape[1]
        
        in_features = torch.cat(in_features, dim = 1)
        print(in_features.shape)
        in_features = in_features.view(batch_size, 3, num_features, in_features.shape[2], in_features.shape[3])
        
        features_u = torch.sum(in_features, dim = 1)
        features_s = self.avg_pool(features_u)
        features_z = self.conv(features_s)
        
        attn_vectors = [fc(features_z) for fc in self.attention_fcs]
        attn_vectors = torch.cat(attn_vectors, dim = 1)
        attn_vectors = attn_vectors.view(batch_size, 3, num_features, 1, 1)
        
        attn_vectors = self.softmax(attn_vectors)
        features_v = torch.sum(in_features * attn_vectors, dim = 1)
        
        return features_v
        

class MSRB(nn.Module):
    
    def __init__(self, num_features, height, width, stride, bias):
        
        super().__init__()
        self.num_features = num_features
        self.height = height
        self.width = width
        
        self.dau_blocks = nn.ModuleList([nn.ModuleList([DAU(int(num_features*stride**i))]*width) for i in range(height)])
        
        feats = [int((stride**i)*num_features) for i in range(height)]
        scale = [2**i for i in range(1, height)]
        
        self.last_up = nn.ModuleDict()
        for i in range(1, height):
            self.last_up.update({f'{i}': UpSample(channels = int(num_features*stride**i), scale_factor = 2**i, stride = stride)})
            
        self.down = nn.ModuleDict()
        i = 0
        
        scale.reverse()
        for f in feats:
            for s in scale[i:]:
                self.down.update({f'{f}_{s}': DownSamp(f, s, stride)})
            i+=1
            
        self.up = nn.ModuleDict()
        i = 0
        
        feats.reverse()
        for f in feats:
            for s in scale[i:]:
                self.up.update({f'{f}_{s}': UpSample(f, s, stride)})
            i+=1
            
        self.out_conv = nn.Conv2d(num_features, num_features, kernel_size = 3, padding = 1, bias = bias)
        self.skff_blocks = nn.ModuleList([SKFF(num_features*stride**i, height) for i in range(height)])
        
    def forward(self, x):
        
        inp = x.clone()
        out = []
        
        for j in range(self.height):
            if j==0:
                inp = self.dau_blocks[j][0](inp)
            else:
                inp = self.dau_blocks[j][0](self.down[f'{inp.size(1)}_{2}'](inp))
            out.append(inp)
            
        for i in range(1, self.width):
            
            if True:
                temp = []
                for j in range(self.height):
                    TENSOR = []
                    nfeats = (2**j)*self.num_features
                    for k in range(self.height):
                        TENSOR.append(self.select_up_down(out[k], j, k))
                    
                    skff = self.skff_blocks[j](TENSOR)
                    temp.append(skff)
                    
            else:
                
                temp = out
                
            for j in range(self.height):
                
                out[j] = self.dau_blocks[j][i](temp[j])
                
        output = []
        for k in range(self.height):
            
            output.append(self.select_last_up(out[k], k))
            
        output = self.skff_blocks[0](output)
        output = self.out_conv(output)
        output = output + x
        return output
    
    def select_up_down(self, tensor, j, k):
        
        if j == k:
            return tensor
        else:
            diff = 2 ** np.abs(j-k)
            if j < k:
                return self.up[f'{tensor.size(1)}_{diff}'](tensor)
            else:
                return self.down[f'{tensor.size(1)}_{diff}'](tensor)
            
    def select_last_up(self, tensor, k):
        
        if k == 0:
            return tensor
        else:
            return self.last_up[f'{k}'](tensor)
        
        
class RRG(nn.Module):
    
    def __init__(self, num_features, num_MSRB, height, width, stride, bias = False):
        
        super().__init__()
        modules = [MSRB(num_features, height, width, stride, bias) for _ in range(num_MSRB)]
        modules.append(nn.Conv2d(num_features, num_features, kernel_size = 3, padding = 1, stride = 1, bias = bias))
        self.blocks = nn.Sequential(*modules)
        
    def forward(self, x):
        
        out = self.blocks(x)
        out += x
        return out
    

class MIRNet(nn.Module):
    
    def __init__(self, in_c = 3, out_c = 3, num_features = 64, kernel_size = 3, stride = 2, 
                 num_MSRB = 2, num_RRG = 3, height = 3, width = 2, bias = False):
        
        super().__init__()
        self.first_conv = nn.Conv2d(in_c, num_features, kernel_size, padding = 1, bias = bias)
        modules = [RRG(num_features, num_MSRB, height, width, stride, bias) for _ in range(num_RRG)]
        self.mir_blocks = nn.Sequential(*modules)
        self.final_conv = nn.Conv2d(num_features, out_c, kernel_size, padding = 1, bias = bias)
        
    def forward(self, x):
        
        out = self.first_conv(x)
        out = self.mir_blocks(out)
        out = self.final_conv(out)
        out += x 
        return out
    
                
if __name__ == '__main__':
    
    #t1 = torch.randn((1, 3, 64, 64))
    model = MIRNet()
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(pytorch_total_params)