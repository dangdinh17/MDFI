import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def generate_it(x, t=0, nf=3, f=7):
    index = np.array([t - nf // 2 + i for i in range(nf)])
    index = np.clip(index, 0, f-1).tolist()
    it = x[:, index, :, :]
    return it

def make_layer(block, num_of_layer):
    layers = []
    for _ in range(num_of_layer):
        layers.append(block())
    return nn.Sequential(*layers)

class ConBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.relu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)
        return out


class ChannelAttention(nn.Module):
    def __init__(self, num_features, reduction=16):
        super(ChannelAttention, self).__init__()
        self.pool = nn.ModuleList([
            nn.AdaptiveAvgPool2d(1),
            nn.AdaptiveMaxPool2d(1)
        ])
        self.mlp = nn.Sequential(
            nn.Linear(num_features, num_features // reduction, bias=False),
            nn.GELU(),
            nn.Linear(num_features // reduction, num_features, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # perform pool with independent Pooling
        b, c, _, _ = x.size()
        avg_feat = self.pool[0](x).view(b, c)
        max_feat = self.pool[1](x).view(b, c)
        # perform mlp with the same mlp sub-net
        avg_out = self.mlp(avg_feat)
        max_out = self.mlp(max_feat)
        # attention
        attention = self.sigmoid(avg_out + max_out).view(b, c, 1, 1)
        return attention * x #B , C, H, W
    
class SpatialAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SpatialAttention, self).__init__()
        self.pool = nn.ModuleList([
            nn.AdaptiveMaxPool2d(1),
            nn.AdaptiveAvgPool2d(1)
        ])
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.GELU(),
            nn.Linear(channels // reduction, channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.shape
        out = self.pool[0](x)
        out = self.pool[1](out)
        attention = self.mlp(out.view(b, c)).view(b, c, 1, 1)
        return x * attention #B , C, H, W


class SKM(nn.Module):
    def __init__(self, channels=64, reduction=4, groups='depthwise', bias=False):
        super().__init__()
        if groups == 'depthwise':
            groups = channels
        elif groups == 'standard':
            groups = 1
        self.channels = channels
        self.fuseconv = nn.Conv2d(channels*2, channels, 1, 1, 0, bias=bias, groups=groups)
        
        self.path1 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, groups=groups, bias=bias),
            ChannelAttention(channels, reduction),
            SpatialAttention(channels, reduction)
        )
        self.path3 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, groups=groups, bias=bias),
            ChannelAttention(channels, reduction),
            SpatialAttention(channels, reduction)
        )
        self.conv = nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0, groups=groups, bias=bias)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels//reduction),
            nn.ReLU(),
            nn.Linear(channels//reduction, channels*2),
            nn.Sigmoid()
            )
        
    def forward(self,x):
        x = self.fuseconv(x)
        b, c, h, w = x.shape
        path1 = self.path1(x)
        path3 = self.path3(x)
        
        path2 = self.conv(x)
        path2 = self.pool(path2).view(b, c)
        path2 = self.mlp(path2)
        path21, path23 = path2[:, :self.channels], path2[:, self.channels:]

        path21 = path21.view(b, c, 1, 1)
        path23 = path23.view(b, c, 1, 1)
        out = path1 * path21 + path3 * path23
        return out

class SKMU(nn.Module):
    def __init__(self, nf):
        super().__init__()
        self.nf = nf
        base_ks = 3
        self.Down0_0 = nn.Sequential(
            nn.Conv2d(nf, nf, base_ks, stride=2, padding=base_ks // 2),
            nn.LeakyReLU(negative_slope=0.1, inplace=True)
        )
        self.conv0_0 = ConBlock(nf, nf)

        self.Down0_1 = nn.Sequential(
            nn.Conv2d(nf, nf, base_ks, stride=2, padding=base_ks // 2),
            nn.LeakyReLU(negative_slope=0.1, inplace=True)
        )
        self.conv0_1 = ConBlock(nf, nf)

        self.Down0_2 = nn.Sequential(
            nn.Conv2d(nf, nf, base_ks, stride=2, padding=base_ks // 2),
            nn.LeakyReLU(negative_slope=0.1, inplace=True)
        )
        self.conv0_2 = ConBlock(nf, nf)

        self.Up1 = nn.Sequential(
            nn.ConvTranspose2d(nf, nf, 4, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True)
        )

        self.SKM_1 = SKM(channels=nf, reduction=8)
        self.Up2 = nn.Sequential(
            nn.ConvTranspose2d(nf, nf, 4, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True)
        )

        self.SKM_2 = SKM(channels=nf, reduction=8)
        self.Up3 = nn.Sequential(
            nn.ConvTranspose2d(nf, nf, 4, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True)
        )
        
    def forward(self, input):
        x0_0 = self.conv0_0(self.Down0_0(input))
        x0_1 = self.conv0_1(self.Down0_1(x0_0))
        x0_2 = self.conv0_2(self.Down0_2(x0_1))
        up0_1 = self.Up1(x0_2)
        b,n,h,w = x0_1.shape
        up0_1 = up0_1[:,:,:h,:w]
        up0_2 = self.Up2(self.SKM_1(torch.cat((up0_1, x0_1), dim=1)))
        up0_3 = self.Up3(self.SKM_2(torch.cat((up0_2, x0_0), dim=1)))
        return up0_3+input


class FER(nn.Module):
    def __init__(self,  in_nc, out_nc, connection=True):
        super(FER, self).__init__()
        self.connection = connection
        if connection==True:
            self.decrease_dim = nn.Conv2d(in_nc, out_nc, 1, stride=1, padding=0)

        self.high_freq = nn.Sequential(
            nn.Conv2d(in_nc, out_nc, 3, stride=1, padding=3 // 2),
            nn.LeakyReLU(negative_slope=0.1, inplace=True)
        )
        self.mid_freq = nn.Sequential(
            nn.Conv2d(in_nc, out_nc, 3, stride=2, padding=3 // 2),
            nn.LeakyReLU(negative_slope=0.1, inplace=True)
        )
        self.low_freq = nn.Sequential(
            nn.Conv2d(in_nc, out_nc, 3, stride=4, padding=3 // 2),
            nn.LeakyReLU(negative_slope=0.1, inplace=True)
        )
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.dcn_1 = nn.Sequential(
            nn.Conv2d(out_nc, out_nc, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True)
        )
        self.dcn_2 = nn.Sequential(
            nn.Conv2d(out_nc, out_nc, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True)
        )
        self.dcn_3 = nn.Sequential(
            nn.Conv2d(out_nc, out_nc, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True)
        )

        self.SKM_1 = SKM(channels=out_nc, reduction=8)
        self.SKM_2 = SKM(channels=out_nc, reduction=8)
        self.se = ChannelAttention(out_nc, reduction=32)
    def forward(self, x):
        f_l = self.low_freq(x)
        x_2 = self.mid_freq(x)
        x_1 = self.high_freq(x)
        f_m = x_2 - self.up(f_l)
        f_h = x_1 - self.up(x_2)

        f_l_enc = self.dcn_3(f_l)

        f_ml_enc = self.dcn_2(self.SKM_1(torch.cat((self.up(f_l_enc), f_m), dim=1)))
        f_mlh_enc = self.dcn_1(self.SKM_2(torch.cat((self.up(f_ml_enc), f_h), dim=1)))
        f_mlh_enc = self.se(f_mlh_enc)
        if self.connection==True:
            x = self.decrease_dim(x)
        return f_mlh_enc+x
    
class PA(nn.Module):
    def __init__(self, out_nc=64, chanfactor=4, groups='depthwise', bias=False):
        super().__init__()
        if groups == 'depthwise':
            groups = out_nc
        elif groups == 'standard':
            groups = 1
        self.channels = out_nc
        self.conv1 = nn.Sequential(
            nn.Conv2d(out_nc, out_nc, kernel_size=1, stride=1, groups=groups, padding=0, bias=bias),
            nn.Sigmoid()
        )
       
        self.conv2 = nn.Conv2d(out_nc, out_nc, kernel_size=3, stride=1, padding=1, groups=groups, bias=bias)
        
    def forward(self,x):
        b, c, h, w = x.shape
        
        out1 = self.conv1(x)
        out2 = self.conv2(x)
        
        out = out1 * out2
        return out
    
class Enhancer(nn.Module):
    def __init__(self, in_nc=32, out_nc=64, chanfactor=4, groups='depthwise', bias=False):
        super().__init__()
        if groups == 'depthwise':
            groups = out_nc
        elif groups == 'standard':
            groups = 1
        self.channels = in_nc
        self.conv1 = nn.Conv2d(in_nc, out_nc, 1, 1, 0, bias=bias)
        # self.conv2 = nn.Conv2d(in_nc + out_nc, out_nc, 3, 1, 1, bias=bias)
        # self.conv3 = nn.Conv2d(in_nc + 2 * out_nc, out_nc, 3, 1, 1, bias=bias)
        # self.conv4 = nn.Conv2d(in_nc + 3 * out_nc, out_nc, 3, 1, 1, bias=bias)
        # self.conv5 = nn.Conv2d(in_nc + 4 * out_nc, out_nc, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        
        self.top_branch = nn.Sequential(
            nn.Conv2d(out_nc, out_nc//chanfactor, kernel_size=1, stride=1, padding=0, groups=groups//chanfactor, bias=bias),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            PA(out_nc=out_nc//chanfactor),
            nn.Conv2d(out_nc//chanfactor, out_nc, kernel_size=1, stride=1, padding=0, groups=groups//chanfactor, bias=bias)
        )
       
        self.bottom_branch = nn.Sequential(
                    nn.Conv2d(out_nc, out_nc//chanfactor, kernel_size=1, stride=1, padding=0, groups=groups//chanfactor, bias=bias),
                    nn.LeakyReLU(negative_slope=0.2, inplace=True),
                    nn.Conv2d(out_nc//chanfactor, out_nc, kernel_size=1, stride=1, padding=0, groups=groups//chanfactor, bias=bias)
        )
        self.fuse = SKM(channels=out_nc)
        
        
    def forward(self,x):
        b, c, h, w = x.shape
        x5 = self.lrelu(self.conv1(x))
        # x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        # x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        # x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        # x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1)) # n 64 h w
        
        top = self.top_branch(x5)
        bottom = self.bottom_branch(x5)
        
        out = torch.cat([top, bottom], dim=1)
        # print(f"Shape after conv1: {out1.shape}, Shape after conv2: {out2.shape}, "
        #         f"Shape after concatenation: {out.shape}, Shape after fusion: {out.shape}")        
        out = self.fuse(out)
        return out + self.conv1(x)
    
    