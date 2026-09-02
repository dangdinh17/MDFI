import os
import torch,functools
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import utils
from ops.dcn.deform_conv import ModulatedDeformConv
from .attentionlayer import DSTA
# from .attentionlayer import MultiHeadNonLocalAttention
from PIL import Image

# ==========
# Spatio-temporal deformable fusion module
# ==========
# The STDF module is implemented by RyanXingQL
# Thanks for his work! you may refer to https://github.com/RyanXingQL/STDF-PyTorch
# for more details about this.
class CSAM_Module(nn.Module):
    """ Channel-Spatial attention module"""
    def __init__(self, in_dim):
        super(CSAM_Module, self).__init__()
        self.chanel_in = in_dim


        self.conv = nn.Conv3d(1, 1, 3, 1, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        #self.softmax  = nn.Softmax(dim=-1)
        self.sigmoid = nn.Sigmoid()
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X N X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X N X N
        """
        m_batchsize, C, height, width = x.size()
        out = x.unsqueeze(1)
        out = self.sigmoid(self.conv(out))
        
        # proj_query = x.view(m_batchsize, N, -1)
        # proj_key = x.view(m_batchsize, N, -1).permute(0, 2, 1)
        # energy = torch.bmm(proj_query, proj_key)
        # energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy
        # attention = self.softmax(energy_new)
        # proj_value = x.view(m_batchsize, N, -1)

        # out = torch.bmm(attention, proj_value)
        # out = out.view(m_batchsize, N, C, height, width)

        out = self.gamma*out
        out = out.view(m_batchsize, -1, height, width)
        x = x * out + x
        return x



## Residual Channel Attention Block (RCAB)
class RCAB(nn.Module):
    def __init__(
        self, n_feat, kernel_size=3, reduction=16,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):
        super(RCAB, self).__init__()
        modules_body = []
        modules_body.append(nn.Conv2d(n_feat, n_feat, kernel_size, padding=1, bias=bias))
        modules_body.append(nn.ReLU(True)) 
        # modules_body.append(CAM(n_feat))  
        # modules_body.append(nn.Conv2d(n_feat, n_feat, kernel_size, padding=1, bias=bias))
        modules_body.append(nn.Conv2d(n_feat, n_feat, kernel_size, padding=1, bias=bias))
        modules_body.append(CALayer(n_feat, reduction))
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x)
        res += x
        return res


## Channel Attention (CA) Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y



class Unet(nn.Module):
    def __init__(self, in_nc, out_nc, nf, nb, base_ks=3):
        """
        Args:
            in_nc: num of input channels.
            out_nc: num of output channels.
            nf: num of channels (filters) of each conv layer.
            nb: num of conv layers.
            deform_ks: size of the deformable kernel.
        """
        super(Unet, self).__init__()

        self.nb = nb
        self.in_nc = in_nc
        # self.deform_ks = deform_ks
        # self.size_dk = deform_ks ** 2
        # u-shape backbone
        # self.in_conv = nn.Sequential(
        #     nn.Conv2d(in_nc, nf, base_ks, padding=base_ks//2),
        #     nn.ReLU(inplace=True)
        #     )
        for i in range(1, nb):
            setattr(
                self, 'dn_conv{}'.format(i), nn.Sequential(
                    nn.Conv2d(nf, nf, base_ks, stride=2, padding=base_ks//2),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(nf, nf, base_ks, padding=base_ks//2),
                    nn.ReLU(inplace=True)
                    )
                )
            setattr(
                self, 'up_conv{}'.format(i), nn.Sequential(
                    nn.Conv2d(2*nf, nf, base_ks, padding=base_ks//2),
                    nn.ReLU(inplace=True),
                    nn.ConvTranspose2d(nf, nf, 4, stride=2, padding=1),
                    nn.ReLU(inplace=True)
                    )
                )
        self.tr_conv = nn.Sequential(
            nn.Conv2d(nf, nf, base_ks, stride=2, padding=base_ks//2),
            nn.ReLU(inplace=True),
            nn.Conv2d(nf, nf, base_ks, padding=base_ks//2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(nf, nf, 4, stride=2, padding=1),
            nn.ReLU(inplace=True)
            )
        self.out_conv = nn.Sequential(
            nn.Conv2d(nf, nf, base_ks, padding=base_ks//2),
            nn.ReLU(inplace=True)
            )

        # regression head
        # why in_nc*3*size_dk?
        #   in_nc: each map use individual offset and mask
        #   2*size_dk: 2 coordinates for each point
        #   1*size_dk: 1 confidence (attention) score for each point
        # self.offset_mask = nn.Conv2d(
        #     nf, in_nc*3*self.size_dk, base_ks, padding=base_ks//2
        #     )

        # deformable conv
        # notice group=in_nc, i.e., each map use individual offset and mask
        # self.deform_conv = ModulatedDeformConv(
        #     in_nc, out_nc, deform_ks, padding=deform_ks//2, deformable_groups=in_nc
        #     )

    def forward(self, inputs):
        nb = self.nb
        in_nc = self.in_nc
        # n_off_msk = self.deform_ks * self.deform_ks

        # feature extraction (with downsampling)
        # out_lst = [self.in_conv(inputs)]  # record feature maps for skip connections
        out_lst = [inputs]  # record feature maps for skip connections
        for i in range(1, nb):
            dn_conv = getattr(self, 'dn_conv{}'.format(i))
            out_lst.append(dn_conv(out_lst[i - 1]))
        # trivial conv
        out = self.tr_conv(out_lst[-1])
        # feature reconstruction (with upsampling)
        for i in range(nb - 1, 0, -1):
            up_conv = getattr(self, 'up_conv{}'.format(i))
            out = up_conv(
                torch.cat([out, out_lst[i]], 1)
                )

        # compute offset and mask
        # offset: conv offset
        # mask: confidence
        # print("OUT SIZE",out.size())
        out = self.out_conv(out)
        # off_msk = self.offset_mask((out))
        # off = off_msk[:, :in_nc*2*n_off_msk, ...]
        # msk = torch.sigmoid(
        #     off_msk[:, in_nc*2*n_off_msk:, ...]
        #     )
        # print("OFF_MSK",off_msk.size(),"OFF",off.size(),"MSK",msk.size())
        # print("INPUS",inputs.size())
        # perform deformable convolutional fusion
        # fused_feat = F.relu(
        #     self.deform_conv(inputs, off, msk), 
        #     inplace=True
        #     )

        return out



class ResidualBlock_noBN(nn.Module):
    '''Residual block w/o BN
    ---Conv-ReLU-Conv-+-
     |________________|
    '''
    def __init__(self, nf=64):
        super(ResidualBlock_noBN, self).__init__()
        self.conv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        # initialization
        # initialize_weights([self.conv1, self.conv2], 0.1)

    def forward(self, x):
        identity = x
        out = F.relu(self.conv1(x), inplace=True)
        out = self.conv2(out)
        return identity + out



class RCABnet(nn.Module):
    def __init__(self, in_nc, out_nc, nf, nb, base_ks=3):
        """
        Args:
            in_nc: num of input channels.
            out_nc: num of output channels.
            nf: num of channels (filters) of each conv layer.
            nb: num of conv layers.
            deform_ks: size of the deformable kernel.
        """
        super(RCABnet, self).__init__()

        self.nb = nb
        self.in_nc = in_nc
        # self.deform_ks = deform_ks
        # self.size_dk = deform_ks ** 2
        # u-shape backbone
        self.in_conv = nn.Sequential(
            nn.Conv2d(in_nc, nf, base_ks, padding=base_ks//2),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(nf, nf, base_ks, padding=base_ks//2),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(nf, nf, base_ks, padding=base_ks//2),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            )
        
        self.resblock1 = ResidualBlock_noBN(nf)
        self.resblock2 = ResidualBlock_noBN(nf)
        self.resblock3 = ResidualBlock_noBN(nf)
        self.resblock4 = ResidualBlock_noBN(nf)
        self.resblock5 = ResidualBlock_noBN(nf)
        self.RCAB = RCAB(nf)
        self.out_conv = nn.Sequential(
            nn.Conv2d(nf, nf, base_ks, padding=base_ks//2),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(nf, nf, base_ks, padding=base_ks//2),
            nn.LeakyReLU(negative_slope=0.1, inplace=True)
            )

    def forward(self, inputs):
        nb = self.nb
        in_nc = self.in_nc

        # feature extraction (with downsampling)
        # out_lst = [self.in_conv(inputs)]  # record feature maps for skip connections
        out_lst = self.in_conv(inputs)  # record feature maps for skip connections
        out_lst = self.resblock1(out_lst)
        out_lst = self.resblock2(out_lst)
        out_lst = self.resblock3(out_lst)
        out_lst = self.resblock4(out_lst)
        out_lst = self.resblock5(out_lst)
        out = self.RCAB(out_lst)
        out = self.out_conv(out)

        return out



class Deform_block(nn.Module):
    def __init__(self, in_nc, out_nc, nf, base_ks=3, deform_ks=3):
        """
        Args:
            in_nc: num of input channels.
            out_nc: num of output channels.
            nf: num of channels (filters) of each conv layer.
            nb: num of conv layers.
            deform_ks: size of the deformable kernel.
        """
        super(Deform_block, self).__init__()

        self.in_nc = in_nc
        self.deform_ks = deform_ks
        self.size_dk = deform_ks ** 2
        # regression head
        # why in_nc*3*size_dk?
        #   in_nc: each map use individual offset and mask
        #   2*size_dk: 2 coordinates for each point
        #   1*size_dk: 1 confidence (attention) score for each point
        self.offset_mask = nn.Conv2d(
            nf, in_nc*3*self.size_dk, base_ks, padding=base_ks//2
            )

        # deformable conv
        # notice group=in_nc, i.e., each map use individual offset and mask
        self.deform_conv = ModulatedDeformConv(
            in_nc, out_nc, deform_ks, padding=deform_ks//2, deformable_groups=in_nc
            )

    def forward(self, inputs, feat):
        in_nc = self.in_nc
        n_off_msk = self.deform_ks * self.deform_ks

        # compute offset and mask
        # offset: conv offset
        # mask: confidence
        # off_msk = self.offset_mask(self.out_conv(out))
        off_msk = self.offset_mask(feat)
        off = off_msk[:, :in_nc*2*n_off_msk, ...]
        msk = torch.sigmoid(
            off_msk[:, in_nc*2*n_off_msk:, ...]
            )
        # perform deformable convolutional fusion
        fused_feat = F.relu(
            self.deform_conv(inputs, off, msk), 
            inplace=True
            )

        return fused_feat



class STDF_0(nn.Module):
    def __init__(self, in_nc, out_nc, nf, nb, base_ks=3, deform_ks=3):
        """
        Args:
            in_nc: num of input channels.
            out_nc: num of output channels.
            nf: num of channels (filters) of each conv layer.
            nb: num of conv layers.
            deform_ks: size of the deformable kernel.
        """
        super(STDF_0, self).__init__()

        self.nb = nb
        self.in_nc = in_nc
        self.deform_ks = deform_ks
        self.size_dk = deform_ks ** 2
        # u-shape backbone
        self.in_conv = nn.Sequential(
            nn.Conv2d(in_nc, nf, base_ks, padding=base_ks//2),
            nn.ReLU(inplace=True)
            )
        for i in range(1, nb):
            setattr(
                self, 'dn_conv{}'.format(i), nn.Sequential(
                    nn.Conv2d(nf, nf, base_ks, stride=2, padding=base_ks//2),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(nf, nf, base_ks, padding=base_ks//2),
                    nn.ReLU(inplace=True)
                    )
                )
            setattr(
                self, 'up_conv{}'.format(i), nn.Sequential(
                    nn.Conv2d(2*nf, nf, base_ks, padding=base_ks//2),
                    nn.ReLU(inplace=True),
                    nn.ConvTranspose2d(nf, nf, 4, stride=2, padding=1),
                    nn.ReLU(inplace=True)
                    )
                )
        self.tr_conv = nn.Sequential(
            nn.Conv2d(nf, nf, base_ks, stride=2, padding=base_ks//2),
            nn.ReLU(inplace=True),
            # nn.Conv2d(nf, nf, base_ks, padding=base_ks//2),
            # nn.ReLU(inplace=True),
            nn.ConvTranspose2d(nf, nf, 4, stride=2, padding=1),
            nn.ReLU(inplace=True)
            )
        # self.RCAB = RCAB(nf)
        # self.CAM = CAM(nf)
        # self.ResBlock_3d = ResBlock_3d(nf)
        # self.out_conv = nn.Sequential(
        #     nn.Conv2d(nf, nf, base_ks, padding=base_ks//2),
        #     nn.ReLU(inplace=True)
        #     )

        # regression head
        # why in_nc*3*size_dk?
        #   in_nc: each map use individual offset and mask
        #   2*size_dk: 2 coordinates for each point
        #   1*size_dk: 1 confidence (attention) score for each point
        self.offset_mask = nn.Conv2d(
            nf, in_nc*3*self.size_dk, base_ks, padding=base_ks//2
            )

        # deformable conv
        # notice group=in_nc, i.e., each map use individual offset and mask
        self.deform_conv = ModulatedDeformConv(
            in_nc, out_nc, deform_ks, padding=deform_ks//2, deformable_groups=in_nc
            )

    def forward(self, inputs):
        nb = self.nb
        in_nc = self.in_nc
        n_off_msk = self.deform_ks * self.deform_ks

        # feature extraction (with downsampling)
        
        out_lst = [self.in_conv(inputs)]  # record feature maps for skip connections
        for i in range(1, nb):
            dn_conv = getattr(self, 'dn_conv{}'.format(i))
            out_lst.append(dn_conv(out_lst[i - 1]))
        # trivial conv
        # out = self.CAM(self.tr_conv(out_lst[-1]))
        out = self.tr_conv(out_lst[-1])
        # feature reconstruction (with upsampling)
        for i in range(nb - 1, 0, -1):
            up_conv = getattr(self, 'up_conv{}'.format(i))
            out = up_conv(
                torch.cat([out, out_lst[i]], 1)
                )

        # compute offset and mask
        # offset: conv offset
        # mask: confidence
        # print("OUT SIZE",out.size())  self.out_conv
        # out = self.RCAB(out)
        # out = self.ResBlock_3d(out)
        off_msk = self.offset_mask((out))
        off = off_msk[:, :in_nc*2*n_off_msk, ...]
        msk = torch.sigmoid(
            off_msk[:, in_nc*2*n_off_msk:, ...]
            )
        # print("OFF_MSK",off_msk.size(),"OFF",off.size(),"MSK",msk.size())
        # print("INPUS",inputs.size())
        # perform deformable convolutional fusion
        fused_feat = F.relu(
            self.deform_conv(inputs, off, msk), 
            inplace=True
            )

        return fused_feat



class STDF(nn.Module):
    def __init__(self, in_nc, out_nc, nf, nb, base_ks=3, deform_ks=3):
        """
        Args:
            in_nc: num of input channels.
            out_nc: num of output channels.
            nf: num of channels (filters) of each conv layer.
            nb: num of conv layers.
            deform_ks: size of the deformable kernel.
        """
        super(STDF, self).__init__()

        self.nb = nb
        self.in_nc = in_nc
        self.deform_ks = deform_ks
        self.size_dk = deform_ks ** 2
        # u-shape backbone
        self.in_conv = nn.Sequential(
            nn.Conv2d(in_nc, nf, base_ks, padding=base_ks//2),
            nn.ReLU(inplace=True)
            )

        self.tra_conv = nn.Sequential(
            nn.Conv2d(nf, out_nc, base_ks, padding=base_ks//2),
            nn.ReLU(inplace=True)
            )

        # self.ResBlock_3d = ResBlock_3d(nf)
        self.RCAB = RCAB(nf)

        # regression head
        # why in_nc*3*size_dk?
        #   in_nc: each map use individual offset and mask
        #   2*size_dk: 2 coordinates for each point
        #   1*size_dk: 1 confidence (attention) score for each point
        self.offset_mask = nn.Conv2d(
            nf, in_nc*3*self.size_dk, base_ks, padding=base_ks//2
            )

        # deformable conv
        # notice group=in_nc, i.e., each map use individual offset and mask
        self.deform_conv = ModulatedDeformConv(
            in_nc, out_nc, deform_ks, padding=deform_ks//2, deformable_groups=in_nc
            )

    def forward(self, inputs):
        nb = self.nb
        in_nc = self.in_nc
        n_off_msk = self.deform_ks * self.deform_ks

        # feature extraction (with downsampling)
        #out_lst = [self.in_conv(inputs)]  # record feature maps for skip connections
        out = self.in_conv(inputs)
        out = self.RCAB(out)
        # out = self.ResBlock_3d(self.RCAB(out))
        # out = self.ResBlock_3d(self.RCAB(out))
        # out = self.ResBlock_3d(self.RCAB(out))
        # out = self.ResBlock_3d(self.RCAB(out))
        # out_in = self.tra_conv(out)


        # compute offset and mask
        # offset: conv offset
        # mask: confidence
        off_msk = self.offset_mask((out))
        off = off_msk[:, :in_nc*2*n_off_msk, ...]
        msk = torch.sigmoid(
            off_msk[:, in_nc*2*n_off_msk:, ...]
            )

        # perform deformable convolutional fusion
        fused_feat = F.relu(self.deform_conv(inputs, off, msk), inplace=True)

        return fused_feat



# ==========
# Quality enhancement module
# ==========

class PlainCNN(nn.Module):
    def __init__(self, in_nc=64, nf=48, nb=8, out_nc=3, base_ks=3,Att=True,Attname='None'):
        """
        Args:
            in_nc: num of input channels from STDF.
            nf: num of channels (filters) of each conv layer.
            nb: num of conv layers.
            out_nc: num of output channel. 3 for RGB, 1 for Y.
        """
        super(PlainCNN, self).__init__()

        self.in_conv = nn.Sequential(
            nn.Conv2d(in_nc, nf, base_ks, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(nf, nf, base_ks, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            )
        hid_conv_lst = []
        for _ in range(nb - 2):
            hid_conv_lst += [
                nn.Conv2d(nf, nf, base_ks, padding=1),
                nn.LeakyReLU(negative_slope=0.1, inplace=True),
                ]
            if Att:
                # hid_conv_lst+=[DSTA(nf)]
                hid_conv_lst+=[RCB(nf)]
                # hid_conv_lst+=[CAM(nf)]
                # hid_conv_lst+=[CCAM(nf)]
                print('[DSTA]')
        self.hid_conv = nn.Sequential(*hid_conv_lst)

        self.out_conv = nn.Conv2d(nf, out_nc, base_ks, padding=1)

    def forward(self, inputs):
        out = self.in_conv(inputs)
        out = self.hid_conv(out)
        out = self.out_conv(out)
        return out


# Empty Layer
class EmptyLayer(nn.Module):
    def __init__(self):
        super(EmptyLayer, self).__init__()
    
    def forward(self,x):
        return x



class ResBlock_3d(nn.Module):
    def __init__(self, nf):
        super(ResBlock_3d, self).__init__()
       
        self.dcn0 = nn.Conv3d(1, nf, kernel_size=3, stride=1, padding=1)
        self.dcn1 = nn.Conv3d(nf, 1, kernel_size=3, stride=1, padding=1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x):
        m_batchsize, C, height, width = x.size()
        x0 = x.unsqueeze(1)
        x1 = self.lrelu(self.dcn0(x0))
        out = self.dcn1(x1) + x0
        out = out.view(m_batchsize, -1, height, width)
        return out



class ContextBlock(nn.Module):

    def __init__(self, n_feat, bias=False):
        super(ContextBlock, self).__init__()

        self.conv_mask = nn.Conv2d(n_feat, 1, kernel_size=1, bias=bias)
        self.softmax = nn.Softmax(dim=2)

        self.channel_add_conv = nn.Sequential(
            nn.Conv2d(n_feat, n_feat, kernel_size=1, bias=bias),
            nn.LeakyReLU(0.2),
            nn.Conv2d(n_feat, n_feat, kernel_size=1, bias=bias))

    def modeling(self, x):
        batch, channel, height, width = x.size()
        input_x = x
        # [N, C, H * W]
        input_x = input_x.view(batch, channel, height * width)
        # [N, 1, C, H * W]
        input_x = input_x.unsqueeze(1)
        # [N, 1, H, W]
        context_mask = self.conv_mask(x)
        # [N, 1, H * W]
        context_mask = context_mask.view(batch, 1, height * width)
        # [N, 1, H * W]
        context_mask = self.softmax(context_mask)
        # [N, 1, H * W, 1]
        context_mask = context_mask.unsqueeze(3)
        # [N, 1, C, 1]
        context = torch.matmul(input_x, context_mask)
        # [N, C, 1, 1]
        context = context.view(batch, channel, 1, 1)

        return context

    def forward(self, x):
        # [N, C, 1, 1]
        # x = self.channel_add_conv(x)
        context = self.modeling(x)

        # [N, C, 1, 1]
        channel_add_term = self.channel_add_conv(context)
        x = x + channel_add_term
        # x = self.channel_add_conv(x)

        return x


### --------- Residual Context Block (RCB) ----------
class RCB(nn.Module):
    def __init__(self, n_feat, kernel_size=3, reduction=8, bias=False, groups=1):
        super(RCB, self).__init__()
        act = nn.LeakyReLU(0.2)
        # self.body = nn.Sequential(
        #     nn.Conv2d(n_feat, n_feat, kernel_size=3, stride=1, padding=1, bias=bias, groups=groups),
        #     act, 
        #     nn.Conv2d(n_feat, n_feat, kernel_size=3, stride=1, padding=1, bias=bias, groups=groups)
        # )

        self.act = act
        # self.cal = CALayer(n_feat, reduction = 16)
        nDiff = 16
        nFeat_slice = 4
        # self.Lattice_unit = LatticeBlock(n_feat, nDiff, nFeat_slice)  WLLBlock
        # self.Lattice_unit = LLBlock(n_feat)
        # self.Lattice_unit = WLLBlock(n_feat)
        self.channel1 = n_feat//2
        self.channel2 = n_feat-self.channel1
        
        self.gcnet = ContextBlock(self.channel1, bias=bias)
        self.tail = nn.Conv2d(n_feat, n_feat, kernel_size=3, stride=1, padding=1, bias=bias, groups=groups)
        
    def forward(self, x):
        # res = self.cal(x)
        # x = self.act(x)
        x1, x2 = torch.split(x,[self.channel1,self.channel2],dim=1)
        res1 = self.act(self.gcnet(x1))
        com1 = res1 + x2
        res2 = self.act(self.gcnet(com1))
        com2 = res2 + com1
        res = self.tail(torch.cat((com1,com2),dim=1))

        return res



## Combination Coefficient
class CC(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CC, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_mean = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )
        self.conv_std = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):

        # mean
        ca_mean = self.avg_pool(x)
        ca_mean = self.conv_mean(ca_mean)

        # std
        m_batchsize, C, height, width = x.size()
        x_dense = x.view(m_batchsize, C, -1)
        ca_std = torch.std(x_dense, dim=2, keepdim=True)
        ca_std = ca_std.view(m_batchsize, C, 1, 1)
        ca_var = self.conv_std(ca_std)

        # Coefficient of Variation
        # # cv1 = ca_std / ca_mean
        # cv = torch.div(ca_std, ca_mean)
        # ram = self.sigmoid(ca_mean + ca_var)

        cc = (ca_mean + ca_var)/2.0
        return cc


class CAM(nn.Module):
    """ Channel attention module"""
    def __init__(self, in_dim):
        super(CAM, self).__init__()
        self.chanel_in = in_dim


        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax  = nn.Softmax(dim=-1)
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X C X C
        """
        m_batchsize, C, height, width = x.size()
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma*out + x
        return out


class CCAM(nn.Module):
    """ Channel attention module"""
    def __init__(self, in_dim):
        super(CCAM, self).__init__()
        self.chanel_in = in_dim

        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax  = nn.Softmax(dim=-1)
        self.gcnet = ContextBlock(in_dim, bias=False)
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X C X C
        """
        x = self.gcnet(x)
        m_batchsize, C, height, width = x.size()
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma*out + x
        
        return out


#lightweight lattice block
class LLBlock(nn.Module):
    def __init__(self, num_fea):
        super(LLBlock, self).__init__()
        self.channel1=num_fea//2
        self.channel2=num_fea-self.channel1
        # self.convblock = nn.Sequential(
        #     nn.Conv2d(self.channel1, self.channel1, 3, 1, 1),
        #     nn.LeakyReLU(0.05),
        #     nn.Conv2d(self.channel1, self.channel1, 3, 1, 1),
        #     nn.LeakyReLU(0.05),
        #     nn.Conv2d(self.channel1, self.channel1, 3, 1, 1),
        # )

        self.convblock = ContextBlock(self.channel1)

        self.A_att_conv = CALayer(self.channel1)
        self.B_att_conv = CALayer(self.channel2)
        # self.CAM = CAM(self.channel1)
        # self.Sigmoid = nn.Sigmoid()

        # self.A_att_conv = ContextBlock(self.channel1)
        # self.B_att_conv = ContextBlock(self.channel2)

        self.fuse1 = nn.Conv2d(num_fea, self.channel1, 1, 1, 0)
        self.fuse2 = nn.Conv2d(num_fea, self.channel2, 1, 1, 0)
        self.fuse = nn.Conv2d(num_fea, num_fea, 1, 1, 0)


    def forward(self, x):
        x1,x2 = torch.split(x,[self.channel1,self.channel2],dim=1)

        x1 = self.convblock(x1)
        # x2 = self.convblock(x2)

        A = self.A_att_conv(x1)
        P = torch.cat((x2, A*x1),dim=1)

        B = self.B_att_conv(x2)
        Q = torch.cat((x1, B*x2),dim=1)

        c = torch.cat((self.fuse1(P),self.fuse2(Q)),dim=1)
        out = self.fuse(c)
        return out


#lightweight lattice block
class WLLBlock(nn.Module):
    def __init__(self, num_fea):
        super(WLLBlock, self).__init__()
        self.channel1=num_fea//2
        self.channel2=num_fea-self.channel1
        # self.convblock = nn.Sequential(
        #     nn.Conv2d(self.channel1, self.channel1, 3, 1, 1),
        #     nn.LeakyReLU(0.05),
        #     nn.Conv2d(self.channel1, self.channel1, 3, 1, 1),
        #     nn.LeakyReLU(0.05),
        #     nn.Conv2d(self.channel1, self.channel1, 3, 1, 1),
        # )

        self.convblock = ContextBlock(self.channel1)

        # self.A_att_conv = CALayer(self.channel1)
        # self.B_att_conv = CALayer(self.channel2)
        self.A_att_conv = nn.Sigmoid()
        self.B_att_conv = nn.Sigmoid()
        # self.CAM = CAM(num_fea)
        self.CB = ContextBlock(num_fea)
        self.act = nn.LeakyReLU(0.2)

        self.fuse1 = nn.Conv2d(num_fea, self.channel1, 1, 1, 0)
        self.fuse2 = nn.Conv2d(num_fea, self.channel2, 1, 1, 0)
        self.fuse = nn.Conv2d(num_fea, num_fea, 1, 1, 0)
        # self.fuse = CALayer(num_fea)

    def forward(self, x):
        x = self.CB(x)
        x1,x2 = torch.split(x,[self.channel1,self.channel2],dim=1)

        x1 = self.convblock(x1)
        x2 = self.convblock(x2)

        A = self.A_att_conv(x1)
        P = torch.cat((x2, A*x1),dim=1)

        B = self.B_att_conv(x2)
        Q = torch.cat((x1, B*x2),dim=1)
        # c = P + Q 
        c = torch.cat((self.fuse1(P),self.fuse2(Q)),dim=1)
        out = self.fuse(c) + x
        out = self.act(self.CB(out))+ out
        # out = self.fuse(self.act(self.CB(out))+ out) 
        return out


class LatticeBlock(nn.Module):
    def __init__(self, nFeat, nDiff, nFeat_slice):
        super(LatticeBlock, self).__init__()

        self.D3 = nFeat
        self.d = nDiff
        self.s = nFeat_slice

        block_0 = []
        block_0.append(nn.Conv2d(nFeat, nFeat, kernel_size=3, padding=1, bias=True))
        block_0.append(nn.LeakyReLU(0.2))
        # block_0.append(nn.Conv2d(nFeat-nDiff, nFeat-nDiff, kernel_size=3, padding=1, bias=True))
        block_0.append(ContextBlock(nFeat, bias=True))
        block_0.append(nn.LeakyReLU(0.2))
        block_0.append(nn.Conv2d(nFeat, nFeat, kernel_size=3, padding=1, bias=True))
        block_0.append(nn.LeakyReLU(0.2))
        self.conv_block0 = nn.Sequential(*block_0)

        # self.fea_ca1 = CC(nFeat)
        # self.x_ca1 = CC(nFeat)

        block_1 = []
        block_1.append(nn.Conv2d(nFeat, nFeat, kernel_size=3, padding=1, bias=True))
        block_1.append(nn.LeakyReLU(0.2))
        # block_1.append(nn.Conv2d(nFeat-nDiff, nFeat-nDiff, kernel_size=3, padding=1, bias=True))
        block_1.append(ContextBlock(nFeat, bias=True))
        block_1.append(nn.LeakyReLU(0.2))
        block_1.append(nn.Conv2d(nFeat, nFeat, kernel_size=3, padding=1, bias=True))
        block_1.append(nn.LeakyReLU(0.2))
        self.conv_block1 = nn.Sequential(*block_1)

        # self.fea_ca2 = CC(nFeat)
        # self.x_ca2 = CC(nFeat)

        self.compress = nn.Conv2d(2 * nFeat, nFeat, kernel_size=1, padding=0, bias=True)
        self.cal = CALayer(nFeat, reduction = 16)

    def forward(self, x):
        # analyse unit
        x_feature_shot = self.conv_block0(x)
        fea_ca1 = 1 # self.fea_ca1(x_feature_shot)
        x_ca1 = 1 # self.x_ca1(x)

        p1z = x + fea_ca1 * x_feature_shot
        q1z = x_feature_shot + x_ca1 * x

        # synthes_unit
        x_feat_long = self.conv_block1(p1z)
        fea_ca2 = 1 # self.fea_ca2(q1z)
        p3z = x_feat_long + fea_ca2 * q1z
        x_ca2 = 1 # self.x_ca2(x_feat_long)
        q3z = q1z + x_ca2 * x_feat_long

        out = torch.cat((p3z, q3z), 1)
        out = self.compress(out)
        out = self.cal(out)

        return out



class STDF_org(nn.Module):
    def __init__(self, in_nc, out_nc, nf, nb, base_ks=3, deform_ks=3):
        """
        Args:
            in_nc: num of input channels.
            out_nc: num of output channels.
            nf: num of channels (filters) of each conv layer.
            nb: num of conv layers.
            deform_ks: size of the deformable kernel.
        """
        super(STDF_org, self).__init__()

        self.nb = nb
        self.in_nc = in_nc
        self.deform_ks = deform_ks
        self.size_dk = deform_ks ** 2
        # print("innc=",in_nc,",,")
        # u-shape backbone
        self.in_conv = nn.Sequential(
            nn.Conv2d(in_nc, nf, base_ks, padding=base_ks//2),
            nn.ReLU(inplace=True)
            )
        for i in range(1, nb):
            setattr(
                self, 'dn_conv{}'.format(i), nn.Sequential(
                    nn.Conv2d(nf, nf, base_ks, stride=2, padding=base_ks//2),
                    nn.ReLU(inplace=True),
                    # nn.Conv2d(nf, nf, base_ks, padding=base_ks//2),
                    # nn.ReLU(inplace=True)
                    )
                )
            setattr(
                self, 'up_conv{}'.format(i), nn.Sequential(
                    nn.Conv2d(2*nf, nf, base_ks, padding=base_ks//2),
                    nn.ReLU(inplace=True),
                    nn.ConvTranspose2d(nf, nf, 4, stride=2, padding=1),
                    nn.ReLU(inplace=True)
                    )
                )
        self.tr_conv = nn.Sequential(
            nn.Conv2d(nf, nf, base_ks, stride=2, padding=base_ks//2),
            nn.ReLU(inplace=True),
            # nn.Conv2d(nf, nf, base_ks, padding=base_ks//2),
            # nn.ReLU(inplace=True),
            nn.ConvTranspose2d(nf, nf, 4, stride=2, padding=1),
            nn.ReLU(inplace=True)
            )
        # self.out_conv = nn.Sequential(
        #     nn.Conv2d(nf, nf, base_ks, padding=base_ks//2),
        #     nn.ReLU(inplace=True)
        #     )

        # regression head
        # why in_nc*3*size_dk?
        #   in_nc: each map use individual offset and mask
        #   2*size_dk: 2 coordinates for each point
        #   1*size_dk: 1 confidence (attention) score for each point
        self.offset_mask = nn.Conv2d(
            nf, in_nc*3*self.size_dk, base_ks, padding=base_ks//2
            )

        # deformable conv
        # notice group=in_nc, i.e., each map use individual offset and mask
        self.deform_conv = ModulatedDeformConv(
            in_nc, out_nc, deform_ks, padding=deform_ks//2, deformable_groups=in_nc
            )

    def forward(self, inputs):
        nb = self.nb
        in_nc = self.in_nc
        n_off_msk = self.deform_ks * self.deform_ks

        # feature extraction (with downsampling)
        out_lst = [self.in_conv(inputs)]  # record feature maps for skip connections
        for i in range(1, nb):
            dn_conv = getattr(self, 'dn_conv{}'.format(i))
            out_lst.append(dn_conv(out_lst[i - 1]))
        # trivial conv
        out = self.tr_conv(out_lst[-1])
        # feature reconstruction (with upsampling)
        for i in range(nb - 1, 0, -1):
            up_conv = getattr(self, 'up_conv{}'.format(i))
            out = up_conv(
                torch.cat([out, out_lst[i]], 1)
                )

        # compute offset and mask
        # offset: conv offset
        # mask: confidence
        # print("OUT SIZE",out.size())
        off_msk = self.offset_mask((out)) # self.out_conv
        off = off_msk[:, :in_nc*2*n_off_msk, ...]
        msk = torch.sigmoid(
            off_msk[:, in_nc*2*n_off_msk:, ...]
            )
        # print("OFF_MSK",off_msk.size(),"OFF",off.size(),"MSK",msk.size())
        # print("INPUS",inputs.size())
        # perform deformable convolutional fusion
        fused_feat = F.relu(
            self.deform_conv(inputs, off, msk), 
            inplace=True
            )

        return fused_feat



class STDF_simple(nn.Module):
    def __init__(self, in_nc, out_nc, nf, nb, base_ks=3, deform_ks=3):
        """
        Args:
            in_nc: num of input channels.
            out_nc: num of output channels.
            nf: num of channels (filters) of each conv layer.
            nb: num of conv layers.
            deform_ks: size of the deformable kernel.
        """
        super(STDF_simple, self).__init__()

        self.nb = nb
        self.in_nc = in_nc
        self.deform_ks = deform_ks
        self.size_dk = deform_ks ** 2
        # u-shape backbone
        self.in_conv = nn.Sequential(
            nn.Conv2d(in_nc, nf, base_ks, padding=base_ks//2),
            nn.LeakyReLU(negative_slope=0.1, inplace=True)
            )

        # for i in range(1, nb):
        #     setattr(
        #         self, 'dn_conv{}'.format(i), nn.Sequential(
        #             nn.Conv2d(nf, nf, base_ks, stride=2, padding=base_ks//2),
        #             nn.ReLU(inplace=True),
        #             # nn.Conv2d(nf, nf, base_ks, padding=base_ks//2),
        #             # nn.ReLU(inplace=True)
        #             )
        #         )
        #     setattr(
        #         self, 'up_conv{}'.format(i), nn.Sequential(
        #             nn.Conv2d(2*nf, nf, base_ks, padding=base_ks//2),
        #             nn.ReLU(inplace=True),
        #             nn.ConvTranspose2d(nf, nf, 4, stride=2, padding=1),
        #             nn.ReLU(inplace=True)
        #             )
        #         )
        # self.tr_conv = nn.Sequential(
        #     nn.Conv2d(nf, nf, base_ks, stride=2, padding=base_ks//2),
        #     nn.ReLU(inplace=True),
        #     # nn.Conv2d(nf, nf, base_ks, padding=base_ks//2),
        #     # nn.ReLU(inplace=True),
        #     nn.ConvTranspose2d(nf, nf, 4, stride=2, padding=1),
        #     nn.ReLU(inplace=True)
        #     )
        # self.out_conv = nn.Sequential(
        #     nn.Conv2d(nf, nf, base_ks, padding=base_ks//2),
        #     nn.ReLU(inplace=True)
        #     )

        # regression head
        # why in_nc*3*size_dk?
        #   in_nc: each map use individual offset and mask
        #   2*size_dk: 2 coordinates for each point
        #   1*size_dk: 1 confidence (attention) score for each point
        self.offset_mask = nn.Conv2d(
            nf, in_nc*3*self.size_dk, base_ks, padding=base_ks//2
            )

        # deformable conv
        # notice group=in_nc, i.e., each map use individual offset and mask
        self.deform_conv = ModulatedDeformConv(
            in_nc, out_nc, deform_ks, padding=deform_ks//2, deformable_groups=in_nc
            )

    def forward(self, inputs):
        nb = self.nb
        in_nc = self.in_nc
        n_off_msk = self.deform_ks * self.deform_ks

        # feature extraction (with downsampling)
        inputs = self.in_conv(inputs)
        # inputs = self.tr_conv(inputs) + inputs
        # inputs = self.RCAB(inputs)
        # out = self.ResBlock_3d(self.RCAB(out))
        # out_in = self.tra_conv(out)

        # compute offset and mask
        # offset: conv offset
        # mask: confidence
        # print("OUT SIZE",out.size())
        off_msk = self.offset_mask((inputs)) # self.out_conv
        off = off_msk[:, :in_nc*2*n_off_msk, ...]
        msk = torch.sigmoid(
            off_msk[:, in_nc*2*n_off_msk:, ...]
            )
        # print("OFF_MSK",off_msk.size(),"OFF",off.size(),"MSK",msk.size())
        # print("INPUS",inputs.size())
        # perform deformable convolutional fusion
        fused_feat = F.relu(
            self.deform_conv(inputs, off, msk), 
            inplace=True
            )

        return fused_feat



# ==========
# TGDA network
# ==========
class TGAF(nn.Module):
    def __init__(self,opts_dict):
        super(TGAF,self).__init__()
        self.radius = 3
        self.input_len = 2 * self.radius + 1
        self.center = self.input_len // 2
        self.color = opts_dict['qenet']['out_nc']

        self.inputconv = nn.Sequential(nn.Conv2d(opts_dict['stdf']['in_nc'] * self.radius, opts_dict['stdf']['out_nc'], 3, padding=3//2),
            nn.LeakyReLU(negative_slope=0.1, inplace=True))
        self.Unet = Unet(
            in_nc=opts_dict['stdf']['out_nc'],   
            out_nc=opts_dict['stdf']['out_nc'], 
            nf=opts_dict['stdf']['nf'], 
            nb=opts_dict['stdf']['nb'], 
            base_ks = 3,
            )


        self.deform_block_1 = Deform_block(
            in_nc=opts_dict['stdf']['in_nc'] * self.radius,  
            out_nc=opts_dict['stdf']['out_nc'], 
            nf=opts_dict['stdf']['nf'], 
            base_ks = 3,
            deform_ks= 1 
        )

        self.deform_block_3 = Deform_block(
            in_nc=opts_dict['stdf']['in_nc'] * self.radius,  
            out_nc=opts_dict['stdf']['out_nc'], 
            nf=opts_dict['stdf']['nf'], 
            base_ks = 3,
            deform_ks= 3 
        )

        self.deform_block_5 = Deform_block(
            in_nc=opts_dict['stdf']['in_nc'] * self.radius,  
            out_nc=opts_dict['stdf']['out_nc'], 
            nf=opts_dict['stdf']['nf'], 
            base_ks = 3,
            deform_ks= 5 
        )       

      
        self.RCAB = RCAB(opts_dict['stdf']['out_nc'])
        self.fuse = nn.Sequential(
            nn.Conv2d(opts_dict['stdf']['out_nc']*2, opts_dict['stdf']['out_nc'], 3, stride=1, padding=3//2),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
        )

        self.wpnet = STDF_simple(
            in_nc=opts_dict['stdf']['out_nc'], 
            out_nc=opts_dict['stdf']['out_nc'] , 
            nf=opts_dict['stdf']['nf'], 
            nb=opts_dict['stdf']['nb'], 
            base_ks = 3,
            deform_ks = 1
            )

        self.qenetname = opts_dict['qenet']['netname']
        if opts_dict['qenet']['netname']=='default':
            att = True
            if not opts_dict['qenet'].__contains__('att') or opts_dict['qenet']['att']==False:
                att = False
            attname = 'None'
            if att:
                attname = opts_dict['qenet']['attname']

            self.qenet = PlainCNN(
                in_nc=opts_dict['qenet']['in_nc'],  
                nf=opts_dict['qenet']['nf'], 
                nb=opts_dict['qenet']['nb'], 
                out_nc=opts_dict['qenet']['out_nc'],
                Att = att,
                Attname = attname,
                )

    # x is the input reference frames
    # y is the preceding hidden state feature
    def forward(self,x):
        x = x.contiguous()
        G_1 = x[:,self.center-1:self.center+2,:,:].contiguous()   #  self.center = 3
        G_2 = x[:,1::2,:,:].contiguous()
        G_3 = x[:,0::3,:,:].contiguous()
        
        # [B F H W]
        out_1 = self.Unet(self.inputconv(G_1))
        out_1 = self.deform_block_1(G_1, out_1)
        out_2 = self.Unet(self.inputconv(G_2))
        out_2 = self.deform_block_3(G_2, out_2)
        out_3 = self.Unet(self.inputconv(G_3))
        out_3 = self.deform_block_5(G_3, out_3)

        # [B 14 H W]
        # out = out_1 + out_2 + out_3
        out_f1 = self.fuse(torch.cat((out_1,out_2),1))
        out_f2 = self.fuse(torch.cat((out_2,out_3),1))
        out = self.RCAB(self.fuse(torch.cat((out_f1,out_f2),1)))
        # [B F H W]
        out = self.wpnet(out)
        # N, C, H, W = out.shape

        final_out = self.qenet(out) + x[:, [self.radius + i*(2*self.radius+1) for i in range(self.color)], ...]
        return final_out


