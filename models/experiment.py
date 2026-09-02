import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import DeformConv2d
import functools
from base_model import OFAE, SKU_Net
import numpy as np


def generate_it(x, t=0, nf=3, f=7):
    index = np.array([t - nf // 2 + i for i in range(nf)])
    index = np.clip(index, 0, f-1).tolist()
    it = x[:, index, :, :]
    return it

class BottleneckDCN(nn.Module):
    """
    Bottleneck wrapper around torchvision.ops.DeformConv2d
    (which does NOT support deformable_groups).
    """
    def __init__(self, in_ch, out_ch, kernel_size=3, bottleneck_mid=None, bottleneck_ratio=2):
        super().__init__()
        if bottleneck_mid is None:
            mid = max(8, in_ch // bottleneck_ratio)
        else:
            mid = bottleneck_mid
        self.mid = mid
        self.k = kernel_size

        # reduce input channels
        self.pre = nn.Sequential(
            nn.Conv2d(in_ch, mid, kernel_size=1, bias=False),
            nn.ReLU(inplace=False)
        )
        # DeformConv2d in torchvision only takes (x, offset, mask=None)
        self.deform = DeformConv2d(mid, mid, kernel_size, padding=kernel_size//2, bias=False)

        # expand back to out_ch
        self.post = nn.Sequential(
            nn.Conv2d(mid, out_ch, kernel_size=1, bias=False),
            nn.ReLU(inplace=False)
        )

    def forward_from_feats(self, x_pre, offset, mask=None):
        try:
            # some versions accept mask
            if mask is not None:
                y = self.deform(x_pre, offset, mask)
            else:
                y = self.deform(x_pre, offset)
        except TypeError:
            # older torchvision DeformConv2d may only accept (x, offset)
            y = self.deform(x_pre, offset)
        return self.post(y)

    def reduce(self, x):
        return self.pre(x)

    def forward(self, x, offset, mask=None):
        x_pre = self.reduce(x)
        return self.forward_from_feats(x_pre, offset, mask)



# === Updated STFF using BottleneckDCN ===
class STFF(nn.Module):
    def __init__(self, in_nc, out_nc, nf, base_ks=3, deform_ks=3, bottleneck_ratio=2):
        super().__init__()
        self.in_nc = in_nc
        self.deform_ks = deform_ks
        self.size_dk = deform_ks ** 2
        self.nf = nf

        self.in_conv = nn.Sequential(
            nn.Conv2d(in_nc, nf, base_ks, padding=base_ks // 2),
            nn.LeakyReLU(negative_slope=0.1, inplace=False)
        )
        self.unet = SKU_Net(nf=nf)

        mid = max(8, nf // bottleneck_ratio)
        self.mid = mid

        self.offset_feat_conv = nn.Sequential(
            nn.Conv2d(nf, mid, base_ks, padding=base_ks // 2),
            nn.LeakyReLU(negative_slope=0.1, inplace=False)
        )

        # Only (2*k^2 + k^2) channels = 3*k^2
        self.offset_mask_conv = nn.Conv2d(mid, 3 * self.size_dk, base_ks, padding=base_ks // 2)

        self.bottleneck = BottleneckDCN(in_nc, out_nc, kernel_size=deform_ks,
                                        bottleneck_mid=mid, bottleneck_ratio=1)

        self.out_conv = nn.Sequential(
            nn.Conv2d(nf, nf, base_ks, padding=base_ks // 2),
            nn.LeakyReLU(negative_slope=0.1, inplace=False)
        )

    def forward(self, inputs):
        feat = self.in_conv(inputs)
        feat = self.unet(feat)
        feat_out = self.out_conv(feat)

        mid_feat = self.offset_feat_conv(feat_out)
        off_msk = self.offset_mask_conv(mid_feat)

        k2 = self.size_dk
        off = off_msk[:, :2 * k2, ...]
        msk = torch.sigmoid(off_msk[:, 2 * k2:3 * k2, ...])

        pre = self.bottleneck.reduce(inputs)

        try:
            fused = F.relu(self.bottleneck.forward_from_feats(pre, off, msk), inplace=False)
        except Exception:
            fused = F.relu(self.bottleneck.forward_from_feats(pre, off, None), inplace=False)

        return fused




class PlainCNN(nn.Module):
    def make_layer(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block())
        return nn.Sequential(*layers)
    def __init__(self, in_nc=64, nf=64, nb=5, out_nc=3, base_ks=3):
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
            nn.ReLU(inplace=True)
        )
        self.reconstruct = self.make_layer(functools.partial(OFAE, nf, nf), nb)
        self.out_conv = nn.Conv2d(nf, out_nc, base_ks, padding=1)

    def forward(self, inputs):
        inputs = self.in_conv(inputs)
        inputs = self.reconstruct(inputs)
        inputs = self.out_conv(inputs)
        return inputs

class OVQE(nn.Module):
    def __init__(self, in_nc=7, nf=32, out_nc=64, nb=5, cpu_cache_length=20):
        super(OVQE, self).__init__()
        self.out_nc = out_nc
        self.cpu_cache_length = cpu_cache_length
        self.stff = STFF(in_nc=in_nc, out_nc=out_nc, nf=nf, deform_ks=3, bottleneck_ratio=2)
        self.first_backward = STFF(
            in_nc=2*out_nc,
            out_nc=out_nc,
            nf=nf,
            deform_ks=1,
            bottleneck_ratio=2
        )
        self.first_forward = STFF(
            in_nc=3*out_nc,
            out_nc=out_nc,
            nf=nf,
            deform_ks=1,
            bottleneck_ratio=2
        )
        self.second_backward = STFF(
            in_nc=3*out_nc,
            out_nc=out_nc,
            nf=nf,
            deform_ks=1,
            bottleneck_ratio=2
        )
        self.second_forward = STFF(
            in_nc=3*out_nc,
            out_nc=out_nc,
            nf=nf,
            deform_ks=1,
            bottleneck_ratio=2
        )
        self.ofae = OFAE(2*self.out_nc,self.out_nc, connection=True)
        self.qenet = PlainCNN(in_nc=4*self.out_nc,nf=self.out_nc,nb=nb,out_nc=1)


    def forward(self, inputs):
        n, t, h, w = inputs.size()
        if t > self.cpu_cache_length:
            self.cpu_cache = True
        else:
            self.cpu_cache = False
#####################################First Backward Propagation############################################
        First_Backward_List = []
        feat = inputs.new_zeros(n, self.out_nc, h, w)
        for i in range(t - 1, -1, -1):
            out = generate_it(inputs, i, 7, t)
            out = self.stff(out)
            feat = torch.cat([out,feat], dim=1)
            feat = self.first_backward(feat)
            feat = self.ofae(torch.cat([out,feat], 1)) + out
            if self.cpu_cache:
                First_Backward_List.append(feat.cpu())
                torch.cuda.empty_cache()
            else:
                First_Backward_List.append(feat)
        First_Backward_List = First_Backward_List[::-1]
#####################################First Forward Propagation##############################################
        First_Forward_List = []
        feat = inputs.new_zeros(n, self.out_nc, h, w)
        for i in range(0, t):
            future = First_Backward_List[i] if i == t - 1 else First_Backward_List[i + 1]
            present = First_Backward_List[i]
            if self.cpu_cache:
                present = present.cuda()
                future = future.cuda()

            feat = torch.cat([feat,present,future], dim=1)
            feat = self.first_forward(feat)
            feat = self.ofae(torch.cat([present, feat], 1)) + present
            if self.cpu_cache:
                First_Forward_List.append(feat.cpu())
                torch.cuda.empty_cache()
            else:
                First_Forward_List.append(feat)
#####################################Second Backward Propagation##########################################
        Second_Backward_List = []
        feat = inputs.new_zeros(n, self.out_nc, h, w)
        for i in range(t - 1, -1, -1):
            future = First_Forward_List[i] if i == 0 else First_Forward_List[i - 1]
            present = First_Forward_List[i]
            if self.cpu_cache:
                present = present.cuda()
                future = future.cuda()

            feat = torch.cat([feat,present,future], dim=1)
            feat = self.second_backward(feat)
            feat = self.ofae(torch.cat([present, feat], 1))  + present
            if self.cpu_cache:
                Second_Backward_List.append(feat.cpu())
                torch.cuda.empty_cache()
            else:
                Second_Backward_List.append(feat)
        Second_Backward_List = Second_Backward_List[::-1]
#####################################Second Forward Propagation############################################
        Enhanced = []
        feat = inputs.new_zeros(n, self.out_nc, h, w)
        for i in range(0, t):
            future = Second_Backward_List[i] if i == t - 1 else Second_Backward_List[i + 1]
            present = Second_Backward_List[i]
            if self.cpu_cache:
                present = present.cuda()
                future = future.cuda()

            feat = torch.cat([feat,present,future], dim=1)
            feat = self.second_forward(feat)
            feat = self.ofae(torch.cat([present, feat], 1))+ present
            if self.cpu_cache:
                out = self.qenet(torch.cat([First_Backward_List[i].cuda(), First_Forward_List[i].cuda(), Second_Backward_List[i].cuda(), feat],dim=1)) + inputs[:, i:i + 1, :, :]
                Enhanced.append(out.cpu())
                torch.cuda.empty_cache()
            else:
                out = self.qenet(torch.cat([First_Backward_List[i], First_Forward_List[i], Second_Backward_List[i], feat],dim=1)) + inputs[:, i:i + 1, :, :]
                Enhanced.append(out)

        return torch.stack(Enhanced, dim=1)


if __name__ == "__main__":
    torch.cuda.set_device(0)
    net = OVQE().cuda()
    from thop import profile
    with torch.no_grad():

        input = torch.randn(1, 7, 1920, 1080).cuda()
        flops, params = profile(net, inputs=(input, ))
        total = sum([param.nelement() for param in net.parameters()])
        print('   Number of params: %.2fM' % (total / 1e6))
        print('   Number of FLOPs: %.2fGFLOPs' % (flops / 1e9))