import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import DeformConv2d
import functools
from .base_model import OFAE, SKU_Net
from .mdfi_base_model import FER, SKMU
# from .base_model import OFAE
import numpy as np
# from architectures import SFT_Net


def generate_it(x, t=0, nf=3, f=7):
    index = np.array([t - nf // 2 + i for i in range(nf)])
    index = np.clip(index, 0, f-1).tolist()
    it = x[:, index, :, :]
    return it

class STFF(nn.Module):
    def __init__(self, in_nc, out_nc, nf, base_ks=3, deform_ks=3):
        """
        Args:
            in_nc: num of input channels.
            out_nc: num of output channels.
            nf: num of channels (filters) of each conv layer.
            nb: num of conv layers.
            deform_ks: size of the deformable kernel.
        """
        super(STFF, self).__init__()

        self.in_nc = in_nc
        self.deform_ks = deform_ks
        self.size_dk = deform_ks ** 2

        self.in_conv = nn.Sequential(
            nn.Conv2d(in_nc, nf, base_ks, padding=base_ks // 2),
            nn.LeakyReLU(negative_slope=0.1, inplace=True)
        )

        self.unet = SKU_Net(nf=nf)
    
        self.out_conv = nn.Sequential(
            nn.Conv2d(nf, nf, base_ks, padding=base_ks // 2),
            nn.LeakyReLU(negative_slope=0.1, inplace=True)
        )
        self.offset_mask = nn.Conv2d(
            nf, in_nc * 3 * self.size_dk, base_ks, padding=base_ks // 2
        )

        self.deform_conv = DeformConv2d(
            in_nc, out_nc, deform_ks, padding=deform_ks // 2
        )

    def forward(self, inputs):
        out = self.in_conv(inputs)
        out = self.unet(out)
        off_msk = self.offset_mask(self.out_conv(out))
        off = off_msk[:, :self.in_nc * 2 * self.size_dk, ...]
        msk = torch.sigmoid(
            off_msk[:, self.in_nc * 2 * self.size_dk:, ...]
        )
        fused_feat = F.relu(
            self.deform_conv(inputs, off, msk),
            inplace=True
        )
        return fused_feat


class SFTLayer(nn.Module):
    def __init__(self, nf=32):
        super(SFTLayer, self).__init__()
        self.SFT_scale_conv0 = nn.Conv2d(nf, nf, 1)
        self.SFT_scale_conv1 = nn.Conv2d(nf, nf, 1)
        self.SFT_shift_conv0 = nn.Conv2d(nf, nf, 1)
        self.SFT_shift_conv1 = nn.Conv2d(nf, nf, 1)

    def forward(self, x):
        scale = self.SFT_scale_conv1(F.leaky_relu(self.SFT_scale_conv0(x[1]), 0.1, inplace=True))
        shift = self.SFT_shift_conv1(F.leaky_relu(self.SFT_shift_conv0(x[1]), 0.1, inplace=True))
        return x[0] * (scale + 1) + shift

class ResBlock_SFT(nn.Module):
    def __init__(self, nf=32):
        super(ResBlock_SFT, self).__init__()
        self.sft0 = SFTLayer(nf)
        self.conv0 = nn.Conv2d(nf, nf, 3, 1, 1)
        self.sft1 = SFTLayer(nf)
        self.conv1 = nn.Conv2d(nf, nf, 3, 1, 1)

    def forward(self, x):
        fea = self.sft0(x)
        fea = F.relu(self.conv0(fea), inplace=True)
        fea = self.sft1((fea, x[1]))
        fea = self.conv1(fea)
        return (x[0] + fea, x[1])

    
class SFT_Net(nn.Module):
    def __init__(self, nf=32, num_sft=16):
        super(SFT_Net, self).__init__()
        # Đầu vào lq_data: (1, 1, Width, Height) -> fea: 32 kênh
        self.lq_conv = nn.Conv2d(1, nf, 3, 1, 1)  # Từ 1 kênh thành 32 kênh
        # CondNet: pred_data (1 kênh) -> cond (32 kênh)
        self.pred_conv = nn.Sequential(
            nn.Conv2d(1, nf, 3, 1, 1), nn.LeakyReLU(0.1, True),  # Từ 1 kênh lên 64
            nn.Conv2d(nf, nf*2, 3, 1, 1), nn.LeakyReLU(0.1, True),  # Lên 128
            nn.Conv2d(nf*2, nf, 1), nn.LeakyReLU(0.1, True)  # Giảm xuống 32 kênh
        )
        # Nhánh SFT với ResBlock_SFT
        sft_branch = []
        for i in range(num_sft):  # 16 khối ResBlock_SFT
            sft_branch.append(ResBlock_SFT(nf=nf))
        self.sft_branch = nn.Sequential(*sft_branch)
        
        self.out_conv = nn.Conv2d(nf, nf, 3, 1, 1)  # Đầu ra 32 kênh, giữ kích thước

        

    def forward(self, lq_data, pred_data):
        # Tạo cond từ pred_data
        pred = self.pred_conv(pred_data)
        
        # Tạo fea từ lq_data
        lq = self.lq_conv(lq_data)
        
        # Nhánh SFT
        res = self.sft_branch((lq, pred))
        out = self.out_conv(res[0])
        # Kết hợp residual và trả về fea đã xử lý
        out = lq + out # Chỉ lấy fea từ tuple (fea, cond)
        return out

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


class CVQE(nn.Module):
    def __init__(self, in_nc=7, nf=32, out_nc=64, nb=5, cpu_cache_length=20, num_sft=16):
        super(CVQE, self).__init__()
        self.out_nc = out_nc
        self.cpu_cache_length = cpu_cache_length
        # Lưu ý: lq_data có 7 kênh, pred_data có 1 kênh nên tổng kênh ban đầu = 7 + 1 = 8.
        # STFF_1 sẽ nhận vào lq_data (7 kênh) và pred_data (1 kênh) riêng biệt.
        self.sft = SFT_Net(nf=nf, num_sft=num_sft)
        self.stff = STFF(in_nc=in_nc+nf, out_nc=out_nc, nf=nf, deform_ks=3)
        self.first_backward = STFF(
            in_nc=2*out_nc,
            out_nc=out_nc,
            nf=nf,
            deform_ks=1
        )
        self.first_forward = STFF(
            in_nc=3*out_nc,
            out_nc=out_nc,
            nf=nf,
            deform_ks=1
        )
        self.second_backward = STFF(
            in_nc=3*out_nc,
            out_nc=out_nc,
            nf=nf,
            deform_ks=1
        )
        self.second_forward = STFF(
            in_nc=3*out_nc,
            out_nc=out_nc,
            nf=nf,
            deform_ks=1
        )
        self.ofae = OFAE(2*self.out_nc+nf, self.out_nc, connection=True)
        self.qenet = PlainCNN(in_nc=4*self.out_nc, nf=self.out_nc, nb=nb, out_nc=1)

    def forward(self, inputs, pred_d):
        """
        Args:
            inputs: tensor lq_data với kích thước [B, frame, H, W]
            pred_d: tensor pred_data với kích thước [B, frame, H, W]
        """
        n, t, h, w = inputs.size()
        self.cpu_cache = True if t > self.cpu_cache_length else False

        ##################################### First Backward Propagation ############################################
        First_Backward_List = []
        SFT_List = []
        feat = inputs.new_zeros(n, self.out_nc, h, w)
        # Vòng lặp duyệt các frame theo thứ tự ngược lại
        for i in range(t - 1, -1, -1):
            # Lấy ra 7 frame của lq_data xung quanh frame i (sử dụng hàm generate_it)
            out = generate_it(inputs, i, 7, t)  # out có kích thước [B, 7, H, W]
            pred_frame = pred_d[:, i:i+1, :, :]  # pred_frame có kích thước [B, 1, H, W]
            sft_out = self.sft(out[:, out.shape[1]//2: out.shape[1]//2 + 1, :, :], pred_frame)
            out = torch.cat((out, sft_out), dim=1)
            out = self.stff(out)
            feat = torch.cat([out, feat], dim=1)
            feat = self.first_backward(feat)
            feat = self.ofae(torch.cat([out, feat, sft_out], 1)) + out
            if self.cpu_cache:
                First_Backward_List.append(feat.cpu())
                SFT_List.append(sft_out.cpu())
                torch.cuda.empty_cache()
            else:
                First_Backward_List.append(feat)
                SFT_List.append(sft_out)
        First_Backward_List = First_Backward_List[::-1]
#####################################First Forward Propagation##############################################
        First_Forward_List = []
        feat = inputs.new_zeros(n, self.out_nc, h, w)
        for i in range(0, t):
            future = First_Backward_List[i] if i == t - 1 else First_Backward_List[i + 1]
            present = First_Backward_List[i]
            sft_out = SFT_List[i]
            if self.cpu_cache:
                present = present.cuda()
                future = future.cuda()
                sft_out = sft_out.cuda()
            feat = torch.cat([feat,present,future], dim=1)
            feat = self.first_forward(feat)
            
            feat = self.ofae(torch.cat([present, feat, sft_out], 1)) + present
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
            sft_out = SFT_List[i]
            if self.cpu_cache:
                present = present.cuda()
                future = future.cuda()
                sft_out = sft_out.cuda()
            feat = torch.cat([feat,present,future], dim=1)
            feat = self.second_backward(feat)
            feat = self.ofae(torch.cat([present, feat, sft_out], 1))  + present
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
            sft_out = SFT_List[i]
            if self.cpu_cache:
                present = present.cuda()
                future = future.cuda()
                sft_out = sft_out.cuda()

            feat = torch.cat([feat,present,future], dim=1)
            feat = self.second_forward(feat)
            feat = self.ofae(torch.cat([present, feat, sft_out], 1))+ present
            if self.cpu_cache:
                out = self.qenet(torch.cat([First_Backward_List[i].cuda(), First_Forward_List[i].cuda(), Second_Backward_List[i].cuda(), feat],dim=1)) + inputs[:, i:i + 1, :, :]
                Enhanced.append(out.cpu())
                torch.cuda.empty_cache()
            else:
                out = self.qenet(torch.cat([First_Backward_List[i], First_Forward_List[i], Second_Backward_List[i], feat],dim=1)) + inputs[:, i:i + 1, :, :]
                Enhanced.append(out)

        return torch.stack(Enhanced, dim=1)

# class OVQE(nn.Module):
#     def __init__(self, sft_net_model=SFT_Net(), ovqe_model=OVQE1()):
#         super(OVQE, self).__init__()
#         self.video_frame_processor = Prior_Ext(sft_net_model)  # Prior_Net xử lý khung hình
#         self.ovqe = ovqe_model  # Mô hình OVQE xử lý video chất lượng cao

#     def forward(self, lq_data, pred_data):

#         # Bước 1: Xử lý từng khung hình từ 2 video bằng Prior_Net
#         processed_frames = self.video_frame_processor(lq_data, pred_data)
        
#         # Bước 2: Đưa kết quả qua mô hình OVQE để nâng cao chất lượng
#         enhanced_video = self.ovqe(processed_frames)
        
#         return enhanced_video

if __name__ == "__main__":
    torch.cuda.set_device(0)
    net = CVQE().cuda()
    from thop import profile
    with torch.no_grad():

        input = torch.randn(1, 15, 32, 32).cuda()
        flops, params = profile(net, inputs=(input, input))
        total = sum([param.nelement() for param in net.parameters()])
        print('   Number of params: %.2fM' % (total / 1e6))
        print('   Number of FLOPs: %.2fGFLOPs' % (flops / 1e9))
