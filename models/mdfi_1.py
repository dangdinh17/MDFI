import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import DeformConv2d
import functools
from .base_model import OFAE, SKU_Net
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
    def __init__(self):
        super(SFTLayer, self).__init__()
        self.SFT_scale_conv0 = nn.Conv2d(32, 32, 1)
        self.SFT_scale_conv1 = nn.Conv2d(32, 32, 1)
        self.SFT_shift_conv0 = nn.Conv2d(32, 32, 1)
        self.SFT_shift_conv1 = nn.Conv2d(32, 32, 1)

    def forward(self, x):
        scale = self.SFT_scale_conv1(F.leaky_relu(self.SFT_scale_conv0(x[1]), 0.1, inplace=True))
        shift = self.SFT_shift_conv1(F.leaky_relu(self.SFT_shift_conv0(x[1]), 0.1, inplace=True))
        return x[0] * (scale + 1) + shift

class ResBlock_SFT(nn.Module):
    def __init__(self):
        super(ResBlock_SFT, self).__init__()
        self.sft0 = SFTLayer()
        self.conv0 = nn.Conv2d(32, 32, 3, 1, 1)
        self.sft1 = SFTLayer()
        self.conv1 = nn.Conv2d(32, 32, 3, 1, 1)

    def forward(self, x):
        fea = self.sft0(x)
        fea = F.relu(self.conv0(fea), inplace=True)
        fea = self.sft1((fea, x[1]))
        fea = self.conv1(fea)
        return (x[0] + fea, x[1])

    
class SFT_Net(nn.Module):
    def __init__(self, num_sft=16):
        super(SFT_Net, self).__init__()
        # Đầu vào lq_data: (1, 1, Width, Height) -> fea: 32 kênh
        self.conv0 = nn.Conv2d(1, 32, 3, 1, 1)  # Từ 1 kênh thành 32 kênh

        # Nhánh SFT với ResBlock_SFT
        sft_branch = []
        for i in range(num_sft):  # 16 khối ResBlock_SFT
            sft_branch.append(ResBlock_SFT())
        sft_branch.append(SFTLayer())  # Thêm một SFTLayer cuối
        sft_branch.append(nn.Conv2d(32, 32, 3, 1, 1))  # Đầu ra 32 kênh, giữ kích thước
        self.sft_branch = nn.Sequential(*sft_branch)

        # CondNet: pred_data (1 kênh) -> cond (32 kênh)
        self.CondNet = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, 1), nn.LeakyReLU(0.1, True),  # Từ 1 kênh lên 64
            nn.Conv2d(64, 128, 3, 1, 1), nn.LeakyReLU(0.1, True),  # Lên 128
            nn.Conv2d(128, 32, 1), nn.LeakyReLU(0.1, True)  # Giảm xuống 32 kênh
        )

    def forward(self, x):
        # x[0]: lq_data (fea), x[1]: pred_data (cond)
        lq_data, pred_data = x[0], x[1]
        
        # Tạo cond từ pred_data
        cond = self.CondNet(pred_data)
        
        # Tạo fea từ lq_data
        fea = self.conv0(lq_data)
        
        # Nhánh SFT
        res = self.sft_branch((fea, cond))
        
        # Kết hợp residual và trả về fea đã xử lý
        out = fea + res[0]  # Chỉ lấy fea từ tuple (fea, cond)
        return out

class STFF_1(nn.Module):
    def __init__(self, lq_nc, out_nc, nf, base_ks=3, deform_ks=3, num_sft=16):
        """
        Args:
            lq_nc: số kênh của tensor lq_data (ví dụ: số frame)
            out_nc: số kênh đầu ra.
            nf: số filters của các lớp conv trung gian.
            base_ks: kích thước kernel của các lớp conv cơ bản.
            deform_ks: kích thước kernel của deformable conv.
        """
        super(STFF_1, self).__init__()
        self.lq_nc = lq_nc
        self.pred_nc = 32  # Đã sửa lại số kênh của cond thành 32
        # Sau khi concat: tổng số kênh = lq_nc + 32
        self.in_nc = lq_nc + self.pred_nc
        self.deform_ks = deform_ks
        self.size_dk = deform_ks ** 2

        # Module trích xuất đặc trưng từ pred_data sử dụng SFT_Net
        self.pred_feature_extractor = SFT_Net(num_sft=num_sft)

        # Lớp conv ban đầu nhận đầu vào đã được concat
        self.in_conv = nn.Sequential(
            nn.Conv2d(self.in_nc, nf, base_ks, padding=base_ks // 2),
            nn.LeakyReLU(negative_slope=0.1, inplace=True)
        )

        self.unet = SKU_Net(nf=nf)

        self.out_conv = nn.Sequential(
            nn.Conv2d(nf, nf, base_ks, padding=base_ks // 2),
            nn.LeakyReLU(negative_slope=0.1, inplace=True)
        )

        # Lớp offset_mask tạo ra offset và mask cho deformable conv
        self.offset_mask = nn.Conv2d(
            nf, self.in_nc * 3 * self.size_dk, base_ks, padding=base_ks // 2
        )

        # Lớp deformable conv nhận đầu vào với số kênh = in_nc (đã được concat)
        self.deform_conv = DeformConv2d(
            self.in_nc, out_nc, deform_ks, padding=deform_ks // 2
        )

    def forward(self, lq_data, pred_data):
        """
        Args:
            lq_data: tensor có kích thước [batch, lq_nc, height, width]
            pred_data: tensor có kích thước [batch, 1, height, width]
        Returns:
            fused feature tensor
        """
        # Lấy frame ở trung tâm của pred_data để làm fea
        # # Điều này có thể được thực hiện bằng cách lấy ra phần tử ở giữa của pred_data (frame trung tâm)
        # pred_center = pred_data[:, :, pred_data.shape[2] // 2, pred_data.shape[3] // 2].unsqueeze(2).unsqueeze(3)
        lq_center = lq_data[:, lq_data.shape[1]//2: lq_data.shape[1]//2 + 1, :, :]
        # Trích xuất đặc trưng từ pred_data sử dụng SFT_Net
        pred_feat = self.pred_feature_extractor((lq_center, pred_data))  # kết quả: [B, 32, H, W]
        
        # Nối (concat) theo chiều kênh: [B, lq_nc + 32, H, W]
        fused_input = torch.cat([lq_data, pred_feat], dim=1)
        
        # Qua các lớp của STFF
        out = self.in_conv(fused_input)
        out = self.unet(out)
        off_msk = self.offset_mask(self.out_conv(out))
        off = off_msk[:, :self.in_nc * 2 * self.size_dk, ...]
        msk = torch.sigmoid(off_msk[:, self.in_nc * 2 * self.size_dk:, ...])
        fused_feat = F.relu(
            self.deform_conv(fused_input, off, msk),
            inplace=True
        )
        return fused_feat
    
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


class MDFI_1(nn.Module):
    def __init__(self, in_nc=7, nf=32, out_nc=64, nb=5, cpu_cache_length=20, num_sft=16):
        super(MDFI_1, self).__init__()
        self.out_nc = out_nc
        self.cpu_cache_length = cpu_cache_length
        # Lưu ý: lq_data có 7 kênh, pred_data có 1 kênh nên tổng kênh ban đầu = 7 + 1 = 8.
        # STFF_1 sẽ nhận vào lq_data (7 kênh) và pred_data (1 kênh) riêng biệt.
        # self.stff = STFF_1(lq_nc=in_nc, out_nc=out_nc, nf=nf, deform_ks=3, num_sft=num_sft)
        # self.first_backward = STFF(
        #     in_nc=2*out_nc,
        #     out_nc=out_nc,
        #     nf=nf,
        #     deform_ks=1
        # )
        # self.first_forward = STFF(
        #     in_nc=3*out_nc,
        #     out_nc=out_nc,
        #     nf=nf,
        #     deform_ks=1
        # )
        # self.second_backward = STFF(
        #     in_nc=3*out_nc,
        #     out_nc=out_nc,
        #     nf=nf,
        #     deform_ks=1
        # )
        # self.second_forward = STFF(
        #     in_nc=3*out_nc,
        #     out_nc=out_nc,
        #     nf=nf,
        #     deform_ks=1
        # )
        # self.ofae = OFAE(2*self.out_nc, self.out_nc, connection=True)
        self.qenet = PlainCNN(in_nc=in_nc+1, nf=self.out_nc, nb=nb, out_nc=1)

    def forward(self, inputs, pred_d):
        """
        Args:
            inputs: tensor lq_data với kích thước [B, frame, H, W]
            pred_d: tensor pred_data với kích thước [B, frame, H, W]
        """
        n, t, h, w = inputs.size()
        self.cpu_cache = True if t > self.cpu_cache_length else False

        ##################################### First Backward Propagation ############################################
#         First_Backward_List = []
#         feat = inputs.new_zeros(n, self.out_nc, h, w)
#         # Vòng lặp duyệt các frame theo thứ tự ngược lại
#         for i in range(t - 1, -1, -1):
#             # Lấy ra 7 frame của lq_data xung quanh frame i (sử dụng hàm generate_it)
#             out = generate_it(inputs, i, 7, t)  # out có kích thước [B, 7, H, W]
#             # Lấy ra frame thứ i của pred_data (dùng slicing i:i+1 để giữ chiều channel)
#             pred_frame = pred_d[:, i:i+1, :, :]  # pred_frame có kích thước [B, 1, H, W]
#             # Gọi STFF_1 với 2 đầu vào: out và pred_frame
#             out = self.stff(out, pred_frame)
#             # Tiến hành các bước xử lý như ban đầu
#             feat = torch.cat([out, feat], dim=1)
#             feat = self.first_backward(feat)
#             feat = self.ofae(torch.cat([out, feat], 1)) + out
#             if self.cpu_cache:
#                 First_Backward_List.append(feat.cpu())
#                 torch.cuda.empty_cache()
#             else:
#                 First_Backward_List.append(feat)
#         First_Backward_List = First_Backward_List[::-1]
# #####################################First Forward Propagation##############################################
#         First_Forward_List = []
#         feat = inputs.new_zeros(n, self.out_nc, h, w)
#         for i in range(0, t):
#             future = First_Backward_List[i] if i == t - 1 else First_Backward_List[i + 1]
#             present = First_Backward_List[i]
#             if self.cpu_cache:
#                 present = present.cuda()
#                 future = future.cuda()

#             feat = torch.cat([feat,present,future], dim=1)
#             feat = self.first_forward(feat)
#             feat = self.ofae(torch.cat([present, feat], 1)) + present
#             if self.cpu_cache:
#                 First_Forward_List.append(feat.cpu())
#                 torch.cuda.empty_cache()
#             else:
#                 First_Forward_List.append(feat)
# #####################################Second Backward Propagation##########################################
#         Second_Backward_List = []
#         feat = inputs.new_zeros(n, self.out_nc, h, w)
#         for i in range(t - 1, -1, -1):
#             future = First_Forward_List[i] if i == 0 else First_Forward_List[i - 1]
#             present = First_Forward_List[i]
#             if self.cpu_cache:
#                 present = present.cuda()
#                 future = future.cuda()

#             feat = torch.cat([feat,present,future], dim=1)
#             feat = self.second_backward(feat)
#             feat = self.ofae(torch.cat([present, feat], 1))  + present
#             if self.cpu_cache:
#                 Second_Backward_List.append(feat.cpu())
#                 torch.cuda.empty_cache()
#             else:
#                 Second_Backward_List.append(feat)
#         Second_Backward_List = Second_Backward_List[::-1]
#####################################Second Forward Propagation############################################
        Enhanced = []
        feat = inputs.new_zeros(n, self.out_nc, h, w)
        for i in range(0, t):
            # future = Second_Backward_List[i] if i == t - 1 else Second_Backward_List[i + 1]
            # present = Second_Backward_List[i]
            # if self.cpu_cache:
            #     present = present.cuda()
            #     future = future.cuda()

            # feat = torch.cat([feat,present,future], dim=1)
            # feat = self.second_forward(feat)
            # feat = self.ofae(torch.cat([present, feat], 1))+ present
            # if self.cpu_cache:
            #     out = self.qenet(torch.cat([First_Backward_List[i].cuda(), First_Forward_List[i].cuda(), Second_Backward_List[i].cuda(), feat],dim=1)) + inputs[:, i:i + 1, :, :]
            #     Enhanced.append(out.cpu())
            #     torch.cuda.empty_cache()
            # else:
                # out = self.qenet(torch.cat([First_Backward_List[i], First_Forward_List[i], Second_Backward_List[i], feat],dim=1)) + inputs[:, i:i + 1, :, :]
                # Enhanced.append(out)
            out = generate_it(inputs, i, 7, t)  # out có kích thước [B, 7, H, W]
            pred_frame = pred_d[:, i:i+1, :, :]
            
            out = self.qenet(torch.cat([out, pred_frame], dim=1)) + inputs[:, i:i + 1, :, :]
            Enhanced.append(out)

        return torch.stack(Enhanced, dim=1)

class MDFI_2(nn.Module):
    def __init__(self, in_nc=7, nf=32, out_nc=64, nb=5, cpu_cache_length=20, num_sft=16):
        super(MDFI_2, self).__init__()
        self.out_nc = out_nc
        self.cpu_cache_length = cpu_cache_length
        # Lưu ý: lq_data có 7 kênh, pred_data có 1 kênh nên tổng kênh ban đầu = 7 + 1 = 8.
        # STFF_1 sẽ nhận vào lq_data (7 kênh) và pred_data (1 kênh) riêng biệt.
        # self.stff = STFF_1(lq_nc=in_nc, out_nc=out_nc, nf=nf, deform_ks=3, num_sft=num_sft)
        self.first_backward = STFF(
            in_nc=in_nc + 1,
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
        self.ofae_1 = OFAE(self.out_nc + in_nc, self.out_nc, connection=True)
        self.ofae = OFAE(2*self.out_nc, self.out_nc, connection=True)
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
        feat = inputs.new_zeros(n, self.out_nc, h, w)
        # Vòng lặp duyệt các frame theo thứ tự ngược lại
        for i in range(t - 1, -1, -1):
            # Lấy ra 7 frame của lq_data xung quanh frame i (sử dụng hàm generate_it)
            out = generate_it(inputs, i, 7, t)  # out có kích thước [B, 7, H, W]
            # Lấy ra frame thứ i của pred_data (dùng slicing i:i+1 để giữ chiều channel)
            pred_frame = pred_d[:, i:i+1, :, :]  # pred_frame có kích thước [B, 1, H, W]
            # Gọi STFF_1 với 2 đầu vào: out và pred_frame
            # out = self.stff(out, pred_frame)
            # Tiến hành các bước xử lý như ban đầu
            feat = torch.cat([out, pred_frame], dim=1)
            feat = self.first_backward(feat)
            # print(feat.shape)
            feat = self.ofae_1(torch.cat([out, feat], 1))
            # print(feat.shape)
            # feat = feat + out
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
            # out = self.qenet(inputs) + inputs[:, i:i + 1, :, :]
            # Enhanced.append(out)

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
    net = MDFI_1().cuda()
    from thop import profile
    with torch.no_grad():

        input = torch.randn(1, 15, 32, 32).cuda()
        flops, params = profile(net, inputs=(input, input))
        total = sum([param.nelement() for param in net.parameters()])
        print('   Number of params: %.2fM' % (total / 1e6))
        print('   Number of FLOPs: %.2fGFLOPs' % (flops / 1e9))