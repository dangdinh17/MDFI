import torch.nn as nn
import torch
import torch.nn.functional as F
from einops import rearrange
import math
import warnings
from torch.nn.init import _calculate_fan_in_and_fan_out
from pdb import set_trace as stx
# import cv2


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)
    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # type: (Tensor, float, float, float, float) -> Tensor # type: ignore
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


def variance_scaling_(tensor, scale=1.0, mode='fan_in', distribution='normal'):
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    if mode == 'fan_in':
        denom = fan_in
    elif mode == 'fan_out':
        denom = fan_out
    elif mode == 'fan_avg':
        denom = (fan_in + fan_out) / 2
    variance = scale / denom
    if distribution == "truncated_normal":
        trunc_normal_(tensor, std=math.sqrt(variance) / .87962566103423978)
    elif distribution == "normal":
        tensor.normal_(std=math.sqrt(variance))
    elif distribution == "uniform":
        bound = math.sqrt(3 * variance)
        tensor.uniform_(-bound, bound)
    else:
        raise ValueError(f"invalid distribution {distribution}")


def lecun_normal_(tensor):
    variance_scaling_(tensor, mode='fan_in', distribution='truncated_normal')
    
def conv(in_channels, out_channels, kernel_size, bias=False, padding=1, stride=1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2), bias=bias, stride=stride)


# input [bs,28,256,310]  output [bs, 28, 256, 256]
def shift_back(inputs, step=2):
    [bs, nC, row, col] = inputs.shape
    down_sample = 256 // row
    step = float(step) / float(down_sample * down_sample)
    out_col = row
    for i in range(nC):
        inputs[:, i, :, :out_col] = \
            inputs[:, i, :, int(step * i):int(step * i) + out_col]
    return inputs[:, :, :, :out_col]

class PreNorm(nn.Module):
    def __init__(self, nf, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(nf)

    def forward(self, x, *args, **kwargs):
        x = self.norm(x)
        return self.fn(x, *args, **kwargs)


class GELU(nn.Module):
    def forward(self, x):
        return F.gelu(x)

class MSA(nn.Module):
    def __init__(
            self,
            nf,
            nf_head=64,
            heads=8,
    ):
        super().__init__()
        self.num_heads = heads
        self.nf_head = nf_head
        self.to_q = nn.Linear(nf, nf_head * heads, bias=False)
        self.to_k = nn.Linear(nf, nf_head * heads, bias=False)
        self.to_v = nn.Linear(nf, nf_head * heads, bias=False)
        self.rescale = nn.Parameter(torch.ones(heads, 1, 1))
        self.proj = nn.Linear(nf_head * heads, nf, bias=True)
        self.pos_emb = nn.Sequential(
            nn.Conv2d(nf, nf, 3, 1, 1, bias=False, groups=nf),
            GELU(),
            nn.Conv2d(nf, nf, 3, 1, 1, bias=False, groups=nf),
        )
        self.nf = nf

    def forward(self, x_in, hi_fea=None):
        """
        x_in: [b,h,w,c]         # input_feature
        hi_fea: [b,h,w,c]         # mask shift? 为什么是 b, h, w, c?
        return out: [b,h,w,c]
        """
        
        b, h, w, c = x_in.shape
        x = x_in.reshape(b, h * w, c)
        q_inp = self.to_q(x)
        k_inp = self.to_k(x)
        v_inp = self.to_v(x)
        
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads),
                                 (q_inp, k_inp, v_inp))
        if hi_fea != None:
            dct_attn = hi_fea
            dct_attn = rearrange(dct_attn.flatten(1, 2), 'b n (h d) -> b h n d', h=self.num_heads)
            v = v * dct_attn
        # q: b,heads,hw,c
        q = q.transpose(-2, -1)
        k = k.transpose(-2, -1)
        v = v.transpose(-2, -1)
        q = F.normalize(q, dim=-1, p=2)
        k = F.normalize(k, dim=-1, p=2)
        attn = (k @ q.transpose(-2, -1))   # A = K^T*Q
        attn = attn * self.rescale
        attn = attn.softmax(dim=-1)
        x = attn @ v   # b,heads,d,hw
        x = x.permute(0, 3, 1, 2)    # Transpose
        x = x.reshape(b, h * w, self.num_heads * self.nf_head)
        out_c = self.proj(x).view(b, h, w, c)
        out_p = self.pos_emb(v_inp.reshape(b, h, w, c).permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        out = out_c + out_p

        return out


class FeedForward(nn.Module):
    def __init__(self, nf, mult=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(nf, nf * mult, 1, 1, bias=False),
            GELU(),
            nn.Conv2d(nf * mult, nf * mult, 3, 1, 1,
                      bias=False, groups=nf * mult),
            GELU(),
            nn.Conv2d(nf * mult, nf, 1, 1, bias=False),
        )

    def forward(self, x):
        """
        x: [b,h,w,c]
        return out: [b,h,w,c]
        """
        out = self.net(x.permute(0, 3, 1, 2).contiguous())
        return out.permute(0, 2, 3, 1)


class AB(nn.Module):
    def __init__(
            self,
            nf,
            nf_head=64,
            heads=8,
            num_blocks=2,
    ):
        super().__init__()
        self.blocks = nn.ModuleList([])
        for _ in range(num_blocks):
            self.blocks.append(nn.ModuleList([
                MSA(nf=nf, nf_head=nf_head, heads=heads),
                PreNorm(nf, FeedForward(nf=nf))
            ]))

    def forward(self, x, hi_fea=None):
        """
        x: [b,c,h,w]
        hi_fea: [b,c,h,w]
        return out: [b,c,h,w]
        """
        x = x.permute(0, 2, 3, 1)
        if hi_fea != None:
            hi_fea = hi_fea.permute(0, 2, 3, 1)
        for (attn, ff) in self.blocks:
            x = attn(x, hi_fea) + x
            x = ff(x) + x
        out = x.permute(0, 3, 1, 2)
        return out


class Enhancer(nn.Module):
    def __init__(self, in_nc, out_nc,nf=32, level=2, num_blocks=[1, 2, 2]):
        super(Enhancer, self).__init__()
        self.nf = nf
        self.level = level

        # Input projection
        self.embedding = nn.Conv2d(in_nc, self.nf, 3, 1, 1, groups=self.nf, bias=False)
        self.connection = nn.Conv2d(in_nc, out_nc, 3, 1, 1, groups=out_nc,bias=False)
        # Encoder
        self.encoder_layers = nn.ModuleList([])
        nf_level = nf
        for i in range(level):
            self.encoder_layers.append(nn.ModuleList([
                AB(
                    nf=nf_level, num_blocks=num_blocks[i], nf_head=nf, heads=1),
                nn.Conv2d(nf_level, nf_level, 4, 2, 1, bias=False),
                nn.Conv2d(nf_level, nf_level, 4, 2, 1, bias=False)
            ]))
            # nf_level *= 2

        # Bottleneck
        self.bottleneck = AB(
            nf=nf_level, nf_head=nf, heads=nf_level // nf, num_blocks=num_blocks[-1])

        # Decoder
        self.decoder_layers = nn.ModuleList([])
        for i in range(level):
            self.decoder_layers.append(nn.ModuleList([
                nn.ConvTranspose2d(nf_level, nf_level, stride=2,
                                   kernel_size=2, padding=0, output_padding=0),
                nn.Conv2d(nf_level*2, nf_level, 1, 1, bias=False),
                AB(
                    nf=nf_level, num_blocks=num_blocks[level - 1 - i], nf_head=nf,
                    heads=1),
            ]))
            # nf_level //= 2

        # Output projection
        self.mapping = nn.Conv2d(self.nf, out_nc, 3, 1, 1, groups=self.nf,bias=False)
        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, hi_fea=None):
        """
        x:          [b,c,h,w]         x是feature, 不是image
        hi_fea:   [b,c,h,w]
        return out: [b,c,h,w]
        """

        # Embedding
        fea = self.embedding(x)

        # Encoder
        fea_encoder = []
        hi_fea_list = []
        for (AB, FeaDownSample, FeaDownsample) in self.encoder_layers:
            fea = AB(fea, hi_fea)  # bchw
            hi_fea_list.append(hi_fea)
            fea_encoder.append(fea)
            fea = FeaDownSample(fea)
            if hi_fea != None:
                hi_fea = FeaDownsample(hi_fea)

        # Bottleneck
        fea = self.bottleneck(fea, hi_fea)

        # Decoder
        for i, (FeaUpSample, Fution, AB) in enumerate(self.decoder_layers):
            fea = FeaUpSample(fea)
            fea = Fution(
                torch.cat([fea, fea_encoder[self.level - 1 - i]], dim=1))
            hi_fea = hi_fea_list[self.level-1-i]
            fea = AB(fea, hi_fea)

        # Mapping
        out = self.mapping(fea) + self.connection(x)
        return out
    
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# net = Enhancer(in_nc=64*2, out_nc=64, nf=32).to(device)

# from ptflops import get_model_complexity_info

# with torch.no_grad():
#     input = torch.randn(1, 128, 64, 64).to(device)
#     output = net(input)
#     # print(output.shape)
#     # print(summary(net, input_size=(1, 7, 64, 64)))
#     macs, params = get_model_complexity_info(net, (1, 64, 64), as_strings=True, print_per_layer_stat=False, verbose=False)
#     # print(f"MACs: {macs/(1e6):.2f}M, Params: {params/(1e3):.2f}K")