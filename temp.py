import torch 

from models import *

torch.cuda.set_device(0)
net = MDFI_2().cuda()
from thop import profile
with torch.no_grad():

    input = torch.randn(1, 7, 32, 32).cuda()
    flops, params = profile(net, inputs=(input, input))
    total = sum([param.nelement() for param in net.parameters()])
    print('   Number of params: %.2fM' % (total / 1e6))
    print('   Number of FLOPs: %.2fGFLOPs' % (flops / 1e9))