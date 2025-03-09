## basic Block - darknet - darknet53
import math
from collections import OrderedDict

import torch
import torch.nn as nn

## 1x1 下降通道数(1,1,0)，3x3提取特征 上升通道数(3,1,1)，接残差边
## CNL: conv, bn, leaky relu 
class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes):
        super(BasicBlock, self).__init__()
        # 1x1
        # inplanes 通道数
        self.conv1 = nn.Conv2d(inplanes, planes[0], kernel_size=1,stride=1,padding=0,bias=False)
        self.bn1 = nn.BatchNorm2d(planes[0])
        self.relu1 = nn.LeakyReLU(0.1)

        # 3x3
        self.conv2  = nn.Conv2d(planes[0], planes[1],kernel_size=3, stride=1,padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes[1])
        self.relu2 = nn.LeakyReLU(0.1)
    
    def forward(self, x):
        residual = x
        # 1x1
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        # 3x3
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        out += residual
        return out
    
class DarkNet(nn.Module):
    def __init__(self, layers):
        # CBL, res1, res2, res8, res8, res4
        # resN, padding+CBL+Nxresunit (_make_layer)
        # resunit, res,CBL,CBL,add(x+res)-out
        super(DarkNet, self).__init__()
        self.inplanes = 32
        # 416,416,3->416,416,32 Conv2d(inchannel, outchannel **kwargs)
        # CBL
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu1 = nn.LeakyReLU(0.1)

        # 416,416,32-> 208,208,64, etc ->104,52,26,13
        # res1, res2, res8, res8, res4
        self.layer1 = self._make_layer([32,64], layers[0])
        self.layer2 = self._make_layer([64,128], layers[1])
        self.layer3 = self._make_layer([128,256], layers[2])
        self.layer4 = self._make_layer([256,512], layers[3])
        self.layer5 = self._make_layer([512,1024], layers[4])

        self.layers_out_filters = [64,128,256,512,1025]

        # weight init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0,math.sqrt(2. /n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    # stride2 3x3conv下采样，残差堆叠, CBL
    def _make_layer(self, planes, blocks):
        layers = []

        layers.append(("ds_conv", nn.Conv2d(self.inplanes, planes[1], kernel_size=3, stride=2, padding=1,bias=False)))
        layers.append(("ds_bn", nn.BatchNorm2d(planes[1])))
        layers.append(("ds_relu", nn.LeakyReLU(0.1)))

        # residual
        self.inplanes = planes[1]
        for i in range(0,blocks):
            layers.append(("residual_{}",format(i), BasicBlock(self.inplanes, planes)))
        return nn.Sequential(OrderedDict(layers))
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        out3 = self.layer3(x)
        out4 = self.layer4(out3)
        out5 = self.layer5(out4)

        return out3, out4, out5



def darknet53(pretrained, **kwargs):
    model = DarkNet([1,2,8,8,4]) # 8,8,4 -> 52,26,13   out3,out4,out5
    if pretrained:
        if isinstance(pretrained,str):
            model.load_state_dict(torch.load(pretrained))
        else:
            raise Exception("darknet pretrain path requested, got [{}]".format(pretrained))
    return model