from collections import OrderedDict

import torch
import torch.nn as nn
from nets.darknet import darknet53

# CBL
def conv2d(filter_in, filter_out, kernel_size):
    pad = (kernel_size -1) //2 if kernel_size else 0

    return nn.Sequential(OrderedDict([
        ("conv", nn.Conv2d(filter_in, filter_out, kernel_size=kernel_size, stride=1, padding=pad, bias=False)),
        ("bn", nn.BatchNorm2d(filter_out)),
        ("relu", nn.LeakyReLU(0.1)),
    ]))


def make_last_layers(filters_list, in_filters, out_filter):
    m = nn.ModuleList([
        conv2d(in_filters,filters_list[0],1),
        conv2d(filters_list[0],filters_list[1],3),
        conv2d(filters_list[1],filters_list[0],1),
        conv2d(filters_list[0],filters_list[1],3),
        conv2d(filters_list[1],filters_list[0],1),
        conv2d(filters_list[0],filters_list[1],3),
        nn.Conv2d(filters_list[1],out_filter,kernel_size=1,stride=1,padding=0,bias=True)
        
    ])
    return m


class YoloBody(nn.Module):
    def __init__(self, config):
        super(YoloBody, self).__init__()
        self.config = config

        ## darknet get three shape for feature: 52x52x256, 26x26x512, 13x13x1024
        self.backbone = darknet53(None)
        # out_filters [64,128,256,512,1024]
        out_filters = self.backbone.layers_out_filters

        final_out_filter0 = len(config["yolo"]["anchors"][0]*(5+config["yoll"]["classes"]))
        self.last_layer0 = make_last_layers([512,1024], out_filters[-1], final_out_filter0)

        final_out_filter1 = len(config["yolo"]["anchors"][1]*(5+config["yoll"]["classes"]))
        # 卷积 上采样
        self.last_layer1_conv = conv2d(512,256,1)
        self.last_layer1_upsample = nn.Upsample(scale_factor=2,mode="nearest")
        self.last_layer1 = make_last_layers([256,512], out_filters[-2]+256, final_out_filter1)

  
        final_out_filter2 = len(config["yolo"]["anchors"][2]*(5+config["yoll"]["classes"]))
        # 卷积 上采样
        self.last_layer2_conv = conv2d(256,128,1)
        self.last_layer2_upsample = nn.Upsample(scale_factor=2,mode="nearest")
        self.last_layer2 = make_last_layers([128,256], out_filters[-3]+128, final_out_filter2)

    def forward(self,x):
        def _branch(last_layer, layer_in):
            for i,e in enumerate(last_layer):
                layer_in = e(layer_in)
                if i == 4:
                    outbranch = layer_in
            return layer_in,outbranch
        
        x2,x1,x0 = self.backbone(x) # res8,res8,res4
        #13, cbl+upsample
        out0, out0_branch = _branch(self.last_layer0,x0)
        x1_in = self.last_layer1_conv(out0_branch)
        x1_in = self.last_layer1_upsample(x1_in)
        # concat dim 1, dim0 is batch, after concat, 256+512
        x1_in = torch.cat([x1_in, x1],1)
        
        #26
        out1, out1_branch = _branch(self.last_layer1,x1_in)
        x2_in = self.last_layer1_conv(out1_branch)
        x2_in = self.last_layer1_upsample(x2_in)
        x2_in = torch.cat([x2_in, x2],1)

        out2, _ = _branch(self.last_layer2, x2_in)
        return out0,out1,out2
        