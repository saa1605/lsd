import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import copy
import src.models.clip as clip 
import sys

def clones(module, N):
    """Produce N identical layers"""
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class LinearProjectionLayers(nn.Module):
    def __init__(
        self, image_channels, linear_hidden_size, rnn_hidden_size, num_hidden_layers
    ):
        super(LinearProjectionLayers, self).__init__()

        if num_hidden_layers == 0:
            # map pixel feature vector directly to score without activation
            self.out_layers = nn.Linear(image_channels + rnn_hidden_size, 1, bias=False)
        else:
            self.out_layers = nn.Sequential(
                nn.Conv2d(
                    image_channels + rnn_hidden_size,
                    linear_hidden_size,
                    kernel_size=1,
                    padding=0,
                    stride=1,
                ),
                nn.ReLU(),
                nn.Conv2d(linear_hidden_size, 1, kernel_size=1, padding=0, stride=1),
            )
            self.out_layers.apply(self.init_weights)

    def forward(self, x):
        return self.out_layers(x)

    def init_weights(self, m):
        if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

class IdentityBlock(nn.Module):
    def __init__(self, in_planes, filters, kernel_size, stride=1, final_relu=True, batchnorm=True):
        super(IdentityBlock, self).__init__()
        self.final_relu = final_relu
        self.batchnorm = batchnorm

        filters1, filters2, filters3 = filters
        self.conv1 = nn.Conv2d(in_planes, filters1, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(filters1) if self.batchnorm else nn.Identity()
        self.conv2 = nn.Conv2d(filters1, filters2, kernel_size=kernel_size, dilation=1,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(filters2) if self.batchnorm else nn.Identity()
        self.conv3 = nn.Conv2d(filters2, filters3, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(filters3) if self.batchnorm else nn.Identity()

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += x
        if self.final_relu:
            out = F.relu(out)
        return out

class ConvBlock(nn.Module):
    def __init__(self, in_planes, filters, kernel_size, stride=1, final_relu=True, batchnorm=True):
        super(ConvBlock, self).__init__()
        self.final_relu = final_relu
        self.batchnorm = batchnorm

        filters1, filters2, filters3 = filters
        self.conv1 = nn.Conv2d(in_planes, filters1, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(filters1) if self.batchnorm else nn.Identity()
        self.conv2 = nn.Conv2d(filters1, filters2, kernel_size=kernel_size, dilation=1,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(filters2) if self.batchnorm else nn.Identity()
        self.conv3 = nn.Conv2d(filters2, filters3, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(filters3) if self.batchnorm else nn.Identity()

        self.shortcut = nn.Sequential(
            nn.Conv2d(in_planes, filters3,
                      kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(filters3) if self.batchnorm else nn.Identity()
        )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        if self.final_relu:
            out = F.relu(out)
        return out

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),                                     # (Mohit): argh... forgot to remove this batchnorm
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),                                     # (Mohit): argh... forgot to remove this batchnorm
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)


class LingUNet(nn.Module):
    def __init__(self, args):
        super(LingUNet, self).__init__()

        self.clip, _ = clip.load('RN50')
        self.clip = self.clip.float()
        args.clip_preprocess = _
        
        # initialize weights
        self.text2convs.apply(self.init_weights)
        self.conv_layers.apply(self.init_weights)
        self.deconv_layers.apply(self.init_weights)
        self.out_layers.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)


    def forward(self, images, tokens):
        B, num_maps, C, H, W = images.size()

        out = self.out_layers(H).squeeze(-1)
        out = out.view(B, num_maps, out.size()[-2], out.size()[-1])
        # out = out * mask
        out = F.log_softmax(out.view(B, -1), 1).view(B, num_maps, height, width)

        # image_encodings, text_encodings = self.get_region_encodings(images, tokens, regions)

        return out#, image_encodings, text_encodings, self.multitask_lambda 



def load_oldArgs(args, oldArgs):
    args.m = oldArgs["num_lingunet_layers"]
    args.image_channels = oldArgs["linear_hidden_size"]
    args.freeze_resnet = oldArgs["freeze_resnet"]
    args.res_connect = oldArgs["res_connect"]
    args.embed_size = oldArgs["embed_size"]
    args.rnn_hidden_size = oldArgs["rnn_hidden_size"]
    args.num_rnn_layers = oldArgs["num_rnn_layers"]
    args.embed_dropout = oldArgs["embed_dropout"]
    args.bidirectional = oldArgs["bidirectional"]
    args.linear_hidden_size = oldArgs["linear_hidden_size"]
    args.num_linear_hidden_layers = oldArgs["num_linear_hidden_layers"]
    args.ds_percent = oldArgs["ds_percent"]
    args.ds_height = oldArgs["ds_height"]
    args.ds_width = oldArgs["ds_width"]

    return args


def convert_model_to_state(model, optimizer, args, rnn_args):
    state = {
        "args": args,
        "rnn_args": rnn_args,
        "state_dict": {},
        "optimizer_state_dict": optimizer.state_dict(),
    }
    # use copies instead of references
    for k, v in model.state_dict().items():
        state["state_dict"][k] = v.clone().to(torch.device("cpu"))

    return state 
