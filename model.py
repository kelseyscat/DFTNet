import torch  # tim，穿梭AdaIN, 通道数全都 / 2
import torch.nn as nn
import torch.nn.functional as F
from torchgan.layers import SpectralNorm2d
import enum
import numpy as np
from ssim import msssim
from torch.autograd import Variable
from torchvision.models.vgg import vgg19
import os
from thop import profile
from thop import clever_format

NUM_BANDS = 4
PATCH_SIZE = 256
SCALE_FACTOR = 16
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def calc_mean_std(feat, eps=1e-5):
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std

def adaptive_instance_normalization(content_feat, style_feat):
    assert (content_feat.size()[:2] == style_feat.size()[:2])
    size = content_feat.size()
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)

    normalized_feat = (content_feat - content_mean.expand(
        size)) / content_std.expand(size)
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)

def conv_1x1(in_channel, out_channel):
    return nn.Sequential(
        nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1, padding=0, bias=True),
        nn.BatchNorm2d(out_channel),
        nn.ReLU(inplace=True)
    )

class TFF(torch.nn.Module):
    def __init__(self, in_channel, out_channel):  # outchannel设256
        super(TFF, self).__init__()
        self.catconvA = conv_3x3(in_channel * 2, in_channel)
        self.catconvB = conv_3x3(in_channel * 2, in_channel)
        self.catconv = conv_3x3(in_channel * 3, out_channel)
        self.convA = nn.Conv2d(in_channel, 1, 1)
        self.convB = nn.Conv2d(in_channel, 1, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, xA, xB):
        x_diff = xA - xB
        x_diffA = self.catconvA(torch.cat([x_diff, xA], dim=1))
        x_diffB = self.catconvB(torch.cat([x_diff, xB], dim=1))

        A_weight = self.sigmoid(self.convA(x_diffA))
        B_weight = self.sigmoid(self.convB(x_diffB))

        xA = A_weight * xA
        xB = B_weight * xB

        x = self.catconv(torch.cat([xA, xB, x_diff], dim=1))
        return xA, xB, x


class ResBlock(nn.Module):
    def __init__(self, in_channels, kernel_size = 3, stride = 1, padding = 1, bias = True):
        super(ResBlock, self).__init__()
        residual = [
            nn.ReflectionPad2d(padding),
            nn.Conv2d(in_channels, in_channels, kernel_size, stride, 0, bias=bias),
            nn.Dropout(0.5),
            nn.LeakyReLU(inplace=True),
            nn.ReflectionPad2d(padding),
            nn.Conv2d(in_channels, in_channels, kernel_size, stride, 0, bias=bias),
            nn.Dropout(0.5),
        ]
        self.residual = nn.Sequential(*residual)

    def forward(self, inputs):
        trunk = self.residual(inputs)
        return trunk + inputs

class InfoExchange(torch.nn.Module):
    def __init__(self, in_channels=NUM_BANDS):
        super(InfoExchange, self).__init__()  # VGG structure
        channels = (16, 32, 64, 128)
        self.Res_layer1 = nn.Sequential(
            nn.Conv2d(in_channels, channels[0], 7, 1, 3),
            ResBlock(channels[0]),
        )
        self.Res_layer2 = nn.Sequential(
            nn.Conv2d(channels[0], channels[1], 3, 2, 1),
            ResBlock(channels[1]),
        )
        self.Res_layer3 = nn.Sequential(
            nn.Conv2d(channels[1], channels[2], 3, 2, 1),
            ResBlock(channels[2]),
        )
        self.Res_layer4 = nn.Sequential(
            nn.Conv2d(channels[2], channels[3], 3, 2, 1),
            ResBlock(channels[3]),
        )
        self.ConcatConv1 = conv_1x1(channels[0] * 2, channels[0])
        self.ConcatConv2 = conv_1x1(channels[1] * 2, channels[1])
        self.ConcatConv3 = conv_1x1(channels[2] * 2, channels[2])
        self.ConcatConv4 = conv_1x1(channels[3] * 2, channels[3])
        self.timeInfo1 = TFF(channels[0], channels[0])
        self.timeInfo2 = TFF(channels[1], channels[1])
        self.timeInfo3 = TFF(channels[2], channels[2])
        self.timeInfo4 = TFF(channels[3], channels[3])


    def forward(self, c1, c2, f):
        c1_1 = self.Res_layer1(c1)
        c2_1 = self.Res_layer1(c2)
        f_1 = self.Res_layer1(f)

        x1, x2, _ = self.timeInfo1(c1_1, c2_1)
        f_1_with_c1_feature = adaptive_instance_normalization(f_1, x1)
        f_1_with_c2_feature = adaptive_instance_normalization(f_1, x2)

        a = torch.cat((c1_1, f_1_with_c1_feature), dim=1)
        c1_1 = self.ConcatConv1(a)
        b = torch.cat((c2_1, f_1_with_c2_feature), dim=1)
        c2_1 = self.ConcatConv1(b)

        # ---------

        c1_2 = self.Res_layer2(c1_1)
        c2_2 = self.Res_layer2(c2_1)
        f_2 = self.Res_layer2(f_1)

        x1, x2, _ = self.timeInfo2(c1_2, c2_2)
        f_2_with_c1_feature = adaptive_instance_normalization(f_2, x1)
        f_2_with_c2_feature = adaptive_instance_normalization(f_2, x2)

        a = torch.cat((c1_2, f_2_with_c1_feature), dim=1)
        c1_2 = self.ConcatConv2(a)
        b = torch.cat((c2_2, f_2_with_c2_feature), dim=1)
        c2_2 = self.ConcatConv2(b)

        # ---------

        c1_3 = self.Res_layer3(c1_2)
        c2_3 = self.Res_layer3(c2_2)
        f_3 = self.Res_layer3(f_2)

        x1, x2, _ = self.timeInfo3(c1_3, c2_3)
        f_3_with_c1_feature = adaptive_instance_normalization(f_3, x1)
        f_3_with_c2_feature = adaptive_instance_normalization(f_3, x2)

        a = torch.cat((c1_3, f_3_with_c1_feature), dim=1)
        c1_3 = self.ConcatConv3(a)
        b = torch.cat((c2_3, f_3_with_c2_feature), dim=1)
        c2_3 = self.ConcatConv3(b)

        # ---------

        c1_4 = self.Res_layer4(c1_3)
        c2_4 = self.Res_layer4(c2_3)
        f_4 = self.Res_layer4(f_3)

        x1, x2, x = self.timeInfo4(c1_4, c2_4)
        f_4_with_c1_feature = adaptive_instance_normalization(f_4, x1)
        f_4_with_c2_feature = adaptive_instance_normalization(f_4, x2)


        return x, [f_1_with_c1_feature, f_1_with_c2_feature], [f_2_with_c1_feature, f_2_with_c2_feature], \
               [f_3_with_c1_feature, f_3_with_c2_feature], [f_4_with_c1_feature, f_4_with_c2_feature]

def conv_3x3(in_channel, out_channel):
    return nn.Sequential(
        nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(out_channel),
        nn.ReLU(inplace=True)
    )

class Decoder(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Decoder, self).__init__()
        channels = (16, 32, 64, 128)
        self.upsample1 = nn.Sequential(
            nn.Conv2d(channels[3] * 3, channels[2] * 4, kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(2),
            nn.LeakyReLU(0.1, True),
        )

        self.upsample2 = nn.Sequential(
            nn.Conv2d(channels[2] * 3, channels[1] * 4, kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(2),
            nn.LeakyReLU(0.1, True),
        )

        self.upsample3 = nn.Sequential(
            nn.Conv2d(channels[1] * 3, channels[0] * 4, kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(2),
            nn.LeakyReLU(0.1, True),
        )

        self.conv = nn.Conv2d(channels[0] * 3, out_channel, kernel_size=3, stride=1, padding=1)

    def forward(self, x, f1, f2, f3, f4):

        x = torch.cat((x, f4[0], f4[1]), dim=1)
        x = self.upsample1(x)

        x = torch.cat((x, f3[0], f3[1]), dim=1)
        x = self.upsample2(x)

        x = torch.cat((x, f2[0], f2[1]), dim=1)
        x = self.upsample3(x)

        x = torch.cat((x, f1[0], f1[1]), dim=1)
        x = self.conv(x)

        return x

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.encoder = InfoExchange(NUM_BANDS)
        self.decoder = Decoder(256, 4)

    def forward(self, inputs):
        c1, c2, f1 = inputs[2], inputs[0], inputs[1]

        x, f1_1, f1_2, f1_3, f1_4 = self.encoder(c1, c2, f1)

        return self.decoder(x, f1_1, f1_2, f1_3, f1_4)

class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False, getIntermFeat=False):
        super(NLayerDiscriminator, self).__init__()
        self.getIntermFeat = getIntermFeat
        self.n_layers = n_layers

        kw = 4
        padw = int(np.ceil((kw-1.0)/2))
        sequence = [[nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]]

        nf = ndf
        for n in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            sequence += [[
                nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=2, padding=padw),
                norm_layer(nf), nn.LeakyReLU(0.2, True)
            ]]

        nf_prev = nf
        nf = min(nf * 2, 512)
        sequence += [[
            nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=1, padding=padw),
            norm_layer(nf),
            nn.LeakyReLU(0.2, True)
        ]]

        sequence += [[nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw)]]

        if use_sigmoid:
            sequence += [[nn.Sigmoid()]]

        if getIntermFeat:
            for n in range(len(sequence)):
                setattr(self, 'model'+str(n), nn.Sequential(*sequence[n]))
        else:
            sequence_stream = []
            for n in range(len(sequence)):
                sequence_stream += sequence[n]
            self.model = nn.Sequential(*sequence_stream)

    def forward(self, input):
        if self.getIntermFeat:
            res = [input]
            for n in range(self.n_layers+2):
                model = getattr(self, 'model'+str(n))
                res.append(model(res[-1]))
            return res[1:]
        else:
            return self.model(input)


class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.cuda.FloatTensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label  # 数据集的label是什么label? 哦！那应该是两张图片的像素块
        self.fake_label = target_fake_label  #
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != input.numel()))
            if create_label:
                real_tensor = self.Tensor(input.size()).fill_(self.real_label)
                self.real_label_var = Variable(real_tensor, requires_grad=False)
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
                self.fake_label_var = Variable(fake_tensor, requires_grad=False)
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, input, target_is_real):
        if isinstance(input[0], list):
            loss = 0
            for input_i in input:
                pred = input_i[-1]
                target_tensor = self.get_target_tensor(pred, target_is_real)
                loss += self.loss(pred, target_tensor)
            return loss
        else:
            target_tensor = self.get_target_tensor(input[-1], target_is_real)
            return self.loss(input[-1], target_tensor)

if __name__ == "__main__":
    c1 = torch.rand([8, 4, 256, 256])
    c2 = torch.rand([8, 4, 256, 256])
    f = torch.rand([8, 4, 256, 256])
    net = Generator()
    input = [c2, f, c1]
    # discriminator = NLayerDiscriminator(input_nc=12, getIntermFeat=True)
    # re = net([c2, f, c1])
    # pred_fake = discriminator(torch.cat((re.detach(), f), dim=1))
    flops, params = profile(net, inputs=(input, ))
    flops, params = clever_format([flops, params], "%.3f")
    print("flops", params)
    print("flops", flops)

