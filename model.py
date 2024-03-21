import torch # 两层, 加空间信息
from torch import nn, einsum
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import numbers
from calflops import calculate_flops

NUM_BANDS = 4
PATCH_SIZE = 256
SCALE_FACTOR = 16
NHEAD = 2
STAGE = 1

# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def conv_1x1(in_channel, out_channel):
    return nn.Sequential(
        nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1, padding=0, bias=True),
        nn.BatchNorm2d(out_channel),
        nn.ReLU(inplace=True)
    )


def conv_3x3(in_channel, out_channel):
    return nn.Sequential(
        nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(out_channel),
        nn.ReLU(inplace=True)
    )

def Spatial_Branch(in_channel, out_channel):
    return nn.Sequential(
        conv_3x3(in_channel, out_channel),
        nn.LeakyReLU(inplace=False),
        conv_3x3(out_channel, out_channel)
    )

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias

class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)

        return x

## Multi-DConv Head Transposed Self-Attention (MDTA)
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads  # 注意力头的个数
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))  # 可学习系数

        # 1*1 升维
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        # 3*3 分组卷积
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        # 1*1 卷积
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x, y):
        b, c, h, w = x.shape  # 输入的结构 batch 数，通道数和高宽

        x_qkv = self.qkv_dwconv(self.qkv(x))
        _, k, v = x_qkv.chunk(3, dim=1)  # 第 1 个维度方向切分成 3 块

        y_qkv = self.qkv_dwconv(self.qkv(y))
        q, _, _ = y_qkv.chunk(3, dim=1)  # 第 1 个维度方向切分成 3 块

        # 改变 q, k, v 的结构为 b head c (h w)，将每个二维 plane 展平
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)  # C 维度标准化，这里的 C 与通道维度略有不同
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)  # 注意力图(严格来说不算图)

        # 将展平后的注意力图恢复
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        # 真正的注意力图
        out = self.project_out(out)
        return out


## Gated-Dconv Feed-Forward Network (GDFN)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        # 隐藏层特征维度等于输入维度乘以扩张因子
        hidden_features = int(dim * ffn_expansion_factor)
        # 1*1 升维
        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)
        # 3*3 分组卷积
        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)
        # 1*1 降维
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)  # 第 1 个维度方向切分成 2 块
        x = F.gelu(x1) * x2  # gelu 相当于 relu+dropout
        x = self.project_out(x)
        return x

## 就是标准的 Transformer 架构
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)  # 层标准化
        self.attn = Attention(dim, num_heads, bias)  # 自注意力
        self.norm2 = LayerNorm(dim, LayerNorm_type)  # 层表转化
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)  # FFN

    def forward(self, x, y):
        x = y + self.attn(self.norm1(x), self.norm1(y))  # 残差
        x = x + self.ffn(self.norm2(x))  # 残差

        return x

class Restormer(nn.Module):
    def __init__(self,
                 inp_channels,  # [16, 32, 64, 128]
                 heads,  # [1, 2, 4, 8]
                 dim=48,  # 特征图维度
                 num_blocks=[4, 6, 6, 8],
                 ffn_expansion_factor=2.66,  # 扩展因子
                 bias=False,
                 LayerNorm_type='WithBias',  ## Other option 'BiasFree'
                 ):

        super(Restormer, self).__init__()

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        self.encoder_level = TransformerBlock(dim=dim, num_heads=heads, ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                             LayerNorm_type=LayerNorm_type)

    def forward(self, X, Y):

        x = self.patch_embed(X)
        y = self.patch_embed(Y)
        out_enc_level = self.encoder_level(x, y)

        return out_enc_level

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.Conv1 = conv_3x3(NUM_BANDS, 16)
        self.Conv2 = conv_3x3(16, 32)
        self.Conv3 = conv_3x3(32, 64)
        self.Conv4 = conv_3x3(64, 128)
        self.Transformer1 = Restormer(inp_channels=16, heads=1)
        self.Transformer2 = Restormer(inp_channels=32, heads=2)
        self.Spatial_1 = Spatial_Branch(NUM_BANDS, 16)
        self.Spatial_2 = Spatial_Branch(16, 32)
        self.Conv1_1 = conv_1x1(112, NUM_BANDS) # 64, 112

    def forward(self, inputs):
        c1, c2, f1 = inputs[2], inputs[0], inputs[1]

        c1_1 = self.Conv1(c1)   # [8, 16, 256, 256]
        c2_1 = self.Conv1(c2)   # [8, 16, 256, 256]
        f1_1 = self.Conv1(f1)   # [8, 16, 256, 256]

        f1_spatial = self.Spatial_1(f1)  # [8, 16, 256, 256]

        # For f_1_amplitude
        f_1_amplitude = torch.abs(torch.fft.fftn(f1_1))  # [8, 16, 256, 256]
        # print(f_1_amplitude.shape)  # torch.Size([8, 16, 256, 256])

        # For c1_1_amplitude and c1_1_phase
        c1_1_amplitude = torch.abs(torch.fft.fftn(c1_1))
        c1_1_phase = torch.angle(torch.fft.fftn(c1_1))
        # print(c1_1_phase.shape)  # torch.Size([8, 16, 256, 256])

        # For c2_1_phase
        c2_1_phase = torch.angle(torch.fft.fftn(c2_1))

        new_1_amplitude = self.Transformer1(c1_1_amplitude, f_1_amplitude)
        new_1_phase = self.Transformer1(c1_1_phase, c2_1_phase)

        # print(new_1_amplitude.shape)

        # 计算复数的实部和虚部
        real_part = new_1_amplitude * torch.cos(new_1_phase)
        imaginary_part = new_1_amplitude * torch.sin(new_1_phase)

        # 将实部和虚部合并成复数
        complex_spectrum = torch.complex(real_part, imaginary_part)

        # 执行逆傅立叶变换
        f_reconstructed = torch.abs(torch.fft.ifftn(complex_spectrum))
        # f_reconstructed = torch.cat((f_reconstructed, f1_1_spatial), dim=1)

        # ----------------------------------
        c1_2 = self.Conv2(c1_1)
        c2_2 = self.Conv2(c2_1)
        f1_2 = self.Conv2(f1_1)

        # f1_2_spatial = self.Spatial_2(f1_1_spatial)

        f_2_amplitude = torch.abs(torch.fft.fftn(f1_2))  # [8, 16, 256, 256]

        c1_2_amplitude = torch.abs(torch.fft.fftn(c1_2))
        c1_2_phase = torch.angle(torch.fft.fftn(c1_2))

        c2_2_phase = torch.angle(torch.fft.fftn(c2_2))

        new_2_amplitude = self.Transformer2(c1_2_amplitude, f_2_amplitude)
        new_2_phase = self.Transformer2(c1_2_phase, c2_2_phase)

        real_part = new_2_amplitude * torch.cos(new_2_phase)
        imaginary_part = new_2_amplitude * torch.sin(new_2_phase)

        complex_spectrum = torch.complex(real_part, imaginary_part)

        f_2_reconstructed = torch.abs(torch.fft.ifftn(complex_spectrum))

        f_reconstructed = torch.cat((f_reconstructed, f_2_reconstructed), dim=1)

        # print(f_reconstructed.shape)

        return self.Conv1_1(torch.cat((f_reconstructed, f1_spatial), dim=1))

class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False, getIntermFeat=False):
        super(NLayerDiscriminator, self).__init__()
        self.getIntermFeat = getIntermFeat
        self.n_layers = n_layers

        kw = 4
        padw = int(np.ceil((kw - 1.0) / 2))
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
                setattr(self, 'model' + str(n), nn.Sequential(*sequence[n]))
        else:
            sequence_stream = []
            for n in range(len(sequence)):
                sequence_stream += sequence[n]
            self.model = nn.Sequential(*sequence_stream)

    def forward(self, input):
        if self.getIntermFeat:
            res = [input]
            for n in range(self.n_layers + 2):
                model = getattr(self, 'model' + str(n))
                res.append(model(res[-1]))
            return res[1:]
        else:
            return self.model(input)


class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.cuda.FloatTensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
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
    re = net([c2, f, c1])

