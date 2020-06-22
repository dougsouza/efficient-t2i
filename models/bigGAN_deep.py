import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
from .modules import ConditionalBatchnorm, SelfAttn, CA_Net


def relu(x, inplace=True):
    return F.leaky_relu(x, inplace=inplace)


class GBlock(nn.Module):
    def __init__(self, in_channels, out_channels,
                embedding_size, activation=relu,
                upsample=False, weight_norm_fn=spectral_norm,
                norm_type=nn.BatchNorm2d, channel_ratio=4):
        super(GBlock, self).__init__()

        self.in_channels, self.out_channels = in_channels, out_channels
        self.hidden_channels = self.in_channels // channel_ratio
        self.embedding_size = embedding_size
        self.activation = activation
        self.upsample = upsample
        self.weight_norm_fn = weight_norm_fn
        self.norm_type = norm_type

        # Conv layers
        self.conv1 = self.weight_norm_fn(nn.Conv2d(self.in_channels,
                            self.hidden_channels, kernel_size=1, padding=0))
        self.conv2 = self.weight_norm_fn(nn.Conv2d(self.hidden_channels,
                            self.hidden_channels, kernel_size=3, padding=1))

        self.conv3 = self.weight_norm_fn(nn.Conv2d(self.hidden_channels,
                            self.hidden_channels, kernel_size=3, padding=1))
        self.conv4 = self.weight_norm_fn(nn.Conv2d(self.hidden_channels,
                            self.out_channels, kernel_size=1, padding=0))
        # Batchnorm layers
        self.bn1 = ConditionalBatchnorm(self.in_channels, self.embedding_size,
                                    self.weight_norm_fn, self.norm_type)
        self.bn2 = ConditionalBatchnorm(self.hidden_channels, self.embedding_size,
                                    self.weight_norm_fn, self.norm_type)
        self.bn3 = ConditionalBatchnorm(self.hidden_channels, self.embedding_size,
                                    self.weight_norm_fn, self.norm_type)
        self.bn4 = ConditionalBatchnorm(self.hidden_channels, self.embedding_size,
                                    self.weight_norm_fn, self.norm_type)


        # upsample layers
        self.upsample = upsample

    def forward(self, x, y):
        # Project down to channel ratio
        h = self.conv1(self.activation(self.bn1(x, y)))
        # Apply next BN-ReLU
        # gammas, betas = self.bn_proj(y)
        h = self.activation(self.bn2(h, y))
        # Drop channels in x if necessary
        if self.in_channels != self.out_channels:
            x = x[:, :self.out_channels]
        # Upsample both h and x at this point
        if self.upsample:
            h = F.interpolate(h, scale_factor=2, mode='nearest')
            x = F.interpolate(x, scale_factor=2, mode='nearest')
        # 3x3 convs
        h = self.conv2(h)
        h = self.conv3(self.activation(self.bn3(h, y)))
        # Final 1x1 conv
        h = self.conv4(self.activation(self.bn4(h, y)))
        return h + x



class DBlock(nn.Module):
    def __init__(self, in_channels, out_channels,
                 activation=relu, downsample=False,
                 weight_norm_fn=spectral_norm, channel_ratio=4):
        super(DBlock, self).__init__()
        self.in_channels, self.out_channels = in_channels, out_channels
        self.hidden_channels = self.out_channels // channel_ratio
        self.weight_norm_fn = weight_norm_fn
        self.activation = activation
        self.downsample = downsample

        # Conv layers
        self.conv1 = self.weight_norm_fn(nn.Conv2d(self.in_channels,
                                        self.hidden_channels,
                                        kernel_size=1, padding=0))
        self.conv2 = self.weight_norm_fn(nn.Conv2d(self.hidden_channels,
                        self.hidden_channels, kernel_size=3, padding=1))

        self.conv3 = self.weight_norm_fn(nn.Conv2d(self.hidden_channels,
                            self.hidden_channels, kernel_size=3, padding=1))
        self.conv4 = self.weight_norm_fn(nn.Conv2d(self.hidden_channels,
                                        self.out_channels,
                                        kernel_size=1, padding=0))

        self.learnable_sc = True if (in_channels != out_channels) else False
        if self.learnable_sc:
            self.conv_sc = self.weight_norm_fn(nn.Conv2d(self.in_channels,
                                        self.out_channels - self.in_channels,
                                        kernel_size=1, padding=0))
        if self.downsample:
            self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2,
                                    ceil_mode=False, count_include_pad=False)

    def shortcut(self, x):
        if self.downsample:
            x = self.avg_pool(x)
        if self.learnable_sc:
            x = torch.cat([x, self.conv_sc(x)], 1)
        return x

    def forward(self, x):
        # 1x1 bottleneck conv
        h = self.conv1(self.activation(x, inplace=False))
        # 3x3 convs
        h = self.conv2(self.activation(h))
        h = self.conv3(self.activation(h))
        # relu before downsample
        h = self.activation(h)
        # downsample
        if self.downsample:
            h = self.avg_pool(h)
        # final 1x1 conv
        h = self.conv4(h)
        return h + self.shortcut(x)



class Discriminator(nn.Module):
    def __init__(self,
                 ndf=96,
                 embedding_size=256,
                 activation=relu):
        super(Discriminator, self).__init__()
        self.weight_norm_fn = lambda x: spectral_norm(x, eps=1e-05)
        self.embedding_size = embedding_size
        self.activation = activation

        # 256
        self.block1 = self.weight_norm_fn(nn.Conv2d(3, ndf,
                                            kernel_size=3, padding=1))

        self.block2 = DBlock(ndf, ndf * 2,
                             activation=self.activation,
                             downsample=True,
                             weight_norm_fn=self.weight_norm_fn)
        # 128
        self.block3 = DBlock(ndf * 2, ndf * 2,
                             activation=self.activation,
                             downsample=False,
                             weight_norm_fn=self.weight_norm_fn)
        self.block4 = DBlock(ndf * 2, ndf * 4,
                             activation=self.activation,
                             downsample=True,
                             weight_norm_fn=self.weight_norm_fn)
        # 64
        self.block5 = DBlock(ndf * 4, ndf * 4,
                             activation=self.activation,
                             downsample=False,
                             weight_norm_fn=self.weight_norm_fn)
        
        self.non_local = SelfAttn(ndf * 4, weight_norm_fn=self.weight_norm_fn)

        self.block6 = DBlock(ndf * 4, ndf * 8,
                             activation=self.activation,
                             downsample=True,
                             weight_norm_fn=self.weight_norm_fn)
        
        self.block7 = DBlock(ndf * 8, ndf * 8,
                             activation=self.activation,
                             downsample=False,
                             weight_norm_fn=self.weight_norm_fn)
        self.block8 = DBlock(ndf * 8, ndf * 8,
                             activation=self.activation,
                             downsample=True,
                             weight_norm_fn=self.weight_norm_fn)
        # 16
        self.block9 = DBlock(ndf * 8, ndf * 8,
                             activation=self.activation,
                             downsample=False,
                             weight_norm_fn=self.weight_norm_fn)
        self.block10 = DBlock(ndf * 8, ndf * 16,
                             activation=self.activation,
                             downsample=True,
                             weight_norm_fn=self.weight_norm_fn)
        # 8
        self.block11 = DBlock(ndf * 16, ndf * 16,
                             activation=self.activation,
                             downsample=False,
                             weight_norm_fn=self.weight_norm_fn)
        self.block12 = DBlock(ndf * 16, ndf * 16,
                             activation=self.activation,
                             downsample=True,
                             weight_norm_fn=self.weight_norm_fn)
        # 4
        self.block13 = DBlock(ndf * 16, ndf * 16,
                             activation=self.activation,
                             downsample=False,
                             weight_norm_fn=self.weight_norm_fn)

        self.main = nn.ModuleList([self.block1, self.block2, self.block3,
                                  self.block4, self.block5, self.non_local,
                                  self.block6, self.block7, self.block8,
                                  self.block9, self.block10, self.block11,
                                  self.block12, self.block13])

        self.emb_proj = self.weight_norm_fn(nn.Linear(self.embedding_size, ndf * 16))
        self.linear_out = self.weight_norm_fn(nn.Linear(ndf * 16, 1))


    def forward(self, x, embedding):
        h = x
        for m in self.main:
            h = m(h)
        gp = torch.sum(self.activation(h, inplace=False), dim=(2, 3))
        out = self.linear_out(gp)
        cond = self.emb_proj(embedding)
        cond = torch.sum(cond * gp, dim=1, keepdim=True)
        out += cond
        return out.view(-1)




class Generator(nn.Module):
    def __init__(self,
                 ngf=96,
                 z_dim=128,
                 bottom_width=4,
                 activation=relu,
                 embedding_size=256,
                 norm='batch'):
        super(Generator, self).__init__()
        self.bottom_width = bottom_width
        self.activation = activation
        self.z_dim = z_dim
        self.embedding_size = embedding_size
        self.condition_dim = self.embedding_size
        self.weight_norm_fn = lambda x: spectral_norm(x, eps=1e-05)
        self.norm_type = nn.BatchNorm2d if norm == 'batch' else nn.InstanceNorm2d


        self.block1 = self.weight_norm_fn(nn.Linear(self.z_dim,
                                             self.bottom_width**2 * ngf * 16))

        # 4
        self.block2 = GBlock(ngf * 16, ngf * 16,
                             upsample=False,
                             embedding_size=self.condition_dim + self.z_dim,
                             activation=self.activation,
                             weight_norm_fn=self.weight_norm_fn,
                             norm_type=self.norm_type)
        self.block3 = GBlock(ngf * 16, ngf * 16,
                             upsample=True,
                             embedding_size=self.condition_dim + self.z_dim,
                             activation=self.activation,
                             weight_norm_fn=self.weight_norm_fn,
                             norm_type=self.norm_type)
        # 8
        self.block4 = GBlock(ngf * 16, ngf * 16,
                             upsample=False,
                             embedding_size=self.condition_dim + self.z_dim,
                             activation=self.activation,
                             weight_norm_fn=self.weight_norm_fn,
                             norm_type=self.norm_type)
        self.block5 = GBlock(ngf * 16, ngf * 8,
                             upsample=True,
                             embedding_size=self.condition_dim + self.z_dim,
                             activation=self.activation,
                             weight_norm_fn=self.weight_norm_fn,
                             norm_type=self.norm_type)
        # 16
        self.block6 = GBlock(ngf * 8, ngf * 8,
                             upsample=False,
                             embedding_size=self.condition_dim + self.z_dim,
                             activation=self.activation,
                             weight_norm_fn=self.weight_norm_fn,
                             norm_type=self.norm_type)
        self.block7 = GBlock(ngf * 8, ngf * 8,
                             upsample=True,
                             embedding_size=self.condition_dim + self.z_dim,
                             activation=self.activation,
                             weight_norm_fn=self.weight_norm_fn,
                             norm_type=self.norm_type)
        # 32
        self.block8 = GBlock(ngf * 8, ngf * 8,
                             upsample=False,
                             embedding_size=self.condition_dim + self.z_dim,
                             activation=self.activation,
                             weight_norm_fn=self.weight_norm_fn,
                             norm_type=self.norm_type)
        
        self.block9 = GBlock(ngf * 8, ngf * 4,
                             upsample=True,
                             embedding_size=self.condition_dim + self.z_dim,
                             activation=self.activation,
                             weight_norm_fn=self.weight_norm_fn,
                             norm_type=self.norm_type)
        # 64
        self.non_local = SelfAttn(ngf * 4, self.weight_norm_fn)
        self.block10 = GBlock(ngf * 4, ngf * 4,
                             upsample=False,
                             embedding_size=self.condition_dim + self.z_dim,
                             activation=self.activation,
                             weight_norm_fn=self.weight_norm_fn,
                             norm_type=self.norm_type)
        self.block11 = GBlock(ngf * 4, ngf * 2,
                             upsample=True,
                             embedding_size=self.condition_dim + self.z_dim,
                             activation=self.activation,
                             weight_norm_fn=self.weight_norm_fn,
                             norm_type=self.norm_type)
        # 128
        self.block12 = GBlock(ngf * 2, ngf * 2,
                             upsample=False,
                             embedding_size=self.condition_dim + self.z_dim,
                             activation=self.activation,
                             weight_norm_fn=self.weight_norm_fn,
                             norm_type=self.norm_type)
        self.block13 = GBlock(ngf * 2, ngf,
                             upsample=True,
                             embedding_size=self.condition_dim + self.z_dim,
                             activation=self.activation,
                             weight_norm_fn=self.weight_norm_fn,
                             norm_type=self.norm_type)
        # 256
        self.bn = self.norm_type(ngf, affine=True)
        self.block14 = self.weight_norm_fn(nn.Conv2d(in_channels=ngf,
                            out_channels=3, kernel_size=3, padding=1))

        modules = [
            self.block2, self.block3, self.block4,
            self.block5, self.block6, self.block7,
            self.block8, self.block9,  self.non_local,
            self.block10, self.block11, self.block12,
            self.block13
        ]
        
        self.main = nn.ModuleList(modules)

    def forward(self, z, sent_emb):
        cond = torch.cat([sent_emb, z], dim=1)
        h = self.block1(z).view(z.size(0), -1, 4, 4)
        for m in self.main:
            h = m(h, cond)
        h = self.activation(self.bn(h))
        h = self.block14(h)
        return torch.tanh(h)
