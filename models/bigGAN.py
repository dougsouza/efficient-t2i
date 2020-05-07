import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
from .modules import ConditionalBatchnorm, SelfAttn, CA_Net


def relu(x, inplace=True):
    return F.relu(x, inplace=inplace)


class GBlock(nn.Module):
    def __init__(self, in_channels, out_channels,
                embedding_size, activation=relu,
                upsample=False, weight_norm_fn=spectral_norm,
                norm_type=nn.BatchNorm2d):
        super(GBlock, self).__init__()
        self.in_channels, self.out_channels = in_channels, out_channels
        self.activation = activation
        self.upsample = upsample
        self.embedding_size = embedding_size
        self.weight_norm_fn = weight_norm_fn
        self.norm_type = norm_type

        self.conv1 = self.weight_norm_fn(nn.Conv2d(self.in_channels,
                                self.out_channels, kernel_size=3, padding=1))
        self.conv2 = self.weight_norm_fn(nn.Conv2d(self.out_channels,
                                self.out_channels, kernel_size=3, padding=1))
        self.learnable_sc = in_channels != out_channels or upsample
        if self.learnable_sc:
            self.conv_sc = self.weight_norm_fn(nn.Conv2d(in_channels,
                                    out_channels, kernel_size=1, padding=0))

        self.bn1 = ConditionalBatchnorm(self.in_channels, self.embedding_size,
                                        self.weight_norm_fn, self.norm_type)
        self.bn2 = ConditionalBatchnorm(self.out_channels, self.embedding_size,
                                        self.weight_norm_fn, self.norm_type)

    def forward(self, x, y):
        h = self.activation(self.bn1(x, y))
        if self.upsample:
            h = F.interpolate(h, scale_factor=2, mode='nearest')
            x = F.interpolate(x, scale_factor=2, mode='nearest')
        h = self.conv1(h)
        h = self.activation(self.bn2(h, y))
        h = self.conv2(h)
        if self.learnable_sc:       
            x = self.conv_sc(x)
        return h + x


class DBlock(nn.Module):
    def __init__(self, in_channels, out_channels,
                    activation=relu, downsample=False,
                    weight_norm_fn=spectral_norm, channel_ratio=4):
        super(DBlock, self).__init__()
        self.in_channels, self.out_channels = in_channels, out_channels
        self.hidden_channels = self.out_channels
        self.activation = activation
        self.downsample = downsample
        self.weight_norm_fn = weight_norm_fn
        # Conv layers
        self.conv1 = self.weight_norm_fn(nn.Conv2d(self.in_channels,
                            self.hidden_channels, kernel_size=3, padding=1))
        self.conv2 = self.weight_norm_fn(nn.Conv2d(self.hidden_channels,
                            self.out_channels, kernel_size=3, padding=1))
        self.learnable_sc = True if (in_channels != out_channels) or downsample else False
        if self.learnable_sc:
            self.conv_sc = self.weight_norm_fn(nn.Conv2d(in_channels, out_channels, 
                                                kernel_size=1, padding=0))
        if self.downsample:
            self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2,
                                    ceil_mode=False, count_include_pad=False)
    def shortcut(self, x):
        if self.learnable_sc:
            x = self.conv_sc(x)
            if self.downsample:
                x = self.avg_pool(x)
        return x

    def forward(self, x):
        h = x
        h = self.activation(h, inplace=False)
        h = self.conv1(h)
        h = self.activation(h)
        h = self.conv2(h)
        if self.downsample:
            h = self.avg_pool(h)
        return h + self.shortcut(x)



class Discriminator(nn.Module):
    def __init__(self,
                 ndf=64,
                 embedding_size=256,
                 activation=relu):
        super(Discriminator, self).__init__()
        self.weight_norm_fn = spectral_norm
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
        self.block3 = DBlock(ndf * 2, ndf * 4,
                             activation=self.activation,
                             downsample=True,
                             weight_norm_fn=self.weight_norm_fn)
        # 64
        self.non_local = SelfAttn(ndf * 4, weight_norm_fn=self.weight_norm_fn)

        self.block4 = DBlock(ndf * 4, ndf * 8,
                             activation=self.activation,
                             downsample=True,
                             weight_norm_fn=self.weight_norm_fn)
        # 32
        self.block5 = DBlock(ndf * 8, ndf * 8,
                             activation=self.activation,
                             downsample=True,
                             weight_norm_fn=self.weight_norm_fn)
        # 16
        self.block6 = DBlock(ndf * 8, ndf * 16,
                             activation=self.activation,
                             downsample=True,
                             weight_norm_fn=self.weight_norm_fn)
        # 8
        self.block7 = DBlock(ndf * 16, ndf * 16,
                             activation=self.activation,
                             downsample=True,
                             weight_norm_fn=self.weight_norm_fn)
        # 4
        self.block8 = DBlock(ndf * 16, ndf * 16,
                             activation=self.activation,
                             downsample=False,
                             weight_norm_fn=self.weight_norm_fn)


        self.main = nn.Sequential(self.block1, self.block2, self.block3,
                                  self.non_local, self.block4, self.block5,
                                  self.block6, self.block7, self.block8)


        # self.emb_proj = self.weight_norm_fn(nn.Linear(self.embedding_size,
        #                                                             ndf * 16))
        self.linear_out = self.weight_norm_fn(nn.Linear(ndf * 16, 1))

        self.emb_linear = self.weight_norm_fn(nn.Linear(ndf * 16,
                                                    self.embedding_size))


    def forward(self, x, embedding):
        h = self.main(x)
        gp = torch.sum(self.activation(h, inplace=False), dim=(2, 3))
        # cond = self.emb_proj(embedding)
        # cond = torch.sum(cond * gp, dim=1, keepdim=True)
        # out = self.linear_out(gp) + cond
        out = self.linear_out(gp)
        return out.view(-1), self.emb_linear(gp)




class Generator(nn.Module):
    def __init__(self,
                 ngf=64,
                 z_dim=140,
                 bottom_width=4,
                 activation=relu,
                 embedding_size=256,
                 norm='batch',
                 cond_aug=False):
        super(Generator, self).__init__()
        assert z_dim % 7 == 0, \
                'for biggan, z_dim must be a multiple of 7, 140 is recomended'
        self.bottom_width = bottom_width
        self.activation = activation
        self.z_dim = z_dim//7
        self.embedding_size = embedding_size
        self.condition_dim = self.embedding_size
        self.weight_norm_fn = spectral_norm
        self.norm_type = nn.BatchNorm2d if norm == 'batch' else nn.InstanceNorm2d
        self.cond_aug = cond_aug

        if self.cond_aug:
            self.ca_net = CA_Net(self.embedding_size,
                            self.embedding_size//2, self.weight_norm_fn)
            self.condition_dim = self.embedding_size//2

        self.block1 = self.weight_norm_fn(nn.Linear(self.z_dim,
                                            self.bottom_width**2 * ngf * 16))


        # 4
        self.block2 = GBlock(ngf * 16, ngf * 16,
                             upsample=True,
                             embedding_size=self.condition_dim + self.z_dim,
                             activation=self.activation,
                             weight_norm_fn=self.weight_norm_fn,
                             norm_type=self.norm_type)
        # 8
        self.block3 = GBlock(ngf * 16, ngf * 8,
                             upsample=True,
                             embedding_size=self.condition_dim + self.z_dim,
                             activation=self.activation,
                             weight_norm_fn=self.weight_norm_fn,
                             norm_type=self.norm_type)
        # 16
        self.block4 = GBlock(ngf * 8, ngf * 8,
                             upsample=True,
                             embedding_size=self.condition_dim + self.z_dim,
                             activation=self.activation,
                             weight_norm_fn=self.weight_norm_fn,
                             norm_type=self.norm_type)
        # 32
        self.block5 = GBlock(ngf * 8, ngf * 4,
                             upsample=True,
                             embedding_size=self.condition_dim + self.z_dim,
                             activation=self.activation,
                             weight_norm_fn=self.weight_norm_fn,
                             norm_type=self.norm_type)
        # 64
        self.non_local = SelfAttn(ngf * 4, self.weight_norm_fn)

        self.block6 = GBlock(ngf * 4, ngf * 2,
                             upsample=True,
                             embedding_size=self.condition_dim + self.z_dim,
                             activation=self.activation,
                             weight_norm_fn=self.weight_norm_fn,
                             norm_type=self.norm_type)
        # 128
        self.block7 = GBlock(ngf * 2, ngf,
                             upsample=True,
                             embedding_size=self.condition_dim + self.z_dim,
                             activation=self.activation,
                             weight_norm_fn=self.weight_norm_fn,
                             norm_type=self.norm_type)
        # 256
        self.bn = self.norm_type(ngf, affine=True)
        self.block14 = self.weight_norm_fn(nn.Conv2d(in_channels=ngf,
                            out_channels=3, kernel_size=3, padding=1))

        self.main = nn.ModuleList([
            self.block2, self.block3, self.block4,
            self.block5, self.non_local, self.block6,
            self.block7])


    def forward(self, z, sent_emb, word_embs):
        if self.cond_aug:
            cond, mu, logvar = self.ca_net(sent_emb)
        zs = torch.split(z, self.z_dim, dim=1)
        h = self.block1(zs[0]).view(z.size(0), -1, 4, 4)
        zidx = 1
        for m in self.main:
            h = m(h, torch.cat([cond, zs[zidx]], dim=1))
            zidx += 1
        h = self.activation(self.bn(h))
        h = self.block14(h)
        if self.cond_aug:
            return torch.tanh(h), mu, logvar
        return torch.tanh(h)
