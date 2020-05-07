import torch
import torch.nn as nn
import torch.nn.functional as F


class CA_Net(nn.Module):
    # some code is modified from vae examples
    # (https://github.com/pytorch/examples/blob/master/vae/main.py)
    def __init__(self, embedding_size, out_size, weight_norm_fn):
        super(CA_Net, self).__init__()
        self.embedding_size = embedding_size
        self.out_size = out_size
        self.weight_norm_fn = weight_norm_fn
        self.fc = self.weight_norm_fn(nn.Linear(self.embedding_size,
                                                self.out_size * 2, bias=True))

    def encode(self, embedding):
        x = F.relu(self.fc(embedding))
        mu, logvar = torch.split(x, self.out_size, dim=1)
        return mu, logvar

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = torch.randn(*std.size(), device=mu.get_device())
        return eps.mul(std).add_(mu)

    def forward(self, text_embedding):
        mu, logvar = self.encode(text_embedding)
        condition = self.reparametrize(mu, logvar)
        return condition, mu, logvar


class NLayerMLP(nn.Module):
    def __init__(self, in_dim, out_dim, weight_norm_fn,
                 activation, n_layers=6, hidden_dim=512):
        super(NLayerMLP, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.weight_norm_fn = weight_norm_fn
        self.activation = activation
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim

        modules = []
        in_dim = in_dim
        for i in range(n_layers-1):
            modules += [self.weight_norm_fn(nn.Linear(in_dim,
                                                      self.hidden_dim))]
            in_dim = self.hidden_dim

        modules += [self.weight_norm_fn(
            nn.Linear(self.hidden_dim, self.out_dim))]
        self.main = nn.ModuleList(modules)

    def forward(self, x):
        for m in self.main:
            x = self.activation(m(x))
        return x


class SelfAttn(nn.Module):
    def __init__(self, in_dim, weight_norm_fn):
        super(SelfAttn, self).__init__()
        # Channel multiplier
        self.in_dim = in_dim
        self.weight_norm_fn = weight_norm_fn
        self.theta = self.weight_norm_fn(nn.Conv2d(self.in_dim, self.in_dim // 8,
                                                   kernel_size=1, padding=0, bias=False))
        self.phi = self.weight_norm_fn(nn.Conv2d(self.in_dim, self.in_dim // 8,
                                                 kernel_size=1, padding=0, bias=False))
        self.g = self.weight_norm_fn(nn.Conv2d(self.in_dim, self.in_dim // 2,
                                               kernel_size=1, padding=0, bias=False))
        self.o = self.weight_norm_fn(nn.Conv2d(self.in_dim // 2, self.in_dim,
                                               kernel_size=1, padding=0, bias=False))
        # Learnable gain parameter
        self.gamma = nn.Parameter(torch.tensor(0.))

    def forward(self, x, y=None):
        # Apply convs
        theta = self.theta(x)
        phi = F.max_pool2d(self.phi(x), [2, 2])
        g = F.max_pool2d(self.g(x), [2, 2])
        # Perform reshapes
        theta = theta.view(-1, self.in_dim // 8, x.shape[2] * x.shape[3])
        phi = phi.view(-1, self.in_dim // 8, x.shape[2] * x.shape[3] // 4)
        g = g.view(-1, self.in_dim // 2, x.shape[2] * x.shape[3] // 4)
        # Matmul and softmax to get attention maps
        beta = F.softmax(torch.bmm(theta.transpose(1, 2), phi), -1)
        # Attention map times g path
        o = self.o(torch.bmm(g, beta.transpose(1, 2)).view(-1,
                                                           self.in_dim // 2, x.shape[2], x.shape[3]))
        return self.gamma * o + x


class ConditionalBatchnorm(nn.Module):
    def __init__(self, num_features, embedding_size, weight_norm_fn, norm_type):
        super(ConditionalBatchnorm,  self).__init__()
        self.weight_norm_fn = weight_norm_fn
        self.num_features = num_features
        self.embedding_size = embedding_size
        self.norm_type = norm_type
        self.bn = self.norm_type(num_features, affine=False)
        self.gammas_projection = self.weight_norm_fn(nn.Linear(
            self.embedding_size, self.num_features, bias=True))
        self.betas_projection = self.weight_norm_fn(nn.Linear(
            self.embedding_size, self.num_features, bias=True))


    def forward(self, x, emb):
        out = self.bn(x)
        gammas = self.gammas_projection(emb) + 1.0
        # gammas = self.gammas_projection(F.relu(self.proj1(emb))) + 1.0
        betas = self.betas_projection(emb)
        # betas = self.betas_projection(F.relu(self.proj2(emb)))
        out = gammas.view(-1, self.num_features, 1, 1) * out + \
            betas.view(-1, self.num_features, 1, 1)
        return out
