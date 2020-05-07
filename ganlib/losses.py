from abc import ABC, abstractmethod
import torch
import torch.nn as nn
import torch.nn.functional as F


class Loss(ABC):
    def __init__(self, device):
        super(Loss, self).__init__()
        self.device = device

    @abstractmethod
    def loss_d_real(self, logits_real, logits_fake=None):
        pass

    @abstractmethod
    def loss_d_fake(self, logits_fake, logits_real=None):
        pass

    @abstractmethod
    def loss_d_additional(self, forward_fn, input_real, input_fake):
        pass

    @abstractmethod
    def loss_g(self, logits_fake, logits_real=None):
        pass

    @abstractmethod
    def loss_g_additional(self, forward_fn, input_fake, input_real):
        pass


class LSGAN(Loss):
    def __init__(self, device):
        super(LSGAN, self).__init__(device)

    def loss_d_real(self, logits_real, logits_fake=None):
        ones = torch.ones(logits_real.size(0), device=self.device)
        return 0.5 * torch.mean((logits_real - ones) ** 2), False

    def loss_d_fake(self, logits_fake, logits_real=None):
        zeros = torch.zeros(logits_fake.size(0), device=self.device)
        return 0.5 * torch.mean((logits_fake - zeros) ** 2), False

    def loss_d_additional(self, forward_fn, input_real, input_fake):
        return None, False

    def loss_g(self, logits_fake, logits_real=None):
        ones = torch.ones(logits_real.size(0), device=self.device)
        return 0.5 * torch.mean((logits_fake - ones) ** 2), False

    def loss_g_additional(self, forward_fn, input_fake, input_real):
        return None, False



class WGANGP(Loss):
    def __init__(self, device, lambda_gp=10):
        super(WGANGP, self).__init__(device)
        self.lambda_gp = lambda_gp
        self.retain_graph = False

    def loss_d_real(self, logits_real, logits_fake=None):
        return -logits_real.mean(), self.retain_graph

    def loss_d_fake(self, logits_fake, logits_real=None):
        return logits_fake.mean(), self.retain_graph

    def loss_d_additional(self, forward_fn, input_real, input_fake):
        alpha = torch.rand(input_real.size(0), 1, 1, 1, device=self.device)
        interpolates = torch.tensor(input_real.clone().detach() * (1 - alpha) + input_fake * alpha, 
                                    requires_grad=True)

        disc_interpolates = forward_fn(interpolates)

        gradients = torch.autograd.grad(outputs=disc_interpolates,
                                        inputs=interpolates,
                                        grad_outputs=torch.ones(disc_interpolates.size(), device=self.device),
                                        create_graph=True,
                                        retain_graph=True,
                                        only_inputs=True)[0]
        
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1.) ** 2).mean() * self.lambda_gp
        return gradient_penalty, self.retain_graph


    def loss_g(self, logits_fake, logits_real=None):
        return -logits_fake.mean(), self.retain_graph

    def loss_g_additional(self, forward_fn, input_fake, input_real):
        return None, self.retain_graph


class WGAN(Loss):
    def __init__(self, device):
        super(WGAN, self).__init__(device)
        self.retain_graph = False

    def loss_d_real(self, logits_real, logits_fake=None):
        return -logits_real.mean(), self.retain_graph

    def loss_d_fake(self, logits_fake, logits_real=None):
        return logits_fake.mean(), self.retain_graph

    def loss_d_additional(self, forward_fn, input_real, input_fake):
        return None, self.retain_graph

    def loss_g(self, logits_fake, logits_real=None):
        return -logits_fake.mean(), self.retain_graph

    def loss_g_additional(self, forward_fn, input_fake, input_real):
        return None, self.retain_graph


class HingeGAN(Loss):
    
    def __init__(self, device):
        super(HingeGAN, self).__init__(device)
        self.retain_graph = False

    def loss_d_real(self, logits_real, logits_fake=None):
        return F.relu_(1.0 - logits_real).mean(), self.retain_graph
    
    def loss_d_fake(self, logits_fake, logits_real=None):
        return F.relu_(1.0 + logits_fake).mean(), self.retain_graph
    
    def loss_d_additional(self, forward_fn, input_real, input_fake):
        return None, self.retain_graph
    
    def loss_g(self, logits_fake, logits_real=None):
        return -logits_fake.mean(), self.retain_graph

    def loss_g_additional(self, forward_fn, input_fake, input_real):
        return None, self.retain_graph


class R1SGAN(Loss):
    
    def __init__(self, device, reg_lambda=10.):
        super(R1SGAN, self).__init__(device)
        self.device = device
        self.bce = nn.BCEWithLogitsLoss()
        self.reg_lambda = reg_lambda

    def loss_d_real(self, logits_real, logits_fake=None):
        self.logits_real = logits_real
        return self.bce(logits_real, torch.ones(logits_real.size(0), device=self.device)), True
    
    def loss_d_fake(self, logits_fake, logits_real=None):
        return self.bce(logits_fake, torch.zeros(logits_fake.size(0), device=self.device)), False
    
    def loss_d_additional(self, forward_fn, input_real, input_fake):
        # logits_real, cls_real = forward_fn(input_real.requires_grad_())
        reg = self.compute_grad2(self.logits_real, input_real)
        reg = self.reg_lambda * reg.mean()
        return reg, False
    
    def compute_grad2(self, d_out, x_in):
        batch_size = x_in.size(0)
        grad_dout = torch.autograd.grad(
            outputs=d_out.sum(), inputs=x_in,
            create_graph=True, retain_graph=True, only_inputs=True
        )[0]
        grad_dout2 = grad_dout.pow(2)
        assert(grad_dout2.size() == x_in.size())
        reg = grad_dout2.view(batch_size, -1).sum(1)
        return reg

    def loss_g(self, logits_fake, logits_real=None):
        return self.bce(logits_fake, torch.ones(logits_fake.size(0), device=self.device)), None

    def loss_g_additional(self, forward_fn, input_fake, input_real):
        return None, False


class SGAN(Loss):
    
    def __init__(self, device, reg_lambda=10.):
        super(SGAN, self).__init__(device)
        self.device = device
        self.bce = nn.BCEWithLogitsLoss()

    def loss_d_real(self, logits_real, logits_fake=None):
        return self.bce(logits_real, torch.ones(logits_real.size(0), device=self.device)), False
    
    def loss_d_fake(self, logits_fake, logits_real=None):
        return self.bce(logits_fake, torch.zeros(logits_fake.size(0), device=self.device)), False
    
    def loss_d_additional(self, forward_fn, input_real, input_fake):
        return None, False
    
    def loss_g(self, logits_fake, logits_real=None):
        return self.bce(logits_fake, torch.ones(logits_fake.size(0), device=self.device)), None

    def loss_g_additional(self, forward_fn, input_fake, input_real):
        return None, False


class RelativisticAverageGAN(Loss):
    def __init__(self, device):
        super(RelativisticAverageGAN, self).__init__(device)
        self.bce = nn.BCEWithLogitsLoss()
        self.device = device

    def loss_d_real(self, logits_real, logits_fake=None):
        ones = torch.ones(logits_real.size(0), device=self.device)
        return self.bce(logits_real - logits_fake.mean(), ones)/2., True
    
    def loss_d_fake(self, logits_fake, logits_real=None):
        zeros = torch.zeros(logits_fake.size(0), device=self.device)
        return self.bce(logits_fake - logits_real.mean(), zeros)/2., True
    
    def loss_d_additional(self, forward_fn, input_real, input_fake):
        return None, False
    
    def loss_g(self, logits_fake, logits_real=None):
        ones = torch.ones(logits_real.size(0), device=self.device)
        zeros = torch.zeros(logits_fake.size(0), device=self.device)
        return (self.bce(logits_real - logits_fake.mean(), zeros) + \
                self.bce(logits_fake - logits_real.mean(), ones))/2., False
                

    def loss_g_additional(self, forward_fn, input_fake, input_real):
        return None, False


class RelativisticAverageLSGAN(Loss):
    def __init__(self, device):
        super(RelativisticAverageLSGAN, self).__init__(device)

    def loss_d_real(self, logits_real, logits_fake=None):
        ones = torch.ones(logits_real.size(0), device=self.device)
        return torch.mean((logits_real - logits_fake.mean() - ones) ** 2)/2., True
    
    def loss_d_fake(self, logits_fake, logits_real=None):
        ones = torch.ones(logits_real.size(0), device=self.device)
        return torch.mean((logits_fake - logits_real.mean() + ones) ** 2)/2., True
    
    def loss_d_additional(self, forward_fn, input_real, input_fake):
        return None, False
    
    def loss_g(self, logits_fake, logits_real=None):
        ones = torch.ones(logits_real.size(0), device=self.device)
        return (torch.mean((logits_real - logits_fake.mean() + ones) ** 2) + \
                torch.mean((logits_fake - logits_real.mean() - ones) ** 2))/2., False
                
    def loss_g_additional(self, forward_fn, input_fake, input_real):
        return None, False


class RelativisticAverageHingeGAN(Loss):
    def __init__(self, device):
        super(RelativisticAverageHingeGAN, self).__init__(device)

    def loss_d_real(self, logits_real, logits_fake=None):
        return torch.mean(F.relu_(1.0 - (logits_real - logits_fake.mean())))/2., True
    
    def loss_d_fake(self, logits_fake, logits_real=None):
        return torch.mean(F.relu_(1.0 + (logits_fake - logits_real.mean())))/2., True
    
    def loss_d_additional(self, forward_fn, input_real, input_fake):
        return None, False
    
    def loss_g(self, logits_fake, logits_real=None):
        return (torch.mean(F.relu_(1.0 + (logits_real - logits_fake.mean()))) + \
                torch.mean(F.relu_(1.0 - (logits_fake - logits_real.mean()))))/2., False
                
    def loss_g_additional(self, forward_fn, input_fake, input_real):
        return None, False




__lossdict__ = {
    'wgan': WGAN,
    'wgangp': WGANGP,
    'hingegan': HingeGAN,
    'rasgan': RelativisticAverageGAN,
    'ralsgan': RelativisticAverageLSGAN,
    'rahingegan': RelativisticAverageHingeGAN,
    'r1sgan': R1SGAN,
    'lsgan': LSGAN,
    'sgan': SGAN
}


def get_loss(loss, device):
    return __lossdict__[loss](device)


def get_losses_names():
    return __lossdict__.keys()
