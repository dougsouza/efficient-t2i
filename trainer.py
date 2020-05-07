import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from tqdm import tqdm
import torchvision.utils as vutils
import itertools
from utils.models import get_current_LR
from utils.models import isnan
import numpy as np



class Trainer():
    def __init__(self, config, init_iter, netD, netG, 
                 dataset, z_dim, loss, sampler_fn, optimizerD, 
                 optimizerG, logdir, device, netG_avg, netG_params):
        self.config = config
        self.init_iter = init_iter
        self.netD = self.ckpt_D = netD
        self.netG = self.ckpt_G = netG
        self.z_dim = z_dim
        self.netG_avg = netG_avg
        self.dataset = dataset
        self.loss = loss
        self.sampler_fn = sampler_fn
        self.optimizerD = optimizerD
        self.optimizerG = optimizerG
        self.logdir = logdir
        self.device = device
        self.batch_size = config.batch_size
        self.num_workers = config.num_workers
        self.ngpu = config.ngpu
        self.netG_params = netG_params
        self.local_rank = self.config.local_rank
        
        if self.ngpu > 1:
            if config.fp16:
                from apex.parallel import DistributedDataParallel as DDP
                from apex import amp
                self.netD = DDP(netD, delay_allreduce=True)
                self.netG = DDP(netG, delay_allreduce=True)
            else:
                self.netD = nn.parallel.DistributedDataParallel(netD,
                    device_ids=[self.local_rank], output_device=self.local_rank)
                self.netG = nn.parallel.DistributedDataParallel(netG,
                    device_ids=[self.local_rank], output_device=self.local_rank)

        if self.local_rank == 0:
            self.summary_writer = SummaryWriter(self.logdir)

        if self.ngpu > 1:
            self.sampler = torch.utils.data.distributed.DistributedSampler(self.dataset,
                                    num_replicas=self.config.ngpu, rank=self.local_rank)
        else:
            self.sampler = None
        self.data_loader = DataLoader(self.dataset, batch_size=self.batch_size, 
                                num_workers=self.num_workers, pin_memory=True,
                                sampler=self.sampler, drop_last=True,
                                shuffle=self.sampler is None)
        
        self.log_step = config.log_step
        self.n_epochs = config.n_epochs
        self.d_steps = config.d_steps
        self.save_sample_step = config.save_sample_step
        self.save_checkpoint_ep = config.save_checkpoint_ep
        self.n_samples = config.n_samples


    def tqdm_fn(self, x, **kwargs):
        if self.local_rank == 0:
            return tqdm(x, **kwargs)
        else:
            return x

    def run(self):
        fixed_z = self.sampler_fn(self.n_samples, 
                            self.z_dim).to(self.device)
        fixed_y = []
        for i in range(self.n_samples):
            emb = self.dataset.embeddings[i][0]
            fixed_y.append(torch.tensor(emb).unsqueeze(0))

        fixed_y = torch.cat(fixed_y, dim=0).to(self.device)
        fixed_mask = None

        if self.config.fp16:
            fixed_z, fixed_y = fixed_z.half(), fixed_y.half()

        it_count = 0
        for ep in self.tqdm_fn(range(self.init_iter, self.n_epochs), 
                             initial=self.init_iter,
                             total=self.n_epochs):

            if (ep  % self.save_checkpoint_ep == 0 or ep == (self.n_epochs-1)) and self.local_rank == 0:
                tqdm.write('saving checkpoints...')
                self.save_ckpt(ep, self.optimizerG, self.ckpt_G,
                            self.logdir / 'netG_epoch_{}.pth'.format(ep))
                self.save_ckpt(ep, self.optimizerD, self.ckpt_D,
                            self.logdir / 'netD_epoch_{}.pth'.format(ep))
                if self.netG_avg is not None:
                    self.save_ckpt(ep, None, self.netG_avg,
                            self.logdir / 'netG_avg_epoch_{}.pth'.format(ep),
                            netG_params=self.netG_params)
            
            d_steps = 0
            for it, (x, sent_emb) in enumerate(self.tqdm_fn(self.data_loader)):
                for p in self.netD.parameters():
                    p.requires_grad = True
                x, sent_emb = x.to(self.device), sent_emb.to(self.device)

                self.netD.zero_grad()
                # ---- update Discriminator
                with torch.no_grad():
                    z = self.sampler_fn(x.size(0), self.z_dim)
                    fake = self.netG(z, sent_emb)

                d_logits_fake = self.netD(fake, sent_emb)
        
                d_loss_fake, retain_graph = self.loss.loss_d_fake(d_logits_fake, None)
                d_loss_fake.backward(retain_graph=retain_graph)

                d_logits_real = self.netD(x, sent_emb)
                d_loss_real, retain_graph = self.loss.loss_d_real(d_logits_real, None)
                d_loss_real.backward(retain_graph=False)
                
                d_loss = d_loss_real.item() + d_loss_fake.item()
                self.optimizerD.step()

                d_steps +=1
                if d_steps < self.d_steps:
                    update_g = False
                else:
                    update_g = True
                    d_steps = 0

                if update_g:
                    for p in self.netD.parameters():
                        p.requires_grad = False

                    # # --- update Generator
                    self.netG.zero_grad()
                    z = self.sampler_fn(sent_emb.size(0), self.z_dim)
                    fake = self.netG(z, sent_emb)
                    d_logits_fake = self.netD(fake, sent_emb)
                    g_loss, retain_graph = self.loss.loss_g(d_logits_fake, None)
                    g_loss.backward(retain_graph=retain_graph)
                    self.optimizerG.step()



                if self.netG_avg:
                    self.update_EMA(self.netG_avg, self.netG, self.device)
            
                if it_count % self.log_step == 0 and self.local_rank == 0 and it > self.d_steps:
                    tqdm.write('[%d] Loss_D: %.4f Loss_G: %.4f Loss_D_real: ' \
                                '%.4f Loss_D_fake %.4f' % (it_count, 
                                d_loss, g_loss.item(), d_loss_real.item(),
                                d_loss_fake.item()))
                    self.summary_writer.add_scalar('D/D_LR',
                                                    get_current_LR(self.optimizerD),
                                                    it_count)
                    self.summary_writer.add_scalar('D/D_loss',
                                                    d_loss, it_count)
                    self.summary_writer.add_scalar('D/D_loss_fake',
                                                    d_loss_fake.item(), it_count)
                    self.summary_writer.add_scalar('D/D_loss_real',
                                                    d_loss_real.item(), it_count)

                    self.summary_writer.add_scalar('G/G_LR',
                                                    get_current_LR(self.optimizerG),
                                                    it_count)
                    self.summary_writer.add_scalar('G/G_loss', 
                                                    g_loss.item(), it_count)
                
                
                if it_count % self.save_sample_step == 0:
                    if self.local_rank == 0:
                        tqdm.write("Saving samples...")
                    self.netG.eval()
                    with torch.no_grad():
                        fake = self.netG(fixed_z, fixed_y)
                        fake = fake.mul(0.5).add(0.5).clamp_(0, 1).cpu()
                        if self.local_rank == 0:
                            self.summary_writer.add_image('G/Samples', 
                                    vutils.make_grid(fake, nrow=4), it_count)
                    self.netG.train()
                    if self.netG_avg is not None:
                        with torch.no_grad():
                            fake = self.netG_avg(fixed_z, fixed_y)
                            fake = fake.mul(0.5).add(0.5).clamp_(0, 1).cpu()
                            if self.local_rank == 0:
                                self.summary_writer.add_image('G/Avg_Samples', 
                                    vutils.make_grid(fake, nrow=4), it_count)                
                it_count += 1


    def save_ckpt(self, it, optim, net, fname, **kwargs):
        opt_state_dict = optim.state_dict() if optim is not None else None
        ckpt = {
                'it': it,
                'optimizer': opt_state_dict,
                'net': net.state_dict(),
                }
        ckpt.update(kwargs)
        torch.save(ckpt, fname)
    
    def update_EMA(self, avg_net, net, device, mu=0.999):
        for (avg_k, avg_v), (k, v) in zip(avg_net.state_dict().items(),
                                                net.state_dict().items()):
            if ('weight' in avg_k or 'bias' in avg_k) and  \
                            not ('weight_v' in avg_k or 'weight_u' in avg_k):
                avg = (1.0 - mu) * v.to(device).data + mu * avg_v.data
                avg_v.data.copy_(avg)
            else:
                avg_v.data.copy_(v.to(device))
