from arguments import read_args
from pathlib import Path
from trainer import Trainer
from ganlib.priors import get_sampler_fn
from ganlib.losses import get_loss
from ganlib.weight_init import get_initializer
from utils.files import setup_dirs
from easydict import EasyDict as edict
from utils.models import load_ckpt, Resize, set_current_LR
from utils.files import store_training_setup
from models import get_dmodel, get_gmodel
from data_loaders import get_loader
from torchvision import transforms
import yaml
import torch
import torch.optim as optim
from PIL import Image


def main(rank, args):
    if args.fp16:
        try:    
            from apex import amp
            import apex
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex.")
                

    print('process of rank {} has been initialzed'.format(rank))
    if args.from_file is not None:
        with open(args.from_file, 'r') as f:
            params = yaml.load(f)
            config = edict(params)
    else:
        config = edict(vars(args))
    config.local_rank = rank
    


    if config.local_rank == 0:
        print(config)


    torch.cuda.set_device(config.local_rank)

    if args.ngpu > 1:
        torch.distributed.init_process_group('nccl', init_method='env://',
                    world_size=config.ngpu, rank=config.local_rank)
    if config.resume is not None and not config.resume_on_new_folder:
        logdir = config.resume.parent
    else:
        logdir = setup_dirs('{}_{}'.format(config.exp_name, config.dataset),
                                        create_folder=config.local_rank == 0)
    
    
    target_tform  = lambda t: torch.tensor(t, dtype=torch.float32)
    tform = transforms.Compose([#transforms.RandomResizedCrop(256,
                                #scale=(0.85, 1.0), ratio=(0.9, 1.1)),
                            transforms.RandomCrop(256),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            transforms.Lambda(lambda img: (img * 2) - 1)])
    dset = get_loader(config.dataset,
                     transform=tform,
                     target_transform=target_tform,
                     interp_sentences=config.interp_sentences)

    device = torch.device('cuda:{}'.format(config.local_rank))
    sampler_fn = get_sampler_fn(config.prior, device=device,
                                normalize=config.normalize_prior)
    loss = get_loss(config.loss, device)
    initializer_fn = get_initializer(config.weight_init)

    netD = get_dmodel(d_model=config.d_model,
                      ndf=config.ndf,
                      conditioning=config.conditioning)
    netD.apply(initializer_fn)
    if config.local_rank == 0:
        print(netD)


    netG_params = {
        'g_model': config.g_model,
        'ngf': config.ngf,
        'z_dim': config.z_dim,
        'norm': config.g_norm,
        'use_attention': config.attention,
        'cond_aug': config.cond_aug
    }
    netG = get_gmodel(**netG_params)
    netG.apply(initializer_fn)
    if config.g_norm == 'batch':
        if config.ngpu > 1:
            if config.fp16:
                netG = apex.parallel.convert_syncbn_model(netG)
            else:
                netG = torch.nn.SyncBatchNorm.convert_sync_batchnorm(netG)
    if config.local_rank == 0:
        print(netG)
    
    params = sum([p.nelement() for p in netD.parameters()])
    if config.local_rank == 0:
        print('netD has {:,} trainable parameters'.format(params))
    
    params = sum([p.nelement() for p in netG.parameters()])
    if config.local_rank == 0:
        print('netG has {:,} trainable parameters'.format(params))

    if config.EMA_G and config.local_rank == 0:
        netG_avg = get_gmodel(**netG_params)
        netG_avg.load_state_dict(netG.state_dict())
        for p in netG_avg.parameters():
            p.requires_grad = False
    else:
        netG_avg = None
    

    g_params = []
    mapping_net_params = []
    for n, p in netG.named_parameters():
        if n.startswith('mapping_net'):
            mapping_net_params.append(p)
        else:
            g_params.append(p)


    optim.Adam = apex.optimizers.FusedAdam if config.fp16 else optim.Adam
    optimizerD = optim.Adam(netD.parameters(),
                            lr=config.d_lr, betas=(0., 0.999))
    optimizerG = optim.Adam(netG.parameters(),
                            lr=config.g_lr, betas=(0., 0.999))
    #optimizerG.add_param_group({'params': mapping_net_params,
    #                            'lr': config.g_lr * 0.5,
    #                            'betas': (0., 0.999)})


    init_iter = 0
    if config.resume:
        if 'netD' in config.resume.name:
            dpath = config.resume
            gpath = config.resume.parent / config.resume.name.replace('netD', 'netG')
        elif 'netG' in config.resume.name:
            gpath = config.resume
            dpath = config.resume.parent / config.resume.name.replace('netG', 'netD')
        else:
            print("Couldn't load checkpoints weights... exiting")
            exit(-1)
        netD, optimizerD, init_iter = load_ckpt(dpath, netD, device, optimizerD)
        netG, optimizerG, _ = load_ckpt(gpath, netG, device, optimizerG)
        set_current_LR(optimizerD, config.d_lr)
        set_current_LR(optimizerG, config.g_lr)
        if config.EMA_G and config.local_rank == 0:
            g_avg_path = gpath.parent / gpath.name.replace('netG', 'netG_avg')
            netG_avg, _, _ = load_ckpt(g_avg_path, netG_avg, device, None)
        print('loaded checkpoint from epoch {}...'.format(init_iter))

    netD, netG = netD.to(device), netG.to(device)
    if netG_avg is not None:
        netG_avg = netG_avg.to(device).eval()

    if config.fp16:
        [netD, netG], [optimizerD, optimizerG] = amp.initialize(
                [netD, netG], [optimizerD, optimizerG],
                opt_level=config.opt_level, loss_scale=1.0)

    trainer = Trainer(init_iter=init_iter,
                      config=config,
                      netD=netD,
                      netG=netG,
                      dataset=dset,
                      z_dim=config.z_dim,
                      loss=loss,
                      sampler_fn=sampler_fn,
                      optimizerD=optimizerD,
                      optimizerG=optimizerG,
                      logdir=logdir,
                      device=device,
                      netG_avg=netG_avg,
                      netG_params=netG_params)

    if config.local_rank == 0:
        store_training_setup(logdir=logdir,
                             config_dict=vars(config),
                             netD=netD,
                             netG=netG)
    trainer.run()




if __name__ == "__main__":
    args = read_args()
    main(args.local_rank, args)
