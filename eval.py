import argparse
import torch
from easydict import EasyDict as edict
from pathlib import Path
import yaml
import subprocess
from torch.utils.data import DataLoader
from data_loaders import get_loader
from models import get_gmodel
from ganlib.priors import get_sampler_fn
from utils.files import ensure_dir
import tqdm
from glob import glob
from torchvision import transforms
import numpy as np
import random
import h5py
from evaluation.fid import calculate_fid_given_paths


def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('metric', type=str,
                            help='metric name', choices=['is', 'fid'])
    parser.add_argument('--model_folder', type=Path,
                        help='folder contaning checkpoints')
    parser.add_argument('--epoch', type=str,
                        help='epoch to compute metric. "all" for all epochs')
    parser.add_argument('--h5_dump', type=Path, default=None,
                        help='epoch to compute metric. "all" for all epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='batch size (default 16)')
    parser.add_argument('--dataset', type=str, default=None,
                        help='dataset (for FID only)')
    return parser.parse_args()


def default_padding(word_embs):
    max_length = max([wb.shape[0] for wb in word_embs])
    targets = torch.zeros(len(word_embs), max_length,
                           word_embs[0].shape[1], dtype=torch.float32)

    for i, wb in enumerate(word_embs):
        if isinstance(wb, np.ndarray):
            wb = torch.tensor(wb)
        end = wb.shape[0]
        targets[i, :end, :] = wb
    return targets



def test_collate(batch):
    imgs = torch.cat([img[0].unsqueeze(0) for img in batch], dim=0)
    sent = [sent[1] for sent in batch]
    word_embs = [word_embs[2] for word_embs in batch]
    caps = [caps[3] for caps in batch]
    return imgs, sent, word_embs, caps


def collate_fn(batch):
    # imgs = torch.cat([img[0].unsqueeze(0) for img in batch], dim=0)
    sent = torch.cat([sent[1].unsqueeze(0) for sent in batch], dim=0)
    word_embs = default_padding([word_embs[2] for word_embs in batch])
    return sent, word_embs



def load_generator(model_path, config, device):
    checkpoint = torch.load(model_path,
                            map_location=lambda storage, loc: storage)
    netG = get_gmodel(**checkpoint['netG_params'])
    netG.load_state_dict(checkpoint['net'])
    sampler_fn = get_sampler_fn(config.prior, device=device,
                                normalize=config.normalize_prior)
    return netG.to(device).eval(), sampler_fn, checkpoint['it']


def warmup_generator(netG, sampler_fn, z_dim, train_loader, device, n_forwards=100):
    train_iter = iter(train_loader)
    netG = netG.train()
    for p in netG.parameters():
        p.requires_grad = False
    print('warming up G')
    for _ in tqdm.trange(n_forwards):
        sent_emb, word_emb = next(train_iter)
        sent_emb, word_emb = sent_emb.to(device), word_emb.to(device)
        # word_emb = torch.cat([word_emb, sent_emb.unsqueeze(1).repeat(1, word_emb.shape[1], 1)], dim=2)
        z = sampler_fn(sent_emb.size(0), z_dim)
        _ = netG(z, sent_emb, word_emb)
    del train_iter
    return netG.eval()


def dump_test_data(data_loader,
                   test_sample_num,
                   fname,
                   netG,
                   imsize,
                   sampler_fn,
                   z_dim,
                   device):
    total_imgs = len(data_loader.dataset) * test_sample_num
    print('total images to save is {}'.format(total_imgs))
    start_idx = 0

    
    with h5py.File(fname, 'w') as h5file:
        samples_dset = h5file.create_dataset('samples', shape=(total_imgs,
                                    imsize, imsize, 3), dtype=np.uint8)
        fidx_dset = h5file.create_dataset('fidx', shape=(total_imgs,),
                                            dtype=np.int32)
        print('exporting imgs...')
        for _, batch_embs, batch_word_embs, batch_caps in tqdm.tqdm(data_loader):
            for embs, word_embs, caps in zip(batch_embs, batch_word_embs, batch_caps):
                emb = torch.tensor(embs, dtype=torch.float32)
                batch_size = emb.size(0)
                for i in range(len(word_embs)):
                    we = word_embs[i]
                    if we.ndim == 1:
                        we = we[None, ...]
                    if we.shape[1] > data_loader.dataset.WORDS_NUM:
                        word_embs[i] = we[:data_loader.dataset.WORDS_NUM, ...]
                word_emb = default_padding(word_embs)
                #word_emb = torch.cat([word_emb, emb.unsqueeze(1).repeat(1, word_emb.shape[1], 1)], dim=2)
                caps = torch.cat([torch.tensor(c, dtype=torch.int64) for c in caps], dim=0)
                mask = caps == 0
                num_words = word_emb.size(1)
                if mask.size(1) > num_words:
                    mask = mask[:, :num_words]

                z = sampler_fn(batch_size, z_dim)
                #print('---', z.shape, word_embs[0].shape)#, embs.shape, word_embs.shape)
                with torch.no_grad():
                    imgs = netG(z, emb.to(device),
                                word_emb.to(device), None)
                imgs_np = imgs.mul(0.5).add(0.5).clamp(0, 1).mul(255) \
                    .clamp_(0, 255).byte().permute(0, 2, 3, 1).cpu().numpy()
                samples_dset[start_idx:(start_idx + batch_size)] = imgs_np
                # fidx_dset[start_idx:(start_idx + batch_size)] = fidx
                start_idx += batch_size
            # for _ in range(test_sample_num):
            #     ridx = random.randint(0, len(emb)-1)
            #     batch_size = emb.size(0)
            #     # ridx = torch.tensor(np.random.choice(10, size=batch_size))
            #     # print('rdx', ridx.size())
            #     z = sampler_fn(batch_size, z_dim)
            #     with torch.no_grad():
            #         input_emb, wembs = emb.to(device), word_embs.to(device)
            #         imgs = netG(z, input_emb, wembs)
            #     imgs_np = imgs.mul(0.5).add(0.5).clamp(0, 1).mul(255) \
            #         .clamp_(0, 255).byte().permute(0, 2, 3, 1).cpu().numpy()
            #     samples_dset[start_idx:(start_idx + batch_size)] = imgs_np
            #     # fidx_dset[start_idx:(start_idx + batch_size)] = fidx
            #     start_idx += batch_size
        print('done')


def dump_fid_test_data(data_loader,
                   test_sample_num,
                   fname,
                   netG,
                   imsize,
                   sampler_fn,
                   z_dim,
                   device):
    total_imgs = len(data_loader.dataset) * test_sample_num
    print('total images to save is {}'.format(total_imgs))
    start_idx = 0
    
    with h5py.File(fname, 'w') as h5file:
        samples_dset = h5file.create_dataset('samples', shape=(total_imgs,
                                    imsize, imsize, 3), dtype=np.uint8)
        fidx_dset = h5file.create_dataset('fidx', shape=(total_imgs,),
                                            dtype=np.int32)
        print('exporting imgs...')
        for sent_emb, word_emb in tqdm.tqdm(data_loader):
            batch_size = sent_emb.size(0)
            z = sampler_fn(batch_size, z_dim)
            with torch.no_grad():
                input_emb, wembs = sent_emb.to(device), word_emb.to(device)
                imgs = netG(z, input_emb, wembs)
            imgs_np = imgs.mul(0.5).add(0.5).clamp(0, 1).mul(255) \
                .clamp_(0, 255).byte().permute(0, 2, 3, 1).cpu().numpy()
            samples_dset[start_idx:(start_idx + batch_size)] = imgs_np
            start_idx += batch_size
    print('done')
    return fname


def main(args):
    print(args)
    results_dir = ensure_dir(args.model_folder / 'results')
    with open(args.model_folder / 'config.yaml', 'r') as f:
        params = yaml.load(f)
        config = edict(params)
    
    device = torch.device('cuda:1')

    train_dset = get_loader(config.dataset, split='train', transform=None,
            target_transform=lambda x: torch.tensor(x, dtype=torch.float32),
            interp_sentences=False, return_embedding_ix=0)

    train_loader = DataLoader(train_dset, batch_size=config.batch_size,
                        collate_fn=collate_fn, shuffle=False, num_workers=4)
    test_dset = get_loader(config.dataset, split='test',
                            transform=transforms.ToTensor(),
                            target_transform=None)
    test_loader = DataLoader(test_dset,
                batch_size=args.batch_size, shuffle=False,
                collate_fn=test_collate, num_workers=4)


    if args.epoch == 'all':
        ckpt_paths = args.model_folder.glob('netG_avg_*')
        ckpt_paths = sorted(ckpt_paths,
                key=lambda x: int(x.name.split('_')[-1].split('.')[0]),
                reverse=True)
    else:
        ckpt_paths = [args.model_folder / \
                                'netG_avg_epoch_{}.pth'.format(args.epoch)]
    if args.h5_dump is not None:
        assert args.epoch != 'all', \
                    'combination not allowed: "epoch: all" and "h5_dump"'
    else:
        # data needs to be dumped first....
        for ckpt_path in ckpt_paths:
            netG, sampler_fn, epoch = load_generator(ckpt_path, config, device)
            netG = warmup_generator(netG, sampler_fn, config.z_dim,
                                                    train_loader, device)

            if args.metric == 'is':
                # compute IS
                dump_dir = results_dir / 'dump_images256.h5'
                dump_test_data(test_loader, 10, dump_dir, netG,
                            256, sampler_fn, config.z_dim,
                            device)
                print('computing IS for model {}'.format(ckpt_path.name))
                process = subprocess.Popen(['python',
                    'evaluation/inception_score/inception_score.py',
                    '--checkpoint_dir',
                    'evaluation/inception_score/inception_finetuned_models/birds_valid299/model.ckpt',
                    '--h5_file', dump_dir,
                    '--gmodel', str(epoch)])
                process.wait()
            if args.metric == 'fid':
                dump_dir = results_dir / 'dump_fid_images256.h5'
                fname = dump_fid_test_data(train_loader, 1, dump_dir, netG,
                           256, sampler_fn, config.z_dim,
                           device)
                fid_value = calculate_fid_given_paths([Path('evaluation/fid/{}_stats.npz'.format(config.dataset)), fname])
                print('Fid for epoch {} is {}'.format(epoch, fid_value))

    




if __name__ == "__main__":
    main(read_args())
