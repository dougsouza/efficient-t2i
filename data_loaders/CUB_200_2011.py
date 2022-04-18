import json
from pathlib import Path
import random

import h5py
import numpy as np
from PIL import Image
import torch
import torch.utils.data as data


def np_softmax(x, axis=0):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


class CUB200Dataset(data.Dataset):
    def __init__(
        self,
        data_path=Path('datasets/CUB'),
        transform=None,
        target_transform=None,
        split='train',
        return_captions=False,
        return_fnames=False,
        interp_sentences=False,
        return_embedding_ix=None,
    ):
        super(CUB200Dataset, self).__init__()
        self.data_path = Path(data_path)
        self.transform = transform
        self.target_transform = target_transform
        self.split = split
        self.return_captions = return_captions
        self.return_fnames = return_fnames
        self.interp_sentences = interp_sentences
        self.return_embedding_ix = return_embedding_ix
        self.__dataset_path = self.data_path / self.split
        
        with open(self.__dataset_path / 'en_data.json') as f:
            self.text_data = json.load(f)

        self.embeddings = np.load(self.__dataset_path / 'attn_embeddings.npy')

        self.images = h5py.File(self.__dataset_path / '304images.h5', 'r')['images']


    def __getitem__(self, ix):
        sent_emb = self.embeddings[ix]

        if self.split == 'train':
            rdix = random.randint(0, 9)
            if self.return_embedding_ix is not None:
                rdix = self.return_embedding_ix
            if self.interp_sentences:
                probs = np_softmax(np.random.uniform(0, 10, (len(sent_emb), 1)))
                sent_emb = np.sum(sent_emb * probs, axis=0)
            else:
                sent_emb = sent_emb[rdix]
        else: # return all sentence embeddings
            sent_emb = sent_emb[None, ...] # unszqueeze
        

        if self.target_transform is not None:
            sent_emb = self.target_transform(sent_emb)

        img = Image.fromarray(self.images[ix]).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)

        output = [img, sent_emb]
        if self.return_captions:
            output.append(self.text_data[ix]['captions'])
        if self.return_fnames:
            output += [self.text_data[ix]['filename'], ix]

        return tuple(output)

    def __len__(self):
        return len(self.images)
