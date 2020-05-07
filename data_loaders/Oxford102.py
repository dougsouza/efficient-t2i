from pathlib import Path
import torch.utils.data as data
from PIL import Image
import numpy as np
import random
import pickle
import torch


def np_softmax(x, axis=0):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


class Oxford102Dataset(data.Dataset):
    def __init__(self,
                data_path=Path('datasets/Oxford102/flowers'),
                transform=None,
                target_transform=None,
                split='train',
                return_fnames=False,
                interp_sentences=False):
        super(Oxford102Dataset, self).__init__()
        self.data_path = Path(data_path)
        self.transform = transform
        self.target_transform = target_transform
        self.split = split
        self.return_fnames = return_fnames
        self.interp_sentences = interp_sentences
        self.__dataset_path = self.data_path / self.split

        
        with open(self.__dataset_path / 'attn_embeddings.pickle', 'rb') as f:
            self.embeddings = pickle.load(f, encoding='latin1')

        with open(self.__dataset_path / 'filenames.pickle', 'rb') as f:
            self.fnames = pickle.load(f, encoding='latin1')

        with open(self.__dataset_path / '304images.pickle', 'rb') as f:
            self.images = pickle.load(f, encoding='latin1')



    def __getitem__(self, ix):
        sent_emb = self.embeddings[ix]

        if self.split == 'train':
            rdix = random.randint(0, len(sent_emb))
            if self.interp_sentences:
                probs = np_softmax(np.random.uniform(0, len(sent_emb), (len(sent_emb), 1)))
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
            with open(self.data_path / 'text' /  \
                        '{}.txt'.format(self.fnames[ix]), 'r') as f:
                text_captions = f.readlines()
            output += [text_captions]
        if self.return_fnames:
            output += [self.fnames[ix], ix]

        return tuple(output)


    def __len__(self):
        return len(self.images)
