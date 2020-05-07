from pathlib import Path
from PIL import Image
import torch
from collections import namedtuple
try:
    import accimage
except ImportError:
    accimage = None


def isnan(x):
    return x != x

def np_softmax(x, axis=0):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=axis)

def get_current_LR(opt):
    for param_group in opt.param_groups:
        return param_group['lr']

def set_current_LR(opt, lr):
    for param_group in opt.param_groups:
        param_group['lr'] = lr


def load_ckpt(path, net, device, optim=None):
    ckpt = torch.load(path, map_location=lambda storage, loc: storage)
    net.load_state_dict(ckpt['net'])
    if optim is not None:
        optim.load_state_dict(ckpt['optimizer'])
    it = ckpt['it']
    if optim is not None:
        for state in optim.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    #state[k] = v.to(device)
                    state[k] = v.cuda()
    
    return net, optim, it


def save_ckpt(net, optim, it, fname):
    ckpt = {
            'net': net.state_dict(),
            'optimizer': optim.state_dict(),
            'it': it
            }
    torch.save(ckpt, fname)



_pil_interpolation_to_str = {
    Image.NEAREST: 'PIL.Image.NEAREST',
    Image.BILINEAR: 'PIL.Image.BILINEAR',
    Image.BICUBIC: 'PIL.Image.BICUBIC',
    Image.LANCZOS: 'PIL.Image.LANCZOS',
}


def _is_pil_image(img):
    if accimage is not None:
        return isinstance(img, (Image.Image, accimage.Image))
    else:
        return isinstance(img, Image.Image)


class Resize(object):
    """Resize the input PIL Image to the given size.
    Args:
        size (sequence or int): Desired output size. If size is a sequence like
            (h, w), output size will be matched to this. If size is an int,
            smaller edge of the image will be matched to this number.
            i.e, if height > width, then image will be rescaled to
            (size * height / width, size)
        interpolation (int, optional): Desired interpolation. Default is
            ``PIL.Image.BILINEAR``
    """

    def __init__(self, size, interpolation=Image.BILINEAR):
        assert isinstance(size, int) or (isinstance(size, Iterable) and len(size) == 2)
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be scaled.
        Returns:
            PIL Image: Rescaled image.
        """
        return resize(img, self.size, self.interpolation)

    def __repr__(self):
        interpolate_str = _pil_interpolation_to_str[self.interpolation]
        return self.__class__.__name__ + '(size={0}, interpolation={1})'.format(self.size, interpolate_str)


def resize(img, size, interpolation=Image.BILINEAR):
    r"""Resize the input PIL Image to the given size.
    Args:
        img (PIL Image): Image to be resized.
        size (sequence or int): Desired output size. If size is a sequence like
            (h, w), the output size will be matched to this. If size is an int,
            the smaller edge of the image will be matched to this number maintaing
            the aspect ratio. i.e, if height > width, then image will be rescaled to
            :math:`\left(\text{size} \times \frac{\text{height}}{\text{width}}, \text{size}\right)`
        interpolation (int, optional): Desired interpolation. Default is
            ``PIL.Image.BILINEAR``
    Returns:
        PIL Image: Resized image.
    """
    if not _is_pil_image(img):
        raise TypeError('img should be PIL Image. Got {}'.format(type(img)))
    if not (isinstance(size, int) or (isinstance(size, Iterable) and len(size) == 2)):
        raise TypeError('Got inappropriate size arg: {}'.format(size))

    if isinstance(size, int):
        w, h = img.size
        if (w <= h and w == size) or (h <= w and h == size):
            return img
        if w > h:
            ow = size
            oh = int(size * h / w)
            return img.resize((ow, oh), interpolation)
        else:
            oh = size
            ow = int(size * w / h)
            return img.resize((ow, oh), interpolation)
    else:
        return img.resize(size[::-1], interpolation)



if __name__ == "__main__":
    tf = Resize(256)
    dset = ImageFolder('/home/douglas/new_coco', transform=tf)
    x, y = dset[503]
    print(x.size)
    plt.imshow(x)
    plt.show()
