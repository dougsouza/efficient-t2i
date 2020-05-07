from pathlib import Path
from datetime import datetime
import yaml


THIS_PATH = Path(__file__).absolute().parent


def store_training_setup(logdir, config_dict, netD, netG):
    with open(logdir / 'config.yaml', 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False)
    with open(logdir / 'netD.txt', 'w') as f:
        f.write(repr(netD))
    with open(logdir / 'netG.txt', 'w') as f:
        f.write(repr(netG))


def unzip(fpath, destination):
    import zipfile
    with zipfile.ZipFile(fpath) as zf:
        zf.extractall(destination)


def untar(fpath, destination):
    import tarfile
    with tarfile.open(fpath, 'r:gz') as tar:
        tar.extractall(path=destination)


def setup_dirs(exp_name, create_folder=True):
    log_folder = ensure_dir(THIS_PATH / '..' / 'logs')
    timestamp = datetime.now().strftime('%m%d_%H%M')
    logdir = log_folder / '{}_{}'.format(exp_name, timestamp)
    if create_folder:
        logdir.mkdir()
    return logdir


def ensure_dir(path):
    if not path.exists():
        path.mkdir(parents=True)
    return path
