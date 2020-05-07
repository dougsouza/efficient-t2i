from __future__ import print_function
import os
import requests
from tqdm import tqdm
from glob import glob
from utils.files import ensure_dir


def download_file(destination, url=None, google_drive_id=None):
    assert url is not None or google_drive_id is not None, "either url or google drive id must be specified"
    if google_drive_id is not None:
        __download_file_from_google_drive(google_drive_id, destination)
    else:
        response = requests.get(url, stream=True)
        __save_response_content(response, destination)
    return destination


def __download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"
    session = requests.Session()
    response = session.get(URL, params={"id": id}, stream=True)
    token = __get_confirm_token(response)
    if token:
        params = {"id": id, "confirm": token}
        response = session.get(URL, params=params, stream=True)
    __save_response_content(response, destination)
    return destination


def __get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value
    return None


def __save_response_content(response, destination, chunk_size=32 * 1024):
    total_size = int(response.headers.get('content-length', 0))
    with open(destination, "wb") as f:
        pbar = tqdm(total=total_size, unit='B', unit_scale=True,
                    desc=destination.split('/')[-1])
        for chunk in response.iter_content(chunk_size):
            if chunk: # filter out keep-alive new chunks
                pbar.update(len(chunk))
                f.write(chunk)
        pbar.close()



def setup_lsun(base_path, category, tag):
    tmp_dir = os.path.join(ROOT_PATH, '_lsun_tmp')
    if os.path.exists(os.path.join(base_path, 'lsun')):
        print("lsun found, skipping download...")
    else:
        ensure_dir(tmp_dir)
        lsun_path = os.path.join(base_path, "lsun")
        if not os.path.exists(lsun_path):
            os.mkdir(lsun_path)
        sets = ["val"]#, "train"]
        for s in sets:
            url = 'http://lsun.cs.princeton.edu/htbin/download.cgi?tag={tag}' \
                  '&category={category}&set={set_name}'.format(tag=tag, category=category, set_name=s)
            print("Downloading lsun {} {} set...".format(args.category, s))
            f_name = download_file(tmp_dir, '{}_{}_lmdb.zip'.format(category, s), url)
            unzip(f_name, tmp_dir)
            print("Decoding lsun {} {} set and creating tf record...".format(args.category, s))
            convert_lsun_to_tfrecord(os.path.join(tmp_dir, "{}_{}_lmdb".format(args.category, s)), lsun_path,
                                     args.category)
            print("Done!")
        # print("cleaning up...")
        # shutil.rmtree(tmp_dir)
        # print("Done!")


def setup_cifar(base_path):
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    print("Downloading cifar10...")
    cifar_path = os.path.join(base_path, "cifar10")
    if os.path.exists(cifar_path):
        print("cifar10 found, skipping donwload...")
    else:
        os.mkdir(cifar_path)
        fname = download_file(cifar_path, url.split("/")[-1], url)
        print("Extracting cifar10...")
        untar(fname, cifar_path)
        print("Cleaning up...")
        os.remove(fname)
        print("Done!")


def setup_svhn(base_path):
    url_train = "http://ufldl.stanford.edu/housenumbers/train_32x32.mat"
    url_test = "http://ufldl.stanford.edu/housenumbers/test_32x32.mat"
    print("Downloading Street View House Number (SVHN)...")
    svhn_path = os.path.join(base_path, "SVHN")
    if os.path.exists(svhn_path):
        print("SVHN found, skipping donwload...")
    else:
        os.mkdir(svhn_path)
        fname_train = download_file(svhn_path, url_train.split("/")[-1], url_train)
        fname_test = download_file(svhn_path, url_test.split("/")[-1], url_test)
        print("Done!")
