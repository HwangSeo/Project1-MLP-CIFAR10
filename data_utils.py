from __future__ import print_function

from builtins import range
from six.moves import cPickle as pickle
import numpy as np
import os
import platform
import urllib.request
import tarfile
import pathlib

def load_pickle(f):
    version = platform.python_version_tuple()
    if version[0] == '2':
        return  pickle.load(f)
    elif version[0] == '3':
        return  pickle.load(f, encoding='latin1')
    raise ValueError("invalid python version: {}".format(version))

def load_CIFAR_batch(filename):
    """ load single batch of cifar """
    with open(filename, 'rb') as f:
        datadict = load_pickle(f)
        X = datadict['data']
        Y = datadict['labels']
        X = X.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("float")
        Y = np.array(Y)
        return X, Y

def load_CIFAR10(ROOT):
    """ load all of cifar """
    xs = []
    ys = []
    for b in range(1,3):
        f = os.path.join(ROOT, 'data_batch_%d' % (b, ))
        X, Y = load_CIFAR_batch(f)
        xs.append(X)
        ys.append(Y)
    Xtr = np.concatenate(xs)
    Ytr = np.concatenate(ys)
    del X, Y
    Xte, Yte = load_CIFAR_batch(os.path.join(ROOT, 'test_batch'))
    return Xtr, Ytr, Xte, Yte

def download_CIFAR10(download_root):
    url = "http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    download_root = os.path.expanduser(str(download_root))
    os.makedirs(download_root, exist_ok=True)

    filename = "cifar-10-python.tar.gz"
    extractname = "cifar-10-batches-py"

    file_path = os.path.join(download_root, filename)
    extract_path = os.path.join(download_root, extractname)

    if not os.path.exists(file_path):
        print(f"Downloading {url} to {file_path}...")
        urllib.request.urlretrieve(url, file_path)
        print("Download complete.")
    else:
        print(f"{file_path} already exists. Skipping download.")

    if not os.path.exists(extract_path):
        print(f"Extracting {file_path} to {download_root}...")
        with tarfile.open(file_path, 'r:gz') as tar:
            tar.extractall(path=download_root)
        print("Extraction complete.")
    else:
        print(f"{extract_path} already exists. Skipping Extract.")

download_CIFAR10("datasets")