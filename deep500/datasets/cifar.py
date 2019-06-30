import tarfile
from typing import List, Tuple, Dict

import numpy as np

from deep500.utils.download import real_download, unzip
from deep500.lv2.dataset import NumpyDataset
from deep500.utils.onnx_interop.losses import SoftmaxCrossEntropy

def download_cifar10_and_get_file_paths(folder=''):
    """
    Download cifar10 from University of Toronto
    The archive contains the files data_batch_1, data_batch_2, ..., data_batch_5, as well as test_batch
    :return: paths to different files
    """
    base_url = "https://www.cs.toronto.edu/~kriz/"
    filenames = [('cifar10', 'cifar-10-binary.tar.gz')]
    sub_folder = '/cifar10'

    local_files = real_download(base_url, filenames, sub_folder, output_dir=folder)
    files = unzip(local_files['cifar10'])

    data_files = [file for file in files if '_batch_' in file]
    test_data = [file for file in files if 'test_batch' in file]

    return data_files, test_data

def download_cifar100_and_get_file_paths(folder=''):
    """
    Download cifar10 from University of Toronto
    The archive contains the files data_batch_1, data_batch_2, ..., data_batch_5, as well as test_batch
    :return: paths to different files
    """
    base_url = "https://www.cs.toronto.edu/~kriz/"
    filenames = [('cifar100', 'cifar-100-binary.tar.gz')]
    sub_folder = '/cifar100'

    local_files = real_download(base_url, filenames, sub_folder, output_dir=folder)
    files = unzip(local_files['cifar100'])

    data_files = [file for file in files if 'train.bin' in file]
    test_data = [file for file in files if 'test.bin' in file]

    return data_files, test_data

def _cifar_shape():
    return (3, 32, 32)
def cifar10_shape():
    return (10, *_cifar_shape())
def cifar100_shape():
    return (100, *_cifar_shape())

def cifar10_loss():
    return SoftmaxCrossEntropy
def cifar100_loss():
    return cifar10_loss()

cifar_mean = {
    'cifar10': (0.4914, 0.4822, 0.4465),
    'cifar100': (0.5071, 0.4867, 0.4408),
}

cifar_std = {
    'cifar10': (0.2023, 0.1994, 0.2010),
    'cifar100': (0.2675, 0.2565, 0.2761),
}

# Fast learning rate schedule for CIFAR-10, obtained from 
# https://github.com/meliketoy/wide-resnet.pytorch
def cifar_learning_rate(epoch, init=0.1):
    optim_factor = 0
    if(epoch > 160):
        optim_factor = 3
    elif(epoch > 120):
        optim_factor = 2
    elif(epoch > 60):
        optim_factor = 1

    return init*math.pow(0.2, optim_factor)

def _cifar_numpy(train_files, test_files, dsname, normalize):
    # Images per batch
    ipb = 50000 if dsname == 'cifar100' else 10000
    test_ipb = 10000
    
    imgsize = 32 * 32 * 3
    entrylen = (imgsize+1) if dsname == 'cifar10' else (imgsize+2)
    entryoff = 0 if dsname == 'cifar10' else 1
    size = ipb * entrylen
    test_size = test_ipb * entrylen
    
    # Create arrays for train and test images
    train_images = np.zeros([len(train_files)*ipb, 3, 32, 32], dtype=np.float32)
    test_images = np.zeros([test_ipb, 3, 32, 32], dtype=np.float32)
    train_labels = np.zeros(len(train_files)*ipb, dtype=np.int64)
    test_labels = np.zeros(test_ipb, dtype=np.int64)

    # Extract training data (label followed by image data)
    for i, file in enumerate(train_files):
        with open(file, 'rb') as f:
            filebuffer = np.frombuffer(f.read(), 'B')
        
        # Read labels and images
        # Adapted from https://mattpetersen.github.io/load-cifar10-with-numpy
        train_labels[i*ipb:(i+1)*ipb] = filebuffer[entryoff::entrylen].astype(np.int64)
        pixeldata = np.delete(filebuffer, np.arange(entryoff, size, entrylen))
        if dsname == 'cifar100':
            pixeldata = np.delete(pixeldata, np.arange(0, size, entrylen-1))
        train_images[i*ipb:(i+1)*ipb] = pixeldata.reshape(-1,3,32,32).astype(np.float32) / 255.0

    # Extract test data        
    with open(test_files[0], 'rb') as f:
        filebuffer = np.frombuffer(f.read(), 'B')
    test_labels[:] = filebuffer[entryoff::entrylen].astype(np.int64)
    pixeldata = np.delete(filebuffer, np.arange(entryoff, test_size, entrylen))
    if dsname == 'cifar100':
        pixeldata = np.delete(pixeldata, np.arange(0, test_size, entrylen-1))
    test_images[:] = pixeldata.reshape(-1,3,32,32).astype(np.float32) / 255.0

    # Normalize if necessary
    if normalize:
        for i in range(3):
            train_images[:,i,:,:] -= cifar_mean[dsname][i]
            test_images[:,i,:,:] -= cifar_mean[dsname][i]
            train_images[:,i,:,:] /= cifar_std[dsname][i]
            test_images[:,i,:,:] /= cifar_std[dsname][i]

    return train_images, train_labels, test_images, test_labels

def _load_cifar(is_cifar100, input_node_name, label_node_name, normalize=True, folder=''):
    if is_cifar100:
        train_batch, test_batch = download_cifar100_and_get_file_paths(folder=folder)
        train_img, train_lbl, test_img, test_lbl = _cifar_numpy(
            train_batch, test_batch, 'cifar100', normalize)
    else:
        train_batch, test_batch = download_cifar10_and_get_file_paths(folder=folder)
        train_img, train_lbl, test_img, test_lbl = _cifar_numpy(
            train_batch, test_batch, 'cifar10', normalize)

    return (NumpyDataset(train_img, input_node_name, train_lbl, label_node_name),
            NumpyDataset(test_img, input_node_name, test_lbl, label_node_name))
    
    
def load_cifar10(input_node_name, label_node_name, *args, normalize=True,
                 folder='', **kwargs):
    return _load_cifar(False, input_node_name, label_node_name, normalize,
                       folder)


def load_cifar100(input_node_name, label_node_name, *args, normalize=True,
                  folder='', **kwargs):
    return _load_cifar(True, input_node_name, label_node_name, normalize,
                       folder)
