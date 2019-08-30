import tarfile
from typing import List, Tuple, Dict
from urllib import request

import numpy as np
import gzip

from deep500.utils.download import real_download
from deep500.lv2.dataset import Dataset, NumpyDataset
from deep500.utils.onnx_interop.losses import SoftmaxCrossEntropy


def download_mnist_and_get_file_paths(folder='') -> Dict[str, str]:
    """
    Downloads pytorch_networks from Yann Lecun's website
    :return: paths to the different files
    """
    base_url = "http://yann.lecun.com/exdb/mnist/"

    filenames = [
        ["train_images", "train-images-idx3-ubyte.gz"],
        ["test_images", "t10k-images-idx3-ubyte.gz"],
        ["train_labels", "train-labels-idx1-ubyte.gz"],
        ["test_labels", "t10k-labels-idx1-ubyte.gz"]
    ]

    sub_folder = '/mnist'

    local_files = real_download(base_url, filenames, sub_folder, output_dir=folder)
    return local_files

def download_fashion_mnist_and_get_file_paths(folder='') -> Dict[str, str]:
    """
    Downloads pytorch_networks from the Zalando Research AWS
    :return: paths to the different files
    """
    base_url = "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/"

    filenames = [
        ["train_images", "train-images-idx3-ubyte.gz"],
        ["test_images", "t10k-images-idx3-ubyte.gz"],
        ["train_labels", "train-labels-idx1-ubyte.gz"],
        ["test_labels", "t10k-labels-idx1-ubyte.gz"]
    ]

    sub_folder = '/fashion_mnist'

    local_files = real_download(base_url, filenames, sub_folder, output_dir=folder)
    return local_files

def mnist_shape():
    return (10, 1, 28, 28)

def fashion_mnist_shape():
    return mnist_shape()

def mnist_loss():
    return SoftmaxCrossEntropy

def fashion_mnist_loss():
    return mnist_loss()

def _load_mnist(downloaded_data, data_node_name, label_node_name, normalize=True) -> Tuple[Dataset, Dataset]:
    """ Returns the training and testing Dataset objects for an MNIST-like dataset.
        @param data_node_name The graph node name for the data inputs.
        @param label_node_name The graph node name for the ground-truth labels.
        @param normalize Normalizes the input images first.
        @return A 2-tuple with the training and test datasets.
    """

    def extract_img(file_path):
        with gzip.open(file_path, 'rb') as f:
            return np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 1, 28, 28)

    def extract_lbl(file_path):
        with gzip.open(file_path, 'rb') as f:
            return np.frombuffer(f.read(), np.uint8, offset=8)

    train_img = extract_img(downloaded_data['train_images'])
    train_lbl = extract_lbl(downloaded_data['train_labels'])
    test_img = extract_img(downloaded_data['test_images'])
    test_lbl = extract_lbl(downloaded_data['test_labels'])

    # normalize
    if normalize:
        train_img = ((train_img - np.min(train_img)) / (np.max(train_img) - np.min(train_img))).astype(np.float32)
        test_img = ((test_img - np.min(test_img)) / (np.max(test_img) - np.min(test_img))).astype(np.float32)

    # prepare
    train_lbl = train_lbl.astype(np.int64)
    test_lbl = test_lbl.astype(np.int64)

    return (NumpyDataset(train_img, data_node_name, train_lbl, label_node_name),
            NumpyDataset(test_img, data_node_name, test_lbl, label_node_name))

def load_mnist(data_node_name, label_node_name, *args, normalize=True, folder='', **kwargs) -> Tuple[Dataset, Dataset]:
    """ Returns the training and testing Dataset objects for MNIST.
        @param data_node_name The graph node name for the data inputs.
        @param label_node_name The graph node name for the ground-truth labels.
        @param normalize Normalizes the input images first.
        @return A 2-tuple with the training and test datasets.
    """

    downloaded_data = download_mnist_and_get_file_paths(folder=folder)
    return _load_mnist(downloaded_data, data_node_name, label_node_name, normalize=normalize)

def load_fashion_mnist(data_node_name, label_node_name, *args, normalize=True, folder='', **kwargs) -> Tuple[Dataset, Dataset]:
    """ Returns the training and testing Dataset objects for Fashion MNIST.
        @param data_node_name The graph node name for the data inputs.
        @param label_node_name The graph node name for the ground-truth labels.
        @param normalize Normalizes the input images first.
        @return A 2-tuple with the training and test datasets.
    """

    downloaded_data = download_fashion_mnist_and_get_file_paths(folder=folder)
    return _load_mnist(downloaded_data, data_node_name, label_node_name, normalize=normalize)

