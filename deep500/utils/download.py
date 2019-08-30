import contextlib
import sys
import tarfile
import os
import glob
import tempfile
from tqdm import tqdm
from typing import Tuple, List, Dict
from urllib import request

BASE_URL_ONNX_ZOO = 'https://s3.amazonaws.com/download.onnx/models'


# Adapted from https://github.com/tqdm/tqdm/blob/master/examples/tqdm_wget.py
def my_hook(t):
    """Wraps tqdm instance.
    Don't forget to close() or __exit__()
    the tqdm instance once you're done with it (easiest using `with` syntax).
    Example
    -------
    >>> with tqdm(...) as t:
    ...     reporthook = my_hook(t)
    ...     urllib.urlretrieve(..., reporthook=reporthook)
    """
    last_b = [0]

    def update_to(b=1, bsize=1, tsize=None):
        """
        b  : int, optional
            Number of blocks transferred so far [default: 1].
        bsize  : int, optional
            Size of each block (in tqdm units) [default: 1].
        tsize  : int, optional
            Total size (in tqdm units). If [default: None] remains unchanged.
        """
        if tsize is not None:
            t.total = tsize
        t.update((b - last_b[0]) * bsize)
        last_b[0] = b

    return update_to


class TqdmUpTo(tqdm):
    """Alternative Class-based version of the above.
    Provides `update_to(n)` which uses `tqdm.update(delta_n)`.
    Inspired by [twine#242](https://github.com/pypa/twine/pull/242),
    [here](https://github.com/pypa/twine/commit/42e55e06).
    """

    def update_to(self, b=1, bsize=1, tsize=None):
        """
        b  : int, optional
            Number of blocks transferred so far [default: 1].
        bsize  : int, optional
            Size of each block (in tqdm units) [default: 1].
        tsize  : int, optional
            Total size (in tqdm units). If [default: None] remains unchanged.
        """
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)  # will also set self.n = b * bsize

def download_onnx_zoo(model):
    """
    Provides basic functionality to download onnx models from onnx model zoo
    """
    if not os.path.isfile('{}.tar.gz'.format(model)):
        print('downloading: {}'.format(model))
        url = '{}/{}.tar.gz'.format(BASE_URL_ONNX_ZOO, model)
        with open('{}.tar.gz'.format(model), 'wb') as out_file:
            with contextlib.closing(request.urlopen(url)) as fp:
                block_size = 2 ** 18
                counter = 1
                size = fp.length

                while True:
                    block = fp.read(block_size)
                    if not block:
                        break
                    out_file.write(block)
                    if counter % 10 == 0:
                        print('='.format(round(100. / size * block_size * counter, 1)), end='')
                        sys.stdout.flush()
                    counter += 1
    if not os.path.isdir(model):
        print('\nunzipping')
        tar = tarfile.open('{}.tar.gz'.format(model))
        tar.extractall('.')
        print('done!')


def is_dir_in_temp(directory: str) -> bool:
    """
    Checks if the given directory name is in the temp directory
    :param directory: name of the directory
    :return: true if directory is there
    """
    temp_directory = tempfile.gettempdir()
    dir_start_backslash = directory.startswith('/')
    temp_dir_ends_backslash = temp_directory.endswith('/')
    if dir_start_backslash and temp_dir_ends_backslash:
        return os.path.isdir(temp_directory[:-1] + directory)
    if not temp_dir_ends_backslash and not dir_start_backslash:
        return os.path.isdir(temp_directory + '/' + directory)
    return os.path.isdir(temp_directory + directory)


def is_file_in_dir(directory_path: str, file_name: str) -> bool:
    """
    Returns true if file exists
    :param directory_path: directory path
    :param file_name: file name
    :return: true if file exists else false
    """
    path = os.path.join(directory_path, file_name)
    return os.path.isfile(path)

def unzip(local_file):
    path = os.path.dirname(os.path.abspath(local_file))
    files = []
    print('\nunzipping in path: {}'.format(path))
    tar = tarfile.open(local_file)
    file = tar.next()  # type: tarfile.TarInfo
    while file is not None:
        if not is_file_in_dir(path, file.name):
            tar.extract(file, path)
        files.append(path + '/' + file.name)
        file = tar.next()
    print('done!')
    return files

def unrar(local_file):
    try:
        import rarfile
    except (ImportError, ModuleNotFoundError) as ex:
        raise ImportError('Cannot use unrar without rarfile: %s' % str(ex))

    path = os.path.dirname(os.path.abspath(local_file))
    print('\nunzipping in path: {}'.format(path))

    rar = rarfile.RarFile(local_file)
    dir = os.path.join(path, sorted(rar.namelist())[0])
    namelist = set([f.rstrip('/') for f in glob.glob("{}/**".format(dir), recursive=True)])
    rar_namelist = set([os.path.join(path, f) for f in rar.namelist()])

    if rar_namelist == namelist:
        dirs = set([f.rstrip('/') for f in glob.glob("{}/**/".format(dir), recursive=True)])
        files = list(namelist - dirs)
    else:
        files = []
        for filename in rar.namelist():
            if not is_file_in_dir(path, filename):
                rar.extract(filename, path)
            files.append(path + '/' + filename)
    print('done!')
    return files

def real_download(base_url, filenames, sub_folder, output_dir=''):
    if output_dir is None or len(output_dir) == 0:
        output_dir = tempfile.gettempdir()

    files_to_download = []
    local_files = {}
    temp_dir = output_dir + sub_folder
    dataset_exists = os.path.isdir(temp_dir)
    if dataset_exists:
        for (name, filename) in filenames:
            if not is_file_in_dir(temp_dir, filename):
                files_to_download.append((name, filename))
            else:
                local_files[name] = temp_dir + '/' + filename
    else:
        files_to_download = filenames
        if not os.path.isdir(temp_dir):
            os.mkdir(temp_dir)
    for (name, filename) in files_to_download:
        print("Downloading " + name + "...")
        path = temp_dir + '/' + filename
        print(path)
        with TqdmUpTo(unit='B', unit_scale=True, unit_divisor=1024, miniters=1) as t:  # all optional kwargs
            request.urlretrieve(base_url + filename, path, reporthook=t.update_to)
        local_files[name] = path
    print("Download complete.")
    return local_files


