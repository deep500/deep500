import zipfile
import os
import PIL.Image
from typing import List, Tuple, Dict
import numpy as np

from deep500.utils.download import real_download, unrar
from deep500.lv2.dataset import FileListDataset
from deep500.utils.onnx_interop.losses import SoftmaxCrossEntropy

# Optionally import PyAV
try:
    import av
except (ImportError, ModuleNotFoundError) as ex:
    av = None
    
    
def ucf101_shape():
    return (101, None, 3, 240, 320)

def ucf101_loss():
    return SoftmaxCrossEntropy

def download_ucf101_and_get_file_paths(folder='', split='01'):
    """
    Download ucf101 from University of Central Florida
    The archive contains the videos of different action classes
    :return: paths to different files
    """
    base_url = "https://www.crcv.ucf.edu/data/UCF101/"
    filenames = [('ucf101', 'UCF101.rar'),
                 ('ucf101_split','UCF101TrainTestSplits-RecognitionTask.zip')]
    sub_folder = '/ucf101'

    local_files = real_download(base_url, filenames, sub_folder, output_dir=folder)
    files = unrar(local_files['ucf101'])

    zip = zipfile.ZipFile(local_files['ucf101_split'])
    path = os.path.dirname(os.path.abspath(local_files['ucf101']))+'/UCF-101/'

    train_files = []
    with zip.open('ucfTrainTestlist/trainlist{}.txt'.format(split)) as file_split:
        for line in file_split:
            file = path + bytes.decode(line.split()[0])
            if file in files:
                train_files.append(file)

    test_files = []
    with zip.open('ucfTrainTestlist/testlist{}.txt'.format(split)) as file_split:
        for line in file_split:
            file = path + bytes.decode(line.strip())
            if file in files:
                test_files.append(file)

    label_list = {}
    with zip.open('ucfTrainTestlist/classInd.txt') as labels:
        for line in labels:
            line = bytes.decode(line.strip())
            label = line.split()[1]
            idx = int(line.split()[0]) - 1
            label_list[label] = idx

    return train_files, test_files, label_list


ucf101_mean = (0.39607886, 0.37930175, 0.351559)
ucf101_std = (0.28261574, 0.27613039, 0.28061599)

class ucf101_loader():
    def __init__(self, normalize=True, max_length=1777, skip_frames=10):
        if av is None:
            raise ImportError('Cannot load ucf101 videos without PyAV. Please see '
                      'https://github.com/mikeboers/PyAV for installation instructions.')

        self.normalize = normalize
        self.max_length = max_length
        self.skip_frames = skip_frames

    def _video_loader(self, video_path):            
        container = av.open(video_path)
        container.streams.video[0].thread_type = 'AUTO'
        _data = [frame.to_ndarray(format='rgb24') for frame in container.decode(video=0)]
        if _data[0].shape != (240, 320, 3):
            _data = [np.array(PIL.Image.fromarray(img, 'RGB').resize((320,240))) for img in _data]
        _data = np.asarray(_data, dtype=np.float32)
        if self.normalize:
            for ch in range(3):
                _data[:,:,:,ch] -= ucf101_mean[ch]
                _data[:,:,:,ch] /= ucf101_std[ch]
        return _data

    def __call__(self, data_path):
        #load multiple videos
        if type(data_path) is np.ndarray:
            data = [self._video_loader(path) for path in data_path]
            max_frames = max([x.shape[0] for x in data])
            data = [np.pad(x, ((max_frames-x.shape[0],0), (0,0), (0,0), (0,0)), 'constant') for x in data]
            data = np.vstack([np.expand_dims(x, axis=0) for x in data])
            data = data[:, :self.max_length:self.skip_frames, :, :, :]

        #load one video
        elif type(data_path) is np.str_:
            data = self._video_loader(data_path)
            data = data[:self.max_length:self.skip_frames, :, :, :]

        data = data / 255.0
        data = np.moveaxis(data, -1, -3)

        return data


def load_ucf101(input_node_name, label_node_name, *args, folder='', split='01',
                normalize=True, max_length=1777, skip_frames=10, **kwargs):
    assert(split in ['01', '02', '03'])
    train_files, test_files, label_list = download_ucf101_and_get_file_paths(folder=folder, split=split)

    train_files = np.asarray(train_files)
    test_files = np.asarray(test_files)

    train_lbl = [label_list[file.split('/')[-2]] for file in train_files]
    test_lbl = [label_list[file.split('/')[-2]] for file in test_files]

    train_lbl = np.asarray(train_lbl).astype(np.int32)
    test_lbl = np.asarray(test_lbl).astype(np.int32)

    loader = ucf101_loader(normalize, max_length, skip_frames)

    return (FileListDataset(train_files, input_node_name, loader, train_lbl, label_node_name),
            FileListDataset(test_files, input_node_name, loader, test_lbl, label_node_name))
