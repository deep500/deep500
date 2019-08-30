import argparse
import torch
import torch.nn as nn
from torchvision.models.resnet import model_urls, load_state_dict_from_url
import numpy as np

import deep500 as d5
import deep500.datasets as d5ds
import deep500.frameworks.reference as d5ref
from deep500.frameworks import pytorch as d5fw
from deep500.networks.pytorch_resnet import ResNet, Bottleneck

try:
    import av
except (ImportError, ModuleNotFoundError) as ex:
    raise ImportError('Cannot load ucf101 videos without av: %s' % str(ex))

class ResNetLSTM(ResNet):
    def __init__(self, num_classes=101, inplanes=64,
                 block=Bottleneck, residual_block=None, layers=[3, 4, 6, 3],
                 width=[64, 128, 256, 512], expansion=4, groups=[1, 1, 1, 1],
                 lstm_hidden_size=256):
        super(ResNetLSTM, self).__init__()
        self.inplanes = inplanes
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        for i in range(len(layers)):
            layer = self._make_layer(block=block, planes=width[i],
                                     blocks=layers[i], expansion=expansion,
                                     stride=2 if i == 0 else 2,
                                     residual_block=residual_block,
                                     groups=groups[i])
            setattr(self, 'layer%s' % str(i + 1), layer)

        self.avgpool = nn.AdaptiveAvgPool2d(1)

        self.lstm = nn.LSTM(
            input_size=width[-1]*expansion,
            hidden_size=lstm_hidden_size,
            num_layers=2, dropout=0.5)
        self.lstm_hidden_size = lstm_hidden_size
        self.fc = nn.Linear(lstm_hidden_size, num_classes)

    def forward(self, x):
        embed_seq = []
        for t in range(x.size(1)):
            embed_seq.append(self.features(x[:,t,:,:,:]))
        embed_seq = torch.stack(embed_seq, dim=0)

        out, _ = self.lstm(embed_seq)
        out = out.clone().transpose_(0, 1) #batch first
        out = self.fc(out.contiguous().view(-1, self.lstm_hidden_size))
        out = out.view(x.size(0), x.size(1), -1).mean(1)

        return out

def _resnetlstm(arch, pretrained, block, layers, **kwargs):
    model = ResNetLSTM(**kwargs)
    # use weights pretrained on the 1000-class ImageNet dataset
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch])
        for key in model.state_dict():
            if 'lstm' in key or 'fc' in key:
                state_dict[key] = model.state_dict()[key]
        model.load_state_dict(state_dict)
    return model

def ResNet50LSTM(num_classes=101, pretrained=True):
    return _resnetlstm('resnet50', pretrained=pretrained,
                       block=Bottleneck, layers=[3, 4, 6, 3], num_classes=num_classes)

def ResNet101LSTM(num_classes=101, pretrained=True):
    return _resnetlstm('resnet101',  pretrained=pretrained,
                       block=Bottleneck, layers=[3, 4, 23, 3], num_classes=num_classes)

EPOCHS = 100
BATCH_SIZE = 4

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=str, nargs="?", default=None)
    args = vars(parser.parse_args())

    ds_shape = d5ds.dataset_shape('ucf101')
    ds_classes, sample_shape = ds_shape[0], ds_shape[1:]

    train_set, validation_set = d5ds.load_ucf101('0', 'label', folder=args['folder'],
                                                 normalize=True, max_length=700, skip_frames=10)

    seq_lengths = [av.open(path).streams.video[0].frames for path in train_set.data]
    train_sampler = d5.BucketSampler(train_set, BATCH_SIZE, seq_lengths, max_length=500,
                                     transformations=[d5ref.Crop((224, 224)),])
    validation_sampler = d5.OrderedSampler(validation_set, BATCH_SIZE,
                                           transformations=[d5ref.Crop((224, 224)),])

    model = ResNet50LSTM(num_classes=ds_classes, pretrained=True)

    loss = torch.nn.CrossEntropyLoss()
    executor = d5fw.PyTorchNativeGraphExecutor(model, loss, device=d5.GPUDevice())
    optimizer = d5fw.GradientDescent(executor, 'loss')

    METRICS = [
        d5.TestAccuracy(),
        d5.WallclockTime(reruns=0, avg_over=1)
    ]

    results = d5.test_training(executor, train_sampler, validation_sampler,
                               optimizer, EPOCHS, BATCH_SIZE, 'output',
                               metrics=[m for m in METRICS])

