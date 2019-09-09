import torch
from torch import nn
import torch.nn.init as init
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np

def conv_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.xavier_uniform(m.weight, gain=np.sqrt(2))
        init.constant(m.bias, 0)
    elif classname.find('BatchNorm') != -1:
        init.constant(m.weight, 1)
        init.constant(m.bias, 0)


class Net(nn.Module):
    def __init__(self, in_channels=1, sz=28, classes=10):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        
        # Compute size after convolutions and pooling
        self.imw = (((sz - 5 + 1) // 2) - 5 + 1) // 2

        self.fc1 = nn.Linear(self.imw*self.imw*20, 50)
        self.fc2 = nn.Linear(50, classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, self.imw*self.imw*20)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

        
def export_simple_cnn(batch_size: int, classes=10, shape=(1, 28, 28),
                      file_path='mnist_cnn.onnx') -> str:
    model = Net(shape[0], shape[1], classes)
    model.apply(conv_init)
    dummy_input = Variable(torch.randn(batch_size, *shape))
    torch.onnx.export(model, dummy_input, file_path, verbose=True)
    return file_path
