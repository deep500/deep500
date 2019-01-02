""" Trains a perceptron to predict a function from a synthetic dataset. """
import random
import numpy as np

import deep500 as d5
from deep500.frameworks import reference as d5ref, tensorflow as d5tf

EPOCHS = 5
BATCH_SIZE = 1
TESTSET_SIZE = 100

##################################################
# The neural network to use 
import torch
from torch import nn
from torch.autograd import Variable

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(1, 1),
        )

    def forward(self, x):
        return self.network(x)

def simple_regression_net(batch_size):
    model = Net()
    file_path = "simple_regression.onnx"
    dummy_input = Variable(torch.randn(batch_size, 1))
    torch.onnx.export(model, dummy_input, file_path, verbose=True)
    return file_path

##################################################    
# The function to learn (we're using a function that cannot be represented by the network
def train_fn(x):
    return x ** 2
     
# A synthetic dataset that draws random numbers and computes their square
class SynthDataset(d5.Dataset):
    def __init__(self, input_node, label_node):
        super().__init__()
        self.input_node = input_node
        self.label_node = label_node
        
    def __getitem__(self, index):
        if isinstance(index, int): # One element
            num = np.array([random.random()], np.float32)
        else: # Minibatch
            if isinstance(index, slice): # Slice
                length = len(range(*index.indices(len(self))))
            else: # List of elements
                length = len(index)
            num = np.random.rand(length, 1).astype(np.float32)
        return {self.input_node: num, self.label_node: train_fn(num)}

    def __len__(self):
        return 1000 # Bogus number to determine epoch length

##################################################
        
def test_loss(executor, input_node, output_node):
    avg_testloss = 0
    for i in range(TESTSET_SIZE):
        input = np.random.rand(BATCH_SIZE, 1).astype(np.float32)
        output = executor.inference({input_node: input, 'ground_truth': train_fn(input)})
        avg_testloss += output['loss']
    print('Test loss: %f' % (avg_testloss / TESTSET_SIZE))
                
        
if __name__ == '__main__':
    # Create network and executor
    #############################
    # Use pytorch to create a 2-layer MLP for regression
    onnxfile = simple_regression_net(BATCH_SIZE)
    model = d5.parser.load_and_parse_model(onnxfile)
    # Add squared loss
    input_node = model.get_input_nodes()[0].name
    output_node = model.get_output_nodes()[0].name
    model.add_operation(d5.ops.MeanSquaredError([output_node, 'ground_truth'], 'loss'))
    executor = d5tf.from_model(model)
    
    # Create dataset and sampler
    ############################
    ds = SynthDataset(input_node, 'ground_truth')
    sampler = d5.ShuffleSampler(ds, batch_size=BATCH_SIZE)
    
    # Create optimizer
    optimizer = d5ref.GradientDescent(executor)
    
    # Initial test
    test_loss(executor, input_node, output_node)
    
    from tqdm import trange
    # Train for one epoch at a time
    for i in range(EPOCHS):
        print('Epoch %d/%d' % (i+1, EPOCHS))
        optimizer.train(len(ds), sampler)
        sampler.reset()
        test_loss(executor, input_node, output_node)
        