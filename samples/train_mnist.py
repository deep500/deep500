# Sample showing the use of models, executors, datasets, and optimizers over 
# MNIST with a simple CNN.
import numpy as np
from tqdm import trange

import deep500 as d5
from deep500 import datasets as d5ds
from deep500 import networks as d5net

import deep500.frameworks.tensorflow as d5tf
import deep500.frameworks.reference as d5ref


LABEL_NODE = 'labels'
BATCH_SIZE = 64

if __name__ == '__main__':
    # Create CNN using ONNX
    onnx_file = d5net.export_network('simple_cnn', BATCH_SIZE)
    model = d5.parser.load_and_parse_model(onnx_file)

    # Recover input and output nodes (assuming only one input and one output)
    INPUT_NODE = model.get_input_nodes()[0].name
    OUTPUT_NODE = model.get_output_nodes()[0].name
    
    # Create dataset and add loss function to model
    train_set, test_set = d5ds.load_mnist(INPUT_NODE, LABEL_NODE)
    model.add_operation(d5.ops.SoftmaxCrossEntropy([OUTPUT_NODE, LABEL_NODE], 'loss'))

    # Create executor and reference SGD optimizer
    tensorflow_executor = d5tf.from_model(model)
    optimizer = d5tf.GradientDescent(tensorflow_executor, 'loss')

    # Initial test set accuracy check
    n = len(test_set)
    wrong = 0
    for i in range(len(test_set) // BATCH_SIZE):
        inp = test_set[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
        out = tensorflow_executor.inference(inp)
        wrong += np.sum(inp[LABEL_NODE] != np.argmax(out[OUTPUT_NODE], axis=1))
    print("Accuracy before: {}".format((n-wrong)/n))
    
    # Train for one epoch (direct pass over Dataset without a Sampler)
    for i in trange(len(train_set) // BATCH_SIZE):
        inp = train_set[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
        optimizer.step(inp)
        
    # Check test set accuracy again
    wrong = 0
    for i in range(len(test_set) // BATCH_SIZE):
        inp = test_set[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
        out = tensorflow_executor.inference(inp)
        wrong += np.sum(inp[LABEL_NODE] != np.argmax(out[OUTPUT_NODE], axis=1))
    print("Accuracy after one epoch: {}".format((n-wrong)/n))
