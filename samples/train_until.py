# Example showing stopping conditions and runner
import sys
import deep500 as d5
from deep500 import networks as d5net
from deep500 import datasets as d5ds
from deep500.frameworks import tensorflow as d5tf, reference as d5ref

LABEL_NODE = 'labels'
BATCH_SIZE = 64
MAX_EPOCHS = 10

if __name__ == '__main__':
    if len(sys.argv) > 4:
        print('USAGE: train_until.py [NETWORK NAME] [DATASET NAME] [DESIRED ACCURACY]')
        print('Defaults: simple_cnn on MNIST, desired accuracy 98%')
        sys.exit(1)
    netname = 'simple_cnn' if len(sys.argv) < 2 else sys.argv[1]
    dsname = 'mnist' if len(sys.argv) < 3 else sys.argv[2]
    accuracy = 98.0 if len(sys.argv) < 4 else float(sys.argv[3])


    # Create CNN using ONNX
    ds_cls, ds_c, ds_h, ds_w = d5ds.dataset_shape(dsname)
    onnx_file = d5net.export_network(netname, BATCH_SIZE, classes=ds_cls,
                                     shape=(ds_c, ds_h, ds_w))
    model = d5.parser.load_and_parse_model(onnx_file)

    # Recover input and output nodes (assuming only one input and one output)
    INPUT_NODE = model.get_input_nodes()[0].name
    OUTPUT_NODE = model.get_output_nodes()[0].name
    
    # Create dataset and add loss function to model
    train_set, test_set = d5ds.load_dataset(dsname, INPUT_NODE, LABEL_NODE)
    model.add_operation(d5.ops.SoftmaxCrossEntropy([OUTPUT_NODE, LABEL_NODE], 'loss'))

    # Create executor and reference SGD optimizer
    executor = d5tf.from_model(model)
    optimizer = d5ref.GradientDescent(executor, 'loss')
    
    # Create samplers
    train_sampler = d5.ShuffleSampler(train_set, BATCH_SIZE)
    test_sampler = d5.ShuffleSampler(test_set, BATCH_SIZE)

    # Create runner (training/test manager)
    runner = d5.Trainer(train_sampler, test_sampler, executor, optimizer,
                        OUTPUT_NODE)
    #############################
    
    # Set up events to stop training after reaching the desired test accuracy
    events = d5.DefaultTrainerEvents(MAX_EPOCHS) + [
        d5.training_events.AccuracyAbortEvent(accuracy)]

    # Run training/test loop
    runner.run_loop(MAX_EPOCHS, events)

