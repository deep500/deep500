import os
import sys
import numpy as np

import deep500 as d5
import deep500.frameworks.tensorflow as d5tf
import deep500.frameworks.reference as d5ref
import deep500.datasets as d5ds
import deep500.networks as d5net

LABEL_NODE = 'labels'
LOSS_NODE = 'loss'
BATCH_SIZE = 64
LR = 0.1
MOMENTUM = 0.9
MAX_EPOCHS = 10
PARTITION_DATASET = True

if __name__ == '__main__':
    if len(sys.argv) < 2 or len(sys.argv) > 4:
        print('USAGE: distributed_training.py <DIST. OPTIMIZER> [NETWORK] [DATASET]')
        print('Distributed optimizer options: local, dsgd, pssgd, dpsgd, asgd, mavg, hvd')
        sys.exit(1)
    distoptname = sys.argv[1]
    netname = 'simple_cnn' if len(sys.argv) < 3 else sys.argv[2]
    dsname = 'mnist' if len(sys.argv) < 4 else sys.argv[3]
    
    # Initialize communication
    if distoptname != 'local':
        comm = d5.CommunicationNetwork()
        print('Detected %d ranks, I am rank %d' % (comm.size, comm.rank))
    else:
        comm = None
 
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
    model.add_operation(d5.ops.SoftmaxCrossEntropy([OUTPUT_NODE, LABEL_NODE], LOSS_NODE))


    executor = d5tf.from_model(model)

    # Create optimizer and distributed optimizer
    optimizer = d5ref.MomentumOptimizer(executor, LOSS_NODE, LR, MOMENTUM)
    if distoptname == 'local':
        pass
    elif distoptname == 'dsgd':
        optimizer = d5ref.ConsistentDecentralized(optimizer, comm)
    elif distoptname == 'pssgd':    
        optimizer = d5ref.ConsistentParameterServer(optimizer, comm)
    elif distoptname == 'dpsgd':    
        optimizer = d5ref.ConsistentNeighbors(optimizer, comm)
    elif distoptname == 'asgd':    
        optimizer = d5ref.InconsistentParameterServer(optimizer, comm)
    elif distoptname == 'mavg':
        optimizer = d5ref.ModelAverageDecentralized(optimizer, comm)
    elif distoptname == 'hvd':    
        optimizer = d5tf.HorovodDistributedOptimizer(
            d5tf.MomentumOptimizer(executor, LOSS_NODE, LR, MOMENTUM), comm)
    else:
        raise NameError('Unrecognized distributed optimizer "%s"' % distoptname)

    # Create distributed samplers
    if PARTITION_DATASET:
        train_sampler = d5.PartitionedDistributedSampler(
            d5.ShuffleSampler(train_set, BATCH_SIZE), comm)
    else:
        train_sampler = d5.DistributedSampler(
            d5.ShuffleSampler(train_set, BATCH_SIZE), comm)

    if comm is None or comm.rank == 0:
        # No need to distribute test_set
        test_sampler = d5.ShuffleSampler(test_set, BATCH_SIZE)
    else:
        # No need to test if not rank 0
        test_sampler = None

    #############################

    # Events: Only print progress on rank 0
    events = d5.DefaultTrainerEvents(MAX_EPOCHS) if comm is None or comm.rank == 0 else []

    # Metrics: Add communication volume
    metrics = d5.DefaultTrainingMetrics() + [d5.CommunicationVolume()]

    # Run distributed training
    d5.test_training(executor, train_sampler, test_sampler, optimizer, 
                     MAX_EPOCHS, BATCH_SIZE, OUTPUT_NODE, events=events, 
                     metrics=metrics)
    
    # Wait for everyone to finish and finalize MPI if necessary
    d5.mpi_end_barrier()
