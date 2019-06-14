'''Implements classes and methods related to sampling datasets in a distributed environment.'''

import numpy as np
from typing import Any, Callable
from deep500.lv2.sampler import Sampler
from deep500.lv2.dataset import Dataset
from deep500.lv3.communication import CommunicationNetwork


class DistributedSampler(Sampler):
    """ The DistributedSampler class manages a Sampler, based on the rank ID
    and total number of ranks. This sampler replicates the entire dataset
    across all ranks. """

    def __init__(self, sampler: Sampler, comm: CommunicationNetwork = None):
        self.sampler = sampler
        self.comm_size = comm.size if comm is not None else 1
        self.comm_rank = comm.rank if comm is not None else 0
        self.reset()

    def __len__(self):
        return len(self.sampler) // self.comm_size

    def __iter__(self):
        return self

    def __call__(self):
        return self.sampler()

    def __next__(self):
        return next(self.sampler)

    def reset(self):
        # Pass samples to sampler and reset
        self.sampler.reset()        

    # Forward properties to sampler
    @property
    def batch_size(self):
        return self.sampler.batch_size

    @property
    def dataset(self):
        return self.sampler.dataset

    @property
    def seed(self):
        return self.sampler.seed

    @property
    def drop_last_batch(self):
        return self.sampler.drop_last_batch

    @property
    def events(self):
        return self.sampler.events

    def add_input_transformation(self, transform: Callable[[Any], Any]):
        self.sampler.add_input_transformation(transform)

    def add_label_transformation(self, transform: Callable[[Any], Any]):
        self.sampler.add_label_transformation(transform)

class PartitionedDataset(Dataset):
    """ Helper class for PartitionedDistributedSampler. """
    def __init__(self, dataset: Dataset, length: int, offset: int):
        self.dataset = dataset
        self.len = length
        self.off = offset
    
    def __len__(self):
        return self.len

    @property
    def input_node(self):
        return self.dataset.input_node

    @property
    def label_node(self):
        return self.dataset.label_node

    def modify(self, ind):
        return (ind % self.len) + self.off

    
    def __getitem__(self, index):
        if isinstance(index, (np.ndarray, int)): # One element or ndarray
            index = self.modify(index)
        else: # Minibatch
            if isinstance(index, slice): # Slice
                index = slice(self.modify(index.start), 
                              self.modify(index.stop),
                              index.step)
            elif isinstance(index, (list, tuple)): # List of elements
                index = type(index)(self.modify(i) for i in index)

        # Use modified indices
        return self.dataset[index]

class PartitionedDistributedSampler(DistributedSampler):    
    ''' A distributed sampler that partitions the data among participating ranks. '''
    def __init__(self, sampler: Sampler, comm: CommunicationNetwork = None):
        super().__init__(sampler, comm)
        # Replace dataset with partitioned dataset
        dataset = self.sampler.dataset
        partlen = len(dataset) // self.comm_size
        partoff = partlen * self.comm_rank
        self.sampler.dataset = PartitionedDataset(dataset, partlen, partoff)

    def __len__(self):
        return len(self.sampler)

    def reset(self):
        self.sampler.reset()
