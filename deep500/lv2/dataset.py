from typing import Any, Callable, Dict, List, Optional

import numpy as np

class Dataset(object):
    """ Encapsulates a dataset of encoded inputs. The purpose of this interface is to 
        provide instructions on how to read (or synthesize) the data, decode the inputs 
        (e.g., JPEG images), and return them to the requester 
        (optimizer, for example).
    """  
    def __init__(self):
        pass
        
    def add_input_transformation(self, transform: Callable[[np.ndarray], np.ndarray]):
        """
        Apply a transformation (e.g., decoding, data augmentation) on each input before
        it is sampled.
        @param transform The transformation to apply (function with one input and one output)
        """
        raise NotImplementedError

    def add_label_transformation(self, transform: Callable[[np.ndarray], np.ndarray]):
        """
        Apply a transformation (e.g., decoding, data augmentation) on each label before
        it is sampled.
        @param transform The transformation to apply (function with one input and one output)
        """
        raise NotImplementedError

    def __iter__(self):
        return self
        
    def __getitem__(self, index) -> Dict[str, np.ndarray]:
        """ Returns a dictionary of the sample's graph node names to data.
            This includes the input and optionally the label.
            @param index The sample index to return.
            @return A mapping from node name to value of the sample index.
        """
        # To be implemented by inheriting classes
        raise NotImplementedError
        
    def __len__(self):
        raise NotImplementedError


class Input(object):
    def __init__(self, node_name: str, data: Any):
        self.node_name = node_name
        self.data = data

        self.transformation = []

    def add_transformation(self, transformation):
        self.transformation.append(transformation)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        data = self.data[item]
        for transform in self.transformation:
            data = transform(data)
        return data
        
        
class NumpyDataset(Dataset):
    """ This class includes facilities to read a numpy ndarray-based labeled dataset. """  
    def __init__(self, data: Input, labels: Optional[Input]):
        super().__init__()
        self.input_node = data.node_name
        self.index = 0
        
        self.data = data

        # If this dataset is labeled
        if labels is not None:
            self.label_node = labels.node_name
            self.labels = labels
            assert (len(self.data) == len(self.labels))
        else:
            self.label_node = None
            self.labels = None

    def add_input_transformation(self, transform: Callable[[np.ndarray], np.ndarray]):
        """
        Apply a transformation (e.g., decoding, data augmentation) on each input before
        it is sampled.
        @param transform The transformation to apply (function with one input and one output)
        """
        self.data.add_transformation(transform)

    def add_label_transformation(self, transform: Callable[[np.ndarray], np.ndarray]):
        """
        Apply a transformation (e.g., decoding, data augmentation) on each label before
        it is sampled.
        @param transform The transformation to apply (function with one input and one output)
        """
        if self.labels is not None:
            self.labels.add_transformation(transform)
        
    def __getitem__(self, index) -> Dict[str, np.ndarray]:
        """ Returns a dictionary of the sample's graph node names to data.
            This includes the input and optionally the label.
            @param index The sample index to return.
            @return A mapping from node name to value of the sample index.
        """
        result = {self.input_node: self.data[index]}
        if self.labels is not None:
            result[self.label_node] = self.labels[index]
        return result
        
    def __next__(self) -> Dict[str, np.ndarray]:
        """ Returns the next sample as a dictionary of graph node names to 
            data. This includes the input and optionally the label.
            Loops forever and accesses data sequentially.
            @return A mapping from node name to value of the sample.
        """       
        # Simple repeating iteration
        index = self.index
        self.index += 1
        if self.index >= len(self.data):
            self.index = 0
            
        return self.__getitem__(index)

    def __len__(self):
        return len(self.data)
