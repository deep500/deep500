from typing import Any, Callable, Dict, List, Optional

import numpy as np


class Dataset(object):
    """ Encapsulates a dataset of encoded inputs. The purpose of this interface is to 
        provide instructions on how to read (or synthesize) the data, decode the inputs 
        (e.g., JPEG images), and return them to the requester 
        (optimizer, for example).
    """  
    def __init__(self):
        self.input_node = None
        self.label_node = None

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


class NumpyDataset(Dataset):
    """ This class includes facilities to read a numpy ndarray-based
        labeled dataset.
    """

    def __init__(self, data: Any, data_node: str,
                 labels: Optional[Any] = None,
                 label_node: Optional[str] = None):
        super().__init__()
        self.input_node = data_node
        self.index = 0
        
        self.data = data

        # If this dataset is labeled
        if labels is not None:
            self.label_node = label_node
            self.labels = labels
            assert (len(self.data) == len(self.labels))
        else:
            self.label_node = None
            self.labels = None

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


class FileListDataset(Dataset):
    """ This class includes facilities to read the data from
        an array of filenames.
    """

    def __init__(self, data: Any, data_node: str, loader: Callable,
                 labels: Optional[Any] = None,
                 label_node: Optional[str] = None):
        super().__init__()
        self.input_node = data_node
        self.index = 0

        self.data = data
        self.loader = loader

        # If this dataset is labeled
        if labels is not None:
            self.label_node = label_node
            self.labels = labels
            assert (len(self.data) == len(self.labels))
        else:
            self.label_node = None
            self.labels = None

    def __getitem__(self, index) -> Dict[str, np.ndarray]:
        """ Returns a dictionary of the sample's graph node names to data.
            This includes the input and optionally the label.
            @param index The sample index to return.
            @return A mapping from node name to value of the sample index.
        """
        result = {self.input_node: self.loader(self.data[index])}
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
