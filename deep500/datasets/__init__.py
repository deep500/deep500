from .mnist import *
from .cifar import *
from .ucf101 import *
from .imagenet import *


def load_dataset(name: str, input_node: str, label_node: str, *args, **kwargs):
    name = name.strip().lower()
    g = globals()
    options = [n[5:] for n in g if n.startswith('load_') and n != 'load_dataset']
    if name not in options:
        raise NameError('Dataset "%s" not found. Options: %s' % (name, 
            ', '.join(options)))
    
    return g['load_' + name](input_node, label_node, *args, **kwargs)


def dataset_shape(name: str):
    """ Returns the number of classes followed by the shape of a sample in a 
        given dataset. """
    name = name.strip().lower()
    g = globals()
    options = [n[5:] for n in g if n.startswith('load_') and n != 'load_dataset']
    if name not in options:
        raise NameError('Dataset "%s" not found. Options: %s' % (name, 
            ', '.join(options)))
    
    return g[name + '_shape']()


def dataset_loss(name: str):
    """ Returns the type of loss function from the dataset. """
    name = name.strip().lower()
    g = globals()
    options = [n[5:] for n in g if n.startswith('load_') and n != 'load_dataset']
    if name not in options:
        raise NameError('Dataset "%s" not found. Options: %s' % (name,
                                                                 ', '.join(options)))

    return g[name + '_loss']()
