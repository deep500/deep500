from .mnist import load_mnist, mnist_shape, load_fashion_mnist, fashion_mnist_shape
from .cifar import load_cifar10, load_cifar100, cifar10_shape, cifar100_shape
from .imagenet import load_imagenet, imagenet_shape


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
