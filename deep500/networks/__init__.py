from .pytorch_mnist import export_simple_cnn
from .pytorch_deep_mnist import export_deep_mnist
from .pytorch_resnet import export_resnet
from .pytorch_wide_resnet import export_wide_resnet

def export_network(name: str, batch_size: int, *args, **kwargs) -> str:
    """ Exports a network by name and returns the path to the output file.
        @param name Network name (options: simple_cnn, deep_mnist, resnet, wide_resnet). 
        @param batch_size Minibatch size to use
        @param args Other arguments to pass to the constructor
        @param kwargs Other keyword arguments to pass to the constructor (e.g., `file_path`)
        @return Path to exported network filename
    """
    name = name.strip().lower()
    g = globals()
    options = [n[7:] for n in g if n.startswith('export_') and n != 'export_network']
    if name not in options:
        raise ValueError('Network "%s" not found, options: %s' % (name,
            ', '.join(options)))
    
    return g['export_' + name](batch_size, *args, **kwargs)

def create_model(network_name: str, batch_size: int, *args, **kwargs):
    """ Creates an ONNX model from a reference network name.
        @param network_name Network name (options: simple_cnn, deep_mnist, resnet, wide_resnet). 
        @param batch_size Minibatch size to use
        @param args Other arguments to pass to the constructor
        @param kwargs Other keyword arguments to pass to the constructor (e.g., `file_path`)
        @return A 3-tuple of (model, input node, output node)
    """
    import deep500 as d5
    onnx_file = export_network(network_name, batch_size, *args, **kwargs)
    model = d5.parser.load_and_parse_model(onnx_file)

    # Recover input and output nodes (assuming only one input and one output)
    innode = model.get_input_nodes()[0].name
    outnode = model.get_output_nodes()[0].name

    return model, innode, outnode
