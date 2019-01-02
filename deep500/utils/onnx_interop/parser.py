import onnx
import logging
from deep500.utils.onnx_interop.onnx_objects import OnnxModel

def parse_model(onnx_model: onnx.ModelProto) -> OnnxModel:
    """
    @param model Existing ONNX binary model
    @return parsed model
    """
    return OnnxModel.create_from_onnx_model(onnx_model)

def load_and_parse_model(path: str) -> OnnxModel:
    """
    @param path Path to .onnx file
    @return parsed model
    """
    with open(path, 'rb') as model_file:
        binary = model_file.read()

    return load_and_parse_binary(binary)

def clean_model(model: OnnxModel) -> OnnxModel:
    # Remove superfluous input nodes
    actual_inputs = set()
    for n in model.graph.nodes:
        for i in n.input:
            actual_inputs.add(i)
    oldlen = len(model.graph.inputs)
    model.graph.inputs = [i for i in model.graph.inputs if i.name in actual_inputs]
    removed_nodes = oldlen - len(model.graph.inputs)
    if removed_nodes > 0:
        print('Removed %d superfluous nodes in graph' % removed_nodes)
    return model

def load_and_parse_binary(binary: onnx.ModelProto) -> OnnxModel:
    model = onnx.load_from_string(binary)
    model = OnnxModel.create_from_onnx_model(model)
    model = clean_model(model)
    return model


def load_model_only(path: str) -> onnx.ModelProto:
    """
     @param path path to file
     @return deserialized onnx model
     """
    with open(path, 'rb') as model_file:
        binary = model_file.read()
    model = onnx.load_from_string(binary)
    return model


def save_onnxdef(onnx_model, path: str):
    """
    @param onnx_model onnx model file
    @param path path to save
    """
    with open(path, 'wb') as onnx_file:
        onnx_file.write(onnx_model.SerializeToString())
