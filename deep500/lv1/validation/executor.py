from typing import Any, Callable, Dict, List
import numpy as np

from deep500.utils.metrics import TestMetric, WallclockTime
from deep500.utils.onnx_interop.onnx_objects import OnnxModel
from deep500.lv1.graph_executor import GraphExecutor

def _test_executor_internal(
    func: Callable,
    name: str,
    model: OnnxModel,
    inputs: Dict[str, np.ndarray],
    reference_outputs: Dict[str, np.ndarray],
    metrics: List[TestMetric],
    *args
) -> List[Any]:
    if inputs is None:
        inputs = {}
     
    # Obtain input nodes
    input_nodes = model.get_input_nodes()
    for input_node in input_nodes:
        # Fill in random inputs if not specified
        if input_node.name not in inputs:
            input_shape = input_node.type.shape.shape
            input = np.random.rand(*input_shape).astype(input_node.type.type.to_numpy())
            inputs[input_node.name] = input
    
    normal_metrics = [m for m in metrics if m.reruns == 0]
    rerun_metrics = [m for m in metrics if m.reruns > 0]

    for metric in normal_metrics:
        metric.begin(inputs)
    outputs = func(inputs, *args)
    for metric in normal_metrics:
        metric.end(outputs)

    for metric in rerun_metrics:
        for i in range(metric.reruns):
            metric.begin(inputs)
            outputs = func(inputs, *args)
            metric.end(outputs)

    results = []
    for k, output in outputs.items():
        if reference_outputs is not None:
            ref_out = reference_outputs[k]
        else:
            ref_out = None
            
        # Execute metrics.
        for metric in metrics:
            result = metric.measure(inputs, output, ref_out)
            results.append(result)
            summary = metric.measure_summary(inputs, output, ref_out)
            print("{} on {} for output '{}': {}".format(
                    type(metric).__name__, name, k, summary))

    return results


def test_executor_inference(
    executor: GraphExecutor,
    inputs: Dict[str, np.ndarray] = None,
    reference_outputs: Dict[str, np.ndarray] = None,
    metrics: List[TestMetric] = [WallclockTime()]
) -> List[Any]:
    """ Tests a graph executor's forward evaluation of a network given 
        input tensors, or random values if not specified.

        @param executor The executor to test.
        @param inputs Mapping of input node names to numpy arrays to feed in.
        @param reference_outputs A dictionary of reference outputs (mapping 
               from node names to values).
        @param metrics A list of TestMetric objects to measure.
        @return List of test metric results for every output.
    """
    return _test_executor_internal(executor.inference, 'inference', 
        executor.model, inputs, reference_outputs, metrics)
    
def test_executor_backprop(
    executor: GraphExecutor,
    output: str,
    inputs: Dict[str, np.ndarray] = None,
    reference_outputs: Dict[str, np.ndarray] = None,
    metrics: List[TestMetric] = [WallclockTime()]
) -> List[Any]:
    """ Tests a graph executor's gradient backpropagation given 
        the node to derive and input tensors (or random values if not specified).

        @param executor The executor to test.
        @param output The node name to compute gradient for.
        @param inputs Mapping of input node names to numpy arrays to feed in.
        @param reference_outputs A dictionary of reference outputs (mapping 
               from node names to values).
        @param metrics A list of TestMetric objects to measure.
        @return List of test metric results for every output.
    """
    return _test_executor_internal(
        executor.inference_and_backprop, 'backprop', executor.model, inputs, 
        reference_outputs, metrics, output)
