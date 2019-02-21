from typing import List, Any, Tuple, Optional

import numpy as np
from caffe2.proto import caffe2_pb2
from caffe2.python import core, workspace

from deep500 import DefaultOpMetrics, TestMetric


def test_nativeop_forward(
        model,
        inputs: List[Tuple[str, np.ndarray]],
        reference_outputs: List[Tuple[str, Optional[np.ndarray]]],
        metrics: List[TestMetric] = DefaultOpMetrics()
) -> List[Any]:
    """ Tests a framework-native operator's forward method on list of given
        input tensors, without surrounding overhead.
        Similar to `test_op_forward`, but does not incur the overhead of
        converting from/to numpy arrays, creating multiple sessions etc.

        @param op An operator to test.
        @param inputs An array of inputs to pass to the operator.
        @param reference_outputs An array of reference outputs.
        @param metrics A list of TestMetric objects to measure.
        @return List of test metric results for every output.
    """
    assert isinstance(inputs, (list, tuple)) == True

    normal_metrics = [m for m in metrics if m.reruns == 0]
    rerun_metrics = [m for m in metrics if m.reruns > 0]

    for (name, inp) in inputs:
        workspace.FeedBlob(name, inp)

    num_outputs = len(reference_outputs)

    # Create a single session
    workspace.CreateNet(model.net)

    for metric in normal_metrics:
        metric.begin(inputs)

    workspace.RunNet(model.net)
    outputs = workspace.FetchBlobs([name for (name, out) in reference_outputs])
    for metric in normal_metrics:
        metric.end(outputs)

    for metric in rerun_metrics:
        if not metric.requires_outputs:
            for i in range(metric.reruns):
                metric.begin(inputs)
                workspace.RunNet(model.net)
                metric.end()
        else:
            for i in range(metric.reruns):
                metric.begin(inputs)
                workspace.RunNet(model.net)
                outputs = workspace.FetchBlobs([name for (name, out) in reference_outputs])
                metric.end(outputs)

    if not isinstance(outputs, (list, tuple)):
        outputs = [outputs]

    results = []
    for i in range(num_outputs):
        # Execute metrics.
        for metric in metrics:
            result = metric.measure(inputs, outputs[i], (reference_outputs[i])[1])
            results.append(result)
            summary = metric.measure_summary(inputs, outputs[i],
                                             reference_outputs[i][1])
            print("{} on native inference for output {}: {}".format(
                type(metric).__name__, i, summary))

    return results