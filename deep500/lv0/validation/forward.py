import functools
import numpy as np
from typing import List, Any
import unittest

from deep500.lv0.validation.gradient import _op_fwd_list
from deep500.lv0.operators.operator_interface import CustomOp
from deep500.lv0.validation.metrics import DefaultOpMetrics
from deep500.utils.metrics import TestMetric


def test_op_forward(
    op: CustomOp,
    inputs: List[np.ndarray],
    reference_outputs: List[np.ndarray],
    metrics: List[TestMetric] = DefaultOpMetrics()
) -> List[Any]:
    """ Tests an operator's forward method on list of given input tensors.

        @param op An operator to test.
        @param inputs An array of inputs to pass to the operator.
        @param reference_outputs An array of reference outputs.
        @param metrics A list of TestMetric objects to measure.
        @return List of test metric results for every output.
    """

    normal_metrics = [m for m in metrics if m.reruns == 0]
    rerun_metrics = [m for m in metrics if m.reruns > 0]

    for metric in normal_metrics:
        metric.begin(inputs)
    outputs = _op_fwd_list(op, inputs)
    for metric in normal_metrics:
        metric.end(outputs)

    for metric in rerun_metrics:
        for i in range(metric.reruns):
            metric.begin(inputs)
            outputs = _op_fwd_list(op, inputs)
            metric.end(outputs)

    results = []
    for i in range(op.num_outputs):
        # Execute metrics.
        for metric in metrics:
            result = metric.measure(inputs, outputs[i], reference_outputs[i])
            results.append(result)
            summary = metric.measure_summary(inputs, outputs[i],
                                             reference_outputs[i])
            print("{} on inference for output {}: {}".format(
                    type(metric).__name__, i, summary))

    return results
