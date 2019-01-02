import functools
import numpy as np
from typing import List, Any

from deep500.utils.metrics import TestMetric
from deep500.lv0.operators.operator_interface import CustomOp
from deep500.lv0.validation.metrics import DefaultOpMetrics


def _numel(arr):
    """ Returns the number of elements for an array using a numpy-like
        interface. """
    return functools.reduce(lambda a, b: int(a) * int(b), arr.shape, 1)

def _op_fwd_list(op, inputs):
    output = op.forward(*inputs)
    if op.num_outputs == 1 and not isinstance(op, list):
        return [output]
    return output

def test_op_gradient(
    op: CustomOp,
    inputs: List[np.ndarray],
    grad_inputs: List[np.ndarray] = None,
    eps: float = 1e-5,
    metrics: List[TestMetric] = DefaultOpMetrics(),
    test_accuracy: bool = True
) -> List[Any]:
    """ Tests an operator's gradient by forward-evaluation of a list of given
        input tensors, using numerical differentiation.
        In particular, the test runs op.forward multiple times to produce the
        Jacobian matrix (using central difference), and compares the result
        with op.backward (where input gradients are set to 1).

        @param op An operator to test.
        @param inputs An array of inputs to pass to the operator.
        @param grad_inputs An array of input gradients to pass to the operator
        @param eps The epsilon value (finite difference) for numerical
                   differentiation.
        @param metrics A list of TestMetric objects to measure.
        @return List of test metric results for every output.
    """

    # Compute the outputs once to get the shape.
    # There is a small issue here with the return type of 'forward': sometimes
    # it is a tensor, and sometimes a list of tensors. To resolve that, we 
    # detect the type in `_op_fwd_list` and return a list anyway
    outputs = _op_fwd_list(op, inputs)

    # Validate input gradients (if given) or generate random ones.
    if grad_inputs:
        assert(op.num_outputs == len(grad_inputs))
        for o, g in zip(op.output_descriptors, grad_inputs):
            assert(list(o.shape) == list(g.shape))
    else:
        grad_inputs = [np.random.rand(*o.shape).astype(o.dtype)
                       for o in op.output_descriptors]

    # Assuming numpy or numpy-interface for tensors.
    # num_in_elements = [_numel(i) for i in inputs]
    num_out_elements = [_numel(o) for o in outputs]

    # Compute analytical gradient outputs.
    # There is a small issue here with the return type of 'backward'.

    normal_metrics = [m for m in metrics if m.reruns == 0]
    rerun_metrics = [m for m in metrics if m.reruns > 0]

    for metric in normal_metrics:
        metric.begin(inputs)
    analytical_grad_outputs = op.backward(grad_inputs, inputs, outputs)
    for metric in normal_metrics:
        metric.end(outputs)

    for metric in rerun_metrics:
        for i in range(metric.reruns):
            metric.begin(grad_inputs)
            grad_outputs = op.backward(grad_inputs, inputs, outputs)
            metric.end(grad_outputs)

    # Compute numerical jacobian.
    if test_accuracy:
        jacobian = numerical_jacobian(op, inputs, num_out_elements, eps)

    results = []
    for i in range(op.num_outputs):
        numerical_grad_outputs = []
        # Compute numerical gradient outputs.
        if test_accuracy:
            grad_in_flat = np.reshape(grad_inputs[i], (1, num_out_elements[i]))
            for j in range(len(inputs)):
                numeric_grad_out_flat = np.dot(grad_in_flat, jacobian[i, j])
                numeric_grad_out = np.reshape(numeric_grad_out_flat,
                                              inputs[j].shape)
                numerical_grad_outputs.append(numeric_grad_out)
        # Execute metrics.
        for metric in metrics:
            rl = i * len(inputs)
            rr = (i + 1) * len(inputs)
            if not test_accuracy:
                numerical_grad_outputs = analytical_grad_outputs[rl:rr]
            result = metric.measure(inputs, analytical_grad_outputs[rl:rr],
                                    numerical_grad_outputs)
            results.append(result)
            summary = metric.measure_summary(inputs,
                                             analytical_grad_outputs[rl:rr],
                                             numerical_grad_outputs)
            print("{} on gradient for output {}: {}".format(
                  type(metric).__name__, i, summary))

    return results


def numerical_jacobian(
    op: CustomOp,
    inputs: List[np.ndarray],
    output_elements: List[int],
    eps: float = 1e-5
) -> np.ndarray:
    """ Computes the jacobian matrix of the operator outputs.

        @param op An operator.
        @param inputs An array of inputs to pass to the operator.
        @param output_elements List of numbers of elements in the output tensors.
        @param eps The epsilon value (finite difference) for numerical
        differentiation.
    """

    jacobian = np.empty((len(output_elements), len(inputs)), dtype=object)

    num_type = inputs[0].dtype

    for i, numel_y in enumerate(output_elements):
        for j, x in enumerate(inputs):

            gradient = np.zeros((numel_y, _numel(x)), dtype=num_type)

            for row in range(_numel(x)):

                x_pos = x.copy()
                x_pos.ravel()[row] += eps
                inputs_pos = inputs[:j] + [x_pos] + inputs[j+1:]
                y_pos = _op_fwd_list(op, inputs_pos)[i]

                x_neg = x.copy()
                x_neg.ravel()[row] -= eps
                inputs_neg = inputs[:j] + [x_neg] + inputs[j+1:]
                y_neg = _op_fwd_list(op, inputs_neg)[i]

                diff = (y_pos - y_neg) / (2 * eps)

                for col in range(numel_y):
                    gradient[col][row] = diff.flatten()[col]

            jacobian[i, j] = gradient

    return jacobian
