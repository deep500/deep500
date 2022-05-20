from typing import List, Any

import numpy as np
# Compatiable tf version for v1
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from deep500.lv0.validation.metrics import DefaultOpMetrics
from deep500.utils.metrics import TestMetric
from deep500.lv0.validation.gradient import _numel, numerical_jacobian

from deep500.frameworks.tensorflow.custom_operators.tf import custom_op_from_native, desc_from_tensor

def test_nativeop_forward(
    op: tf.Operation,
    inputs: List[tf.Tensor],
    reference_outputs: List[np.ndarray],
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

    # Construct TF graph first
    output_tensors = op(*inputs)
    if isinstance(output_tensors, (list, tuple)):
        num_outputs = len(output_tensors)
    else:
        num_outputs = 1

    # Create a single session
    with tf.Session() as sess:
        # Initialize session once
        sess.run(tf.initializers.global_variables())


        for metric in normal_metrics:
            metric.begin(inputs)
        outputs = sess.run(output_tensors)
        for metric in normal_metrics:
            metric.end(outputs)
        
        for metric in rerun_metrics:
            for i in range(metric.reruns):
                metric.begin(inputs)
                outputs = sess.run(output_tensors)
                metric.end(outputs)

        if not isinstance(outputs, (list, tuple)) and num_outputs == 1:
            outputs = [outputs]

        results = []
        for i in range(num_outputs):
            # Execute metrics.
            for metric in metrics:
                result = metric.measure(inputs, outputs[i], reference_outputs[i])
                results.append(result)
                summary = metric.measure_summary(inputs, outputs[i],
                                                 reference_outputs[i])
                print("{} on native inference for output {}: {}".format(
                        type(metric).__name__, i, summary))

        return results

def test_nativeop_gradient(
    op: tf.Operation,
    inputs: List[tf.Tensor],
    grad_inputs: List[tf.Tensor] = None,
    eps: float = 1e-5,
    metrics: List[TestMetric] = DefaultOpMetrics()
) -> List[Any]:
    """ Tests an operator's gradient by forward-evaluation of a list of given
        input tensors, using numerical differentiation.
        In particular, the test runs op.forward multiple times to produce the
        Jacobian matrix (using central difference), and compares the result
        with op.backward (where input gradients are set to 1).

        @param op An operator to test
        @param inputs An array of inputs to pass to the operator
        @param grad_inputs An array of input gradients to pass to the operator
        @param eps The epsilon value (finite difference) for numerical
        differentiation.
        @return List of test metric results for every output.
    """
    assert isinstance(inputs, (list, tuple)) == True

    # Construct graph and obtain gradient tensors
    output_tensors = op(*inputs)
    grad_tensors = tf.gradients(output_tensors, inputs, grad_ys=grad_inputs)

    if isinstance(output_tensors, (list, tuple)):
        num_outputs = len(output_tensors)
    else:
        num_outputs = 1
        output_tensors = [output_tensors]
    num_grads = len(inputs)

    # num_in_elements = [_numel(i) for i in inputs]
    num_out_elements = [_numel(o) for o in output_tensors]

    normal_metrics = [m for m in metrics if m.reruns == 0]
    rerun_metrics = [m for m in metrics if m.reruns > 0]

    # Create a single session
    with tf.Session() as sess:
        # Initialize session once
        sess.run(tf.initializers.global_variables())

        # Compute analytical gradient outputs.
        for metric in normal_metrics:
            metric.begin(inputs)
        analytical_grad_outputs = sess.run(grad_tensors)
        for metric in normal_metrics:
            metric.end(analytical_grad_outputs)

        for metric in rerun_metrics:
            for i in range(metric.reruns):
                metric.begin(grad_inputs)
                grad_outputs = sess.run(grad_tensors)
                metric.end(grad_outputs)

        # Compute numerical jacobian.
        cop = custom_op_from_native(op, [desc_from_tensor(i) for i in inputs],
                                    [desc_from_tensor(o) for o in output_tensors])
        # Convert Tensorflow tensors to numpy arrays
        numpy_inputs = sess.run(inputs)
        jacobian = numerical_jacobian(cop, numpy_inputs, num_out_elements, eps)

        results = []
        for i in range(len(num_out_elements)):

            # Compute numerical gradient outputs.
            numerical_grad_outputs = []
            if grad_inputs is not None:
                grad_in_flat = np.reshape(grad_inputs[i], (1, num_out_elements[i]))
            else:
                grad_in_flat = np.ones(shape=(1, num_out_elements[i]), 
                                       dtype=jacobian[i, 0].dtype)

            for j in range(len(inputs)):
                numeric_grad_out_flat = np.dot(grad_in_flat, jacobian[i, j])
                numeric_grad_out = np.reshape(numeric_grad_out_flat,
                                              inputs[j].shape)
                numerical_grad_outputs.append(numeric_grad_out)

            # Convert outputs to numpy array
            rl = i * len(inputs)
            rr = (i + 1) * len(inputs)
            numpy_grad_outputs = analytical_grad_outputs[rl:rr]

            # Execute metrics.
            for metric in metrics:
                result = metric.measure(inputs, numpy_grad_outputs,
                                        numerical_grad_outputs)
                results.append(result)
                summary = metric.measure_summary(inputs,
                                                 numpy_grad_outputs,
                                                 numerical_grad_outputs)
                print("{} on native gradient for output {}: {}".format(
                      type(metric).__name__, i, summary))

        return results
