# Validates a native TensorFlow operator (here tf.matmul) using a Deep500 
# Operator.
import numpy as np
import tensorflow as tf
from deep500.frameworks import tensorflow as d5tf

from deep500.frameworks import reference as d5ref
from deep500 import test_op_forward, test_op_gradient

if __name__ == '__main__':
    with tf.Session() as sess:
        # Create two matrices A and B
        A = np.random.rand(5, 40).astype(np.float64)
        B = np.random.rand(40, 5).astype(np.float64)

        # Native (TensorFlow) versions of A and B
        native_A = tf.Variable(A, expected_shape=A.shape, dtype=A.dtype)
        native_B = tf.Variable(B, expected_shape=B.shape, dtype=B.dtype)

        # Create a Deep500 Operator from the tf.matmul function.
        # The native TF operator accepts two inputs and one output, whose
        # tensor descriptors (i.e., shape and data type) are created below.
        op = d5tf.custom_op_from_native(tf.matmul, 
            [d5ref.desc_from_tensor(A), d5ref.desc_from_tensor(B)],
            [d5ref.desc_from_tensor(A@B)])

        # Test forward (first using Deep500, which copies to/from numpy, 
        # then natively through the framework)
        test_op_forward(op, [A, B], [A@B])
        d5tf.test_nativeop_forward(tf.matmul, [native_A, native_B], [A @ B])
    
        # Test backward (gradient) using Deep500 and native tests.
        test_op_gradient(op, [A, B], eps=1e-7)
        d5tf.test_nativeop_gradient(tf.matmul, [native_A, native_B], eps=1e-7)
