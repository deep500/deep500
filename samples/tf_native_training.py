# Trains a native TensorFlow model using a Deep500 executor and a native 
# optimizer.
import tensorflow as tf
import numpy as np

import deep500 as d5
from deep500.frameworks import tensorflow as d5tf
from deep500.frameworks import reference as d5ref


# Simple linear model in TensorFlow
def tfmodel(x, y):  
    W = tf.Variable(5.)
    b = tf.Variable(5.)
    
    pred = W * x + b
    cost = tf.squared_difference(pred, y)
    return pred, cost

if __name__ == '__main__':
    x = tf.placeholder(tf.float32)
    y = tf.placeholder(tf.float32)

    # Demonstrating the use of tensors in native optimizers
    lr = tf.placeholder(tf.float32)

    pred, cost = tfmodel(x, y)

    executor = d5tf.TensorflowNativeGraphExecutor(cost, pred.name)

    # Either optimizer works, but d5tf is a native operator and thus faster
    optimizer = d5tf.GradientDescent(executor, cost.name, lr)
    #optimizer = d5ref.GradientDescent(executor, cost.name)

    print('Before', executor.inference({x: 3., y: 6.}))

    for i in range(1000):
        r = np.random.rand() * 3
        optimizer.step({x: r, y: 2 * r, lr: 0.1})

    # After training
    print('After', executor.inference({x: 3., y: 6.}))
