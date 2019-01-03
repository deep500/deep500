import time
import tensorflow as tf
import deepbench

AVG_OVER = 10
RUNS = 30

print('############ Convolution ############')
print('Vanilla Tensorflow')

# Native Tensorflow results
with open('0_tf_conv_deepbench.log', 'w') as fp:
    fp.write('n,c,h,w,k,r,s,h_stride,w_stride,time\n')

for test in deepbench.conv_training_set:
    # Skip tests with padding
    #if test.pad_w > 0 or test.pad_h > 0:
    #    continue
    print(test)
    with tf.Session() as sess:
        try:
            # Create Tensorflow graph
            var_X = tf.Variable(tf.random_uniform([test.n, test.c, test.h, test.w]), name="var_X", dtype=tf.float32)
            var_C = tf.layers.conv2d(var_X, filters=test.k, kernel_size=[test.r, test.s], strides=[test.hstride, test.wstride], data_format='channels_first')

            # Initialize random matrices
            init = tf.global_variables_initializer()
            sess.run(init)
            times = []

            # Warmup run
            sess.run(var_C)

            for i in range(RUNS):
                s = time.time()
                for j in range(AVG_OVER):
                    sess.run(var_C)
                e = time.time()
                times.append((e - s) / AVG_OVER)
        except:
            raise
            times = [-1.0]
    tf.reset_default_graph()

    with open('0_tf_conv_deepbench.log', 'a') as fp:
        fp.writelines(['{test.n},{test.c},{test.h},{test.w},{test.k},{test.r},{test.s},{test.hstride},{test.wstride},{time:.15f}\n'.format(test=test, time=time) for time in times])

##############################################################################

print('Deep500 Tensorflow - test_nativeop_*')
# Deep500 Tensorflow

import numpy as np
import deep500 as d5
from deep500.frameworks import tensorflow as d5tf
from deep500.frameworks import reference as d5ref
from deep500.frameworks.reference.custom_operators.python import conv_op

with open('0_d5tf_conv_deepbench.log', 'w') as fp:
    fp.write('n,c,h,w,k,r,s,h_stride,w_stride,time,l2_error,max_error\n')

for test in deepbench.conv_training_set:
    # Skip tests with padding
    #if test.pad_w > 0 or test.pad_h > 0:
    #    continue
    print(test)
    try:
        # Create Tensorflow graph and numpy arrays for verification
        X = np.random.rand(test.n, test.c, test.h, test.w).astype(np.float32)
        W = np.random.rand(test.k, test.c, test.r, test.s).astype(np.float32)

        var_X = tf.Variable(X, name="var_X", dtype=tf.float32)
        var_C = tf.layers.conv2d(var_X, filters=test.k, kernel_size=[test.r, test.s], 
                                 strides=[test.hstride, test.wstride], data_format='channels_first',
                                 kernel_initializer=tf.constant_initializer(W))

        # Create a conv2d function that only receives one variable
        cnv2d = lambda x: tf.layers.conv2d(x, filters=test.k, kernel_size=[test.r, test.s], 
                                           strides=[test.hstride, test.wstride], data_format='channels_first', 
                                           kernel_initializer=tf.constant_initializer(W))

        times, = \
            d5tf.test_nativeop_forward(cnv2d, [var_X], [None],
                                       metrics=[d5.WallclockTime(RUNS*AVG_OVER, AVG_OVER)])
    except Exception as ex:
        print('Exception:', ex)
        times = [-1.0]
    # Clear memory
    tf.reset_default_graph()

    with open('0_d5tf_conv_deepbench.log', 'a') as fp:
        fp.writelines(['{test.n},{test.c},{test.h},{test.w},'
                       '{test.k},{test.r},{test.s},'
                       '{test.hstride},{test.wstride},'
                       '{time:.15f}\n'.format(test=test, 
                            time=time) 
                        for time in times])
