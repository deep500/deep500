import time
import tensorflow as tf
import deepbench

AVG_OVER = 10
RUNS = 30

print('############ Matrix Multiplication ############')
print('Vanilla Tensorflow')

# Native Tensorflow results
with open('0_tf_gemm_deepbench.log', 'w') as fp:
    fp.write('m,n,k,a_trans,b_trans,time\n')

for test in deepbench.gemm_training_set:
    print(test)
    with tf.Session() as sess:
        try:
            # Create Tensorflow graph
            if test.a_trans:
                var_A = tf.Variable(tf.random_uniform([test.k, test.m]), name="var_A", dtype=tf.float32)
            else:
                var_A = tf.Variable(tf.random_uniform([test.m, test.k]), name="var_A", dtype=tf.float32)
            if test.b_trans:
                var_B = tf.Variable(tf.random_uniform([test.n, test.k]), name="var_B", dtype=tf.float32)
            else:
                var_B = tf.Variable(tf.random_uniform([test.k, test.n]), name="var_B", dtype=tf.float32)
            var_C = tf.matmul(var_A, var_B, test.a_trans, test.b_trans)
            

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
            times = [-1.0]
    tf.reset_default_graph()

    with open('0_tf_gemm_deepbench.log', 'a') as fp:
        fp.writelines(['{test.m},{test.n},{test.k},{test.a_trans},{test.b_trans},{time:.15f}\n'.format(test=test, time=time) for time in times])

##############################################################################

print('Deep500 Tensorflow - test_nativeop_*')
# Deep500 Tensorflow

import numpy as np
import deep500 as d5
from deep500.frameworks import tensorflow as d5tf

with open('0_d5tf_gemm_deepbench.log', 'w') as fp:
    fp.write('m,n,k,a_trans,b_trans,time,l2_error,max_error\n')

for test in deepbench.gemm_training_set:
    print(test)
    try:
        # Create Tensorflow graph and numpy arrays for verification
        if test.a_trans:
            A = np.random.rand(test.k, test.m).astype(np.float32)
            var_A = tf.Variable(A, name="var_A", dtype=tf.float32)
            Amult = A.transpose()
        else:
            A = np.random.rand(test.m, test.k).astype(np.float32)
            var_A = tf.Variable(A, name="var_A", dtype=tf.float32)
            Amult = A
        if test.b_trans:
            B = np.random.rand(test.n, test.k).astype(np.float32)
            var_B = tf.Variable(B, name="var_B", dtype=tf.float32)
            Bmult = B.transpose()
        else:
            B = np.random.rand(test.k, test.n).astype(np.float32)
            var_B = tf.Variable(B, name="var_B", dtype=tf.float32)
            Bmult = B

        # Create a matmul function that only receives two variables
        mm = lambda a,b: tf.matmul(a,b, test.a_trans, test.b_trans)

        # Run Deep500 test
        l2err, maxerr, times = \
            d5tf.test_nativeop_forward(mm, [var_A, var_B], [Amult @ Bmult],
                                       metrics=[d5.L2Error(), d5.MaxError(), 
                                                d5.WallclockTime(RUNS*AVG_OVER, AVG_OVER)])
    except:
        l2err = -1.0
        maxerr = -1.0
        times = [-1.0]
    # Clear memory
    tf.reset_default_graph()

    with open('0_d5tf_gemm_deepbench.log', 'a') as fp:
        fp.writelines(['{test.m},{test.n},{test.k},'
                       '{test.a_trans},{test.b_trans},'
                       '{time:.15f},{l2err},{maxerr}\n'.format(test=test, 
                            time=time, l2err=l2err, maxerr=maxerr) 
                        for time in times])
