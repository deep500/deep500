import time
from caffe2.proto import caffe2_pb2
from caffe2.python import workspace, core, model_helper
import deepbench
import numpy as np

AVG_OVER = 10
RUNS = 30

print('############ Matrix Multiplication ############')
print('Vanilla Caffe2')

device_option = core.DeviceOption(caffe2_pb2.CUDA)  # device = core.DeviceOption(caffe2_pb2.CPU)

# Native caffe2 results
with open('0_cf2_gemm_deepbench.log', 'w') as fp:
    fp.write('m,n,k,a_trans,b_trans,time\n')

for test in deepbench.gemm_training_set:
    print(test)
    with core.DeviceScope(device_option):
        m = model_helper.ModelHelper(name="test_net")
        try:
            var_A = np.random.rand(test.k, test.m).astype(np.float32) \
                if test.a_trans else np.random.rand(test.m, test.k).astype(np.float32)
            var_B = np.random.rand(test.n, test.k).astype(np.float32) \
                if test.b_trans else np.random.rand(test.k, test.n).astype(np.float32)

            workspace.FeedBlob("A", var_A)
            workspace.FeedBlob("B", var_B)

            # create net
            m.net.MatMul(["A", "B"], ["C"], trans_a=test.a_trans, trans_b=test.b_trans)

            times = []

            # Warmup run
            workspace.RunNetOnce(m.param_init_net)
            workspace.RunNetOnce(m.net)

            for i in range(RUNS):
                s = time.time()
                for j in range(AVG_OVER):
                    workspace.RunNetOnce(m.net)
                e = time.time()
                times.append((e - s) / AVG_OVER)
        except:
            times = [-1.0]

    with open('0_cf2_gemm_deepbench.log', 'a') as fp:
        fp.writelines(
            ['{test.m},{test.n},{test.k},{test.a_trans},{test.b_trans},{time:.15f}\n'.format(test=test, time=time) for
             time in times])

##############################################################################

print('Deep500 Caffe2 - test_nativeop_*')
# Deep500 Caffe2

import numpy as np
import deep500 as d5
from deep500.frameworks import caffe2 as d5cf2

with open('0_d5cf2_gemm_deepbench.log', 'w') as fp:
    fp.write('m,n,k,a_trans,b_trans,time,l2_error,max_error\n')

for test in deepbench.gemm_training_set:
    print(test)
    with core.DeviceScope(device_option):
        m = model_helper.ModelHelper(name="test_net")
        try:
            if test.a_trans:
                A = np.random.rand(test.k, test.m).astype(np.float32)
                workspace.FeedBlob("A", A)
                Amult = A.transpose()
            else:
                A = np.random.rand(test.m, test.k).astype(np.float32)
                workspace.FeedBlob("A", A)
                Amult = A
            if test.b_trans:
                B = np.random.rand(test.n, test.k).astype(np.float32)
                workspace.FeedBlob("B", B)
                Bmult = B.transpose()
            else:
                B = np.random.rand(test.k, test.n).astype(np.float32)
                workspace.FeedBlob("B", B)
                Bmult = B

            # Create a matmul function that only receives two variables
            trans_a = 1 if test.a_trans else 0
            trans_b = 1 if test.b_trans else 0
            m.net.MatMul(["A", "B"], ["C"], trans_a=trans_a, trans_b=trans_b)

            # Run Deep500 test
            l2err, maxerr, times = \
                d5cf2.test_nativeop_forward(m, [("A", A), ("B", B)], [("C", Amult @ Bmult)],
                                            metrics=[d5.L2Error(), d5.MaxError(),
                                                     d5.WallclockTime(RUNS * AVG_OVER, AVG_OVER)])
        except:
            l2err = -1.0
            maxerr = -1.0
            times = [-1.0]
    # Clear memory
    workspace.ResetWorkspace()

    with open('0_d5cf2_gemm_deepbench.log', 'a') as fp:
        fp.writelines(['{test.m},{test.n},{test.k},'
                       '{test.a_trans},{test.b_trans},'
                       '{time:.15f},{l2err},{maxerr}\n'.format(test=test,
                                                               time=time, l2err=l2err, maxerr=maxerr)
                       for time in times])
