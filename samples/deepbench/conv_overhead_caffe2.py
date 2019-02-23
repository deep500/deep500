import time
import numpy as np
from caffe2.proto import caffe2_pb2
from caffe2.python import workspace, core, model_helper, utils, brew
import deepbench

AVG_OVER = 100
RUNS = 30

device_option = core.DeviceOption(caffe2_pb2.CUDA)  # device = core.DeviceOption(caffe2_pb2.CPU)
print('############ Convolution ############')
print('Vanilla Caffe2')

# Native Caffe2 results
with open('0_cf2_conv_deepbench.log', 'w') as fp:
    fp.write('n,c,h,w,k,r,s,h_stride,w_stride,time\n')

for test in deepbench.conv_training_set:
    print(test)
    try:
        with core.DeviceScope(device_option):
            m = model_helper.ModelHelper(name="test_net")
            # Create Pytorch "model"
            X = np.random.rand(test.n, test.c, test.h, test.w).astype(np.float32)
            W = np.random.rand(test.k, test.c, test.r, test.s).astype(np.float32)
            B = np.random.rand(test.k).astype(np.float32)
            workspace.FeedBlob("X", X, device_option=device_option)
            workspace.FeedBlob("W", W, device_option=device_option)
            workspace.FeedBlob("B", B, device_option=device_option)

            order = "NCHW"

            m.net.Conv(["X", "W", "B"], ["Y"],
                         kernels=[test.r,test.s],
                         strides=[test.hstride, test.wstride],
                         pads=[test.pad_h, test.pad_w] * 2)

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
    except Exception as ex:
        print('Exception:', ex)
        times = [-1.0]

    with open('0_cf2_conv_deepbench.log', 'a') as fp:
        fp.writelines([
                          '{test.n},{test.c},{test.h},{test.w},{test.k},{test.r},{test.s},{test.hstride},{test.wstride},{time:.15f}\n'.format(
                              test=test, time=time) for time in times])

##############################################################################


print('Deep500 Caffe2 - test_nativeop_*')
# Deep500 Caffe2

import deep500 as d5
from deep500.frameworks import caffe2 as d5cf2

with open('0_d5cf2_conv_deepbench.log', 'w') as fp:
    fp.write('n,c,h,w,k,r,s,h_stride,w_stride,time,l2_error,max_error\n')

for test in deepbench.conv_training_set:
    print(test)
    try:
        with core.DeviceScope(device_option):
            m = model_helper.ModelHelper(name="test_net")
            # Create Pytorch "model"
            X = np.random.rand(test.n, test.c, test.h, test.w).astype(np.float32)
            W = np.random.rand(test.k, test.c, test.r, test.s).astype(np.float32)
            B = np.random.rand(test.k).astype(np.float32)

            workspace.FeedBlob("W", W, device_option=device_option)
            workspace.FeedBlob("B", B, device_option=device_option)
            m.net.Conv(["X", "W", "B"], ["Y"],
                         kernels=[test.r, test.s],
                         strides=[test.hstride, test.wstride],
                         pads=[test.pad_h, test.pad_w] * 2)

            times, = d5cf2.test_nativeop_forward(m, [("X", X)], [("Y", None)],
                                                 metrics=[d5.WallclockTime(RUNS * AVG_OVER, AVG_OVER)])
    except Exception as ex:
        print('Exception:', ex)
        times = [-1.0]

    with open('0_d5cf2_conv_deepbench.log', 'a') as fp:
        fp.writelines(['{test.n},{test.c},{test.h},{test.w},'
                       '{test.k},{test.r},{test.s},'
                       '{test.hstride},{test.wstride},'
                       '{time:.15f}\n'.format(test=test, time=time)
                       for time in times])
