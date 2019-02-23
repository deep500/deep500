import time
import torch
import numpy as np
import deepbench

AVG_OVER = 100
RUNS = 30

print('############ Convolution ############')
print('Vanilla Pytorch')

dev = torch.device('cuda')

# Native Pytorch results
with open('0_pt_conv_deepbench.log', 'w') as fp:
    fp.write('n,c,h,w,k,r,s,h_stride,w_stride,time\n')

for test in deepbench.conv_training_set:
    print(test)
    try:
        # Create Pytorch "model"
        X = np.random.rand(test.n, test.c, test.h, test.w).astype(np.float32)
        W = np.random.rand(test.k, test.c, test.r, test.s).astype(np.float32)
        B = np.random.rand(test.k).astype(np.float32)
        var_X = torch.tensor(X, dtype=torch.float32, device=dev)
        var_W = torch.tensor(W, dtype=torch.float32, device=dev)
        var_B = torch.tensor(B, dtype=torch.float32, device=dev)
        model = torch.nn.Conv2d(test.c, test.k, (test.r, test.s), stride=(test.hstride, test.wstride), padding=(test.pad_h, test.pad_w))

        # Set weights and bias
        model.weight = torch.nn.Parameter(var_W)
        model.bias = torch.nn.Parameter(var_B)

        times = []

        # Warmup run
        model(var_X)

        for i in range(RUNS):
            torch.cuda.synchronize()
            s = time.time()
            for j in range(AVG_OVER):
                model(var_X)
            torch.cuda.synchronize()
            e = time.time()
            times.append((e - s) / AVG_OVER)
    except Exception as ex:
        print('Exception:', ex)
        times = [-1.0]

    with open('0_pt_conv_deepbench.log', 'a') as fp:
        fp.writelines(['{test.n},{test.c},{test.h},{test.w},{test.k},{test.r},{test.s},{test.hstride},{test.wstride},{time:.15f}\n'.format(test=test, time=time) for time in times])

##############################################################################


print('Deep500 Pytorch - test_nativeop_*')
# Deep500 Pytorch

import deep500 as d5
from deep500.frameworks import pytorch as d5pt

with open('0_d5pt_conv_deepbench.log', 'w') as fp:
    fp.write('n,c,h,w,k,r,s,h_stride,w_stride,time,l2_error,max_error\n')

for test in deepbench.conv_training_set:
    print(test)
    try:
        # Create Pytorch "model"
        X = np.random.rand(test.n, test.c, test.h, test.w).astype(np.float32)
        W = np.random.rand(test.k, test.c, test.r, test.s).astype(np.float32)
        B = np.random.rand(test.k).astype(np.float32)
        var_X = torch.tensor(X, dtype=torch.float32, device=dev)
        var_W = torch.tensor(W, dtype=torch.float32, device=dev)
        var_B = torch.tensor(B, dtype=torch.float32, device=dev)
        model = torch.nn.Conv2d(test.c, test.k, (test.r, test.s), stride=(test.hstride, test.wstride), padding=(test.pad_h, test.pad_w))

        # Set weights and bias
        model.weight = torch.nn.Parameter(var_W)
        model.bias = torch.nn.Parameter(var_B)

        times, = \
            d5pt.test_nativeop_forward(model, [var_X], [None],
                                       metrics=[d5.WallclockTime(RUNS*AVG_OVER, AVG_OVER)])
    except Exception as ex:
        print('Exception:', ex)
        times = [-1.0]

    with open('0_d5pt_conv_deepbench.log', 'a') as fp:
        fp.writelines(['{test.n},{test.c},{test.h},{test.w},'
                       '{test.k},{test.r},{test.s},'
                       '{test.hstride},{test.wstride},'
                       '{time:.15f}\n'.format(test=test, 
                            time=time)
                        for time in times])
