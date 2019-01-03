import time
import torch
import numpy as np
import deepbench

AVG_OVER = 100
RUNS = 30

print('############ Matrix Multiplication ############')
print('Vanilla Pytorch')

dev = torch.device('cuda')

# Native Pytorch results
with open('0_pt_gemm_deepbench.log', 'w') as fp:
    fp.write('m,n,k,a_trans,b_trans,time\n')

for test in deepbench.gemm_training_set:
    print(test)
    try:
        if test.a_trans:
            var_A = torch.tensor(np.random.rand(test.k, test.m), device=dev, dtype=torch.float32)
        else:
            var_A = torch.tensor(np.random.rand(test.m, test.k), device=dev, dtype=torch.float32)
        if test.b_trans:
            var_B = torch.tensor(np.random.rand(test.n, test.k), device=dev, dtype=torch.float32)
        else:
            var_B = torch.tensor(np.random.rand(test.k, test.n), device=dev, dtype=torch.float32)
        
        # Create Pytorch "model"
        def model(A, B):
            if test.a_trans:
                Amul = A.transpose(0, 1)
            else:
                Amul = A
            if test.b_trans:
                Bmul = B.transpose(0, 1)
            else:
                Bmul = B
            return torch.matmul(Amul, Bmul)
   

        times = []

        # Warmup run
        model(var_A, var_B)

        for i in range(RUNS):
            torch.cuda.synchronize()
            s = time.time()
            for j in range(AVG_OVER):
                model(var_A, var_B)
            torch.cuda.synchronize()
            e = time.time()
            times.append((e - s) / AVG_OVER)
    except Exception as ex:
        print('Exception:', ex)
        times = [-1.0]

    with open('0_pt_gemm_deepbench.log', 'a') as fp:
        fp.writelines(['{test.m},{test.n},{test.k},{test.a_trans},{test.b_trans},{time:.15f}\n'.format(test=test, time=time) for time in times])

##############################################################################


print('Deep500 Pytorch - test_nativeop_*')
# Deep500 Pytorch

import deep500 as d5
from deep500.frameworks import pytorch as d5pt

with open('0_d5pt_gemm_deepbench.log', 'w') as fp:
    fp.write('m,n,k,a_trans,b_trans,time,l2_error,max_error\n')

for test in deepbench.gemm_training_set:
    print(test)
    try:
        # Create Pytorch "model"
        if test.a_trans:
            A = np.random.rand(test.k, test.m).astype(np.float32)
            var_A = torch.tensor(A, device=dev, dtype=torch.float32)
            Amult = A.transpose()
        else:
            A = np.random.rand(test.m, test.k).astype(np.float32)
            var_A = torch.tensor(A, device=dev, dtype=torch.float32)
            Amult = A
        if test.b_trans:
            B = np.random.rand(test.n, test.k).astype(np.float32)
            var_B = torch.tensor(B, device=dev, dtype=torch.float32)
            Bmult = B.transpose()
        else:
            B = np.random.rand(test.k, test.n).astype(np.float32)
            var_B = torch.tensor(B, device=dev, dtype=torch.float32)
            Bmult = B

        # Create Pytorch "model" that only receives two variables
        def model(A, B):
            if test.a_trans:
                Amul = A.transpose(0, 1)
            else:
                Amul = A
            if test.b_trans:
                Bmul = B.transpose(0, 1)
            else:
                Bmul = B
            return torch.matmul(Amul, Bmul)
   

        # Run Deep500 test
        l2err, maxerr, times = \
            d5pt.test_nativeop_forward(model, [var_A, var_B], [Amult @ Bmult],
                                       metrics=[d5.L2Error(), d5.MaxError(), 
                                                d5.WallclockTime(RUNS*AVG_OVER, AVG_OVER)])
    except Exception as ex:
        print('Exception:', ex)
        l2err = -1.0
        maxerr = -1.0
        times = [-1.0]

    with open('0_d5pt_gemm_deepbench.log', 'a') as fp:
        fp.writelines(['{test.m},{test.n},{test.k},'
                       '{test.a_trans},{test.b_trans},'
                       '{time:.15f},{l2err},{maxerr}\n'.format(test=test, 
                            time=time, l2err=l2err, maxerr=maxerr) 
                        for time in times])
