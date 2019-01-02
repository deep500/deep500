import os
import numpy as np
import sys
import time
from pydoc import locate

import deep500 as d5

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('USAGE: network_inference.py <ONNX FILE> <FRAMEWORK>')
        exit(1)

    # Find framework dynamically
    fwname = sys.argv[2]
    d5fw = locate('deep500.frameworks.' + fwname)
    if d5fw is None:
        raise ValueError('Unrecognized framework ' + fwname)
        
    model = d5.parser.load_and_parse_model(sys.argv[1])
    executor = d5fw.from_model(model)

    # Because the reference operators are relatively slow, run fewer times
    if fwname == 'reference':
        metrics = [d5.WallclockTime(reruns=5, avg_over=1)]
        d5.test_executor_inference(executor, metrics=metrics)
    else:
        d5.test_executor_inference(executor)
