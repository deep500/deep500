import sys
import random

import numpy as np
import deep500 as d5

from deep500.frameworks import tensorflow as d5tf

# Uses the ONNX Test Parser to run a test
if __name__ == '__main__':
    if len(sys.argv) > 2:
        print('USAGE: onnxtest.py [TESTNAME]')
        print('If TESTNAME is not specified, chooses and runs a random test')
        sys.exit(1)
        
    parser = d5.OnnxTestParser()
    if len(sys.argv) == 1:
        testname = random.choice(parser.all_tests['Node'])
        print('Chose test "%s"' % testname)
    else:
        testname = sys.argv[1]
        
    test = parser.get_test(testname)
        
    if test is None:
        print('ERROR: Test "%s" not found' % testname)
        sys.exit(2)
    
    model = d5.parser.load_and_parse_model(test.model)
    executor = d5tf.from_model(model)
    d5.test_executor_inference(executor, inputs=test.data_sets[0].inputs, 
                               reference_outputs=test.data_sets[0].outputs,
                               metrics=[d5.L2Error()])

    