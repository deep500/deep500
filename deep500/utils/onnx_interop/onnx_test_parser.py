from collections import namedtuple
import glob
import os
from typing import Dict, List, Tuple

import numpy as np
import onnx
import onnx.backend.test.runner as onnx_runner
import onnx.backend.test.loader as test_loader
from onnx import numpy_helper

OnnxTestData = namedtuple('OnnxTestData', ['inputs', 'outputs'])
OnnxTest = namedtuple('OnnxTest', ['model', 'data_sets'])

class OnnxTestParser:

    def __init__(self):
        self.tests = {}
        for rt in test_loader.load_model_tests(kind='node'):
            self._add_model_test(rt, 'Node')

        for rt in test_loader.load_model_tests(kind='real'):
            self._add_model_test(rt, 'Real')

        for rt in test_loader.load_model_tests(kind='simple'):
            self._add_model_test(rt, 'Simple')

        for ct in test_loader.load_model_tests(kind='pytorch-converted'):
            self._add_model_test(ct, 'PyTorchConverted')

        for ot in test_loader.load_model_tests(kind='pytorch-operator'):
            self._add_model_test(ot, 'PyTorchOperator')

    def _add_model_test(self, test, test_type: str):
        if test_type not in self.tests:
            self.tests[test_type] = list()
        self.tests[test_type].append(test)

    def get_test(self, name: str) -> (str, List[Tuple]):
        """
        @param name test name to load
        @return onnx_model file path, list that holds tuples of (input, expected output)
        """
        for (k, v) in self.tests.items():
            for test in v:
                if test.name == name:
                    return self._load_input_output_of_test(test)
        return None

    @property
    def all_tests(self) -> Dict[str, List[str]]:
        """
        Returns all available ONNX tests.
        @return A dictionary that maps from test type to a list of test names
        """
        return {k: [t.name for t in v] for k, v in self.tests.items()}

    def _load_input_output_of_test(self, test) -> OnnxTest:
        model_dir = test.model_dir
        if model_dir is None:  # download test if not already there
            model_dir = onnx_runner._prepare_model_data(test)
        onnx_model_file = os.path.join(model_dir, 'model.onnx')

        #data = ([], []) # type: Tuple[List, List]
        data_sets = [] # type: List[OnnxTestData]
        
        # Older ONNX test format
        self._load_numpy_data(model_dir, data_sets) 
        
        self._load_protobuf_data(model_dir, data_sets)
        
        return OnnxTest(onnx_model_file, data_sets)

    def _load_numpy_data(self, model_dir, 
                         data_sets: List[OnnxTestData]):
        for test_data_npz in glob.glob(os.path.join(model_dir, 'test_data_*.npz')):
            test_data = np.load(test_data_npz, encoding='bytes')
            ref_inputs = list(test_data['inputs'])
            ref_outputs = list(test_data['outputs'])          
            inputs = {str(i): v for i,v in enumerate(ref_inputs)}
            outputs = {str(i): v for i,v in enumerate(ref_outputs)}
            data_sets.append(OnnxTestData(inputs, outputs))

    def _load_protobuf_data(self, model_dir,
                            data_sets: List[OnnxTestData]):
        for test_data_dir in glob.glob(os.path.join(model_dir, "test_data_set*")):
            inputs = {}
            outputs = {}
            
            inputs_num = len(glob.glob(os.path.join(test_data_dir, 'input_*.pb')))
            for i in range(inputs_num):
                input_file = os.path.join(test_data_dir, 'input_{}.pb'.format(i))
                tensor = onnx.TensorProto()
                with open(input_file, 'rb') as f:
                    tensor.ParseFromString(f.read())
                inputs[tensor.name] = numpy_helper.to_array(tensor)
            ref_outputs_num = len(glob.glob(os.path.join(test_data_dir, 'output_*.pb')))
            for i in range(ref_outputs_num):
                output_file = os.path.join(test_data_dir, 'output_{}.pb'.format(i))
                tensor = onnx.TensorProto()
                with open(output_file, 'rb') as f:
                    tensor.ParseFromString(f.read())
                outputs[tensor.name] = numpy_helper.to_array(tensor)

            data_sets.append(OnnxTestData(inputs, outputs))

                
if __name__ == '__main__':
    print(OnnxTestParser().get_test('test_reduce_log_sum_desc_axes'))
