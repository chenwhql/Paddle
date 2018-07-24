# Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest
import numpy as np
from op_test import OpTest


def _instance_norm(x, scale, bias, epsilon, data_format):
    if len(x.shape) == 2:
        if data_format == "NCHW":
            x = np.reshape(x, (x.shape[0], x.shape[1], 1, 1))
        elif data_format == "NHWC":
            x = np.reshape(x, (x.shape[0], 1, 1, x.shape[1]))
        else:
            raise ValueError("Unknown data order.")

    dims_x = len(x.shape)
    if data_format == "NCHW":
        axis = tuple(range(2, dims_x))
    elif data_format == "NHWC":
        axis = tuple(range(1, dims_x - 1))
    else:
        raise ValueError("Unknown data order.")

    mean = np.mean(x, axis=axis, keepdims=True)
    var = np.var(x, axis=axis, keepdims=True)
    dims_ones = (1, ) * (dims_x - 2)
    scale = scale.reshape(-1, *dims_ones)
    bias = bias.reshape(-1, *dims_ones)
    y = scale * (x - mean) / np.sqrt(var + epsilon) + bias
    return y, np.squeeze(mean, axis=axis), np.squeeze(var, axis=axis)


# Base Test 1
class TestInstanceNormOp(OpTest):
    def setUp(self):
        self.op_type = "instance_norm"
        self.init_test_case()
        x = np.random.random(self.shape).astype(np.float32)
        if self.data_format == "NCHW":
            c = x.shape[1]
        elif self.data_format == "NHWC":
            c = x.shape[3]
        else:
            raise ValueError("Unknown data order.")
        scale = np.random.randn(c).astype(np.float32)
        bias = np.random.randn(c).astype(np.float32)
        y, mean, var = _instance_norm(x, scale, bias, self.epsilon,
                                      self.data_format)
        print mean
        self.inputs = {'X': x, 'Scale': scale, 'Bias': bias}
        self.attrs = {'epsilon': self.epsilon, 'data_format': self.data_format}
        self.outputs = {'Y': y, 'SavedMean': mean, 'SavedVariance': var}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['X', 'Scale', 'Bias'], 'Y')

    def init_test_case(self):
        self.shape = [1, 2, 1, 3]
        self.epsilon = 1e-8
        self.data_format = "NCHW"


"""
# Base Test 2
class TestInstanceNormOp2(TestInstanceNormOp):
    def init_test_case(self):
        self.shape = [2, 3, 4, 5]
        self.epsilon = 1e-8
        self.data_format = "NCHW"


# Data Format Test 1
class TestInstanceNormOp3(TestInstanceNormOp):
    def init_test_case(self):
        self.shape = [2, 3, 4, 5]
        self.epsilon = 1e-8
        self.data_format = "NHWC"


# Input Shape Test 
class TestInstanceNormOp5(TestInstanceNormOp):
    def init_test_case(self):
        self.shape = [2, 3]
        self.epsilon = 1e-8
        self.data_format = "NCHW"
"""

if __name__ == '__main__':
    unittest.main()
