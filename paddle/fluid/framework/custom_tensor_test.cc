// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "gtest/gtest.h"
#include "paddle/extension.h"
#include "paddle/fluid/framework/lod_tensor.h"

paddle::Tensor InitGPUTensorForTest() {
  std::vector<int> tensor_shape = {5, 5};
  auto t1 = paddle::Tensor(paddle::PlaceType::kGPU);
  t1.mutable_data<float>(paddle::PlaceType::kGPU);
  t1.reshape(tensor_shape);
  for (size_t i = 0; i < t1.size(); i++) {
    t1.data<float>()[i] = 5;
  }
  return t1;
}
template <typename T>
void TestCopyToCpuOfGpuTensor() {
  auto t1 = InitGPUTensorForTest();
  auto t1_cpu_cp = t1.copy_to_cpu<T>();
  CHECK_EQ(paddle::PlaceType::kCPU, t1_cpu_cp.place());
  for (int64_t i = 0; i < t1.size(); i++) {
    CHECK_EQ(t1_cpu_cp.template data<T>()[i], 5);
  }
}
void GroupTestCopy() {
  TestCopyToCpuOfGpuTensor<float>();
  TestCopyToCpuOfGpuTensor<double>();
}

TEST(CustomTensor, copyTest) { GroupTestCopy(); }
