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
  t1.reshape(tensor_shape);
  t1.mutable_data<float>(paddle::PlaceType::kGPU);
  for (int64_t i = 0; i < t1.size(); i++) {
    t1.data<float>()[i] = 5;
  }
  return t1;
}

paddle::Tensor InitCPUTensorForTest() {
  std::vector<int> tensor_shape = {5, 5};
  auto t1 = paddle::Tensor(paddle::PlaceType::kCPU);
  t1.reshape(tensor_shape);
  t1.mutable_data<float>(paddle::PlaceType::kCPU);
  for (int64_t i = 0; i < t1.size(); i++) {
    t1.data<float>()[i] = 5;
  }
  return t1;
}
template <typename T>
void TestCopyToCpuFromGpuTensor() {
  auto t1 = InitGPUTensorForTest();
  auto t1_cpu_cp = t1.copy_to_cpu<T>();
  CHECK((paddle::PlaceType::kCPU == t1_cpu_cp.place()));
  for (int64_t i = 0; i < t1.size(); i++) {
    CHECK_EQ(t1_cpu_cp.template data<T>()[i], 5);
  }
}

template <typename T>
void TestCopyToGPUFromCpuTensor() {
  auto t1 = InitGPUTensorForTest();
  auto t1_gpu_cp = t1.copy_to_gpu<T>();
  CHECK((paddle::PlaceType::kGPU == t1_gpu_cp.place()));
  for (int64_t i = 0; i < t1.size(); i++) {
    CHECK_EQ(t1_gpu_cp.template data<T>()[i], 5);
  }
}

void TestAPIPlace() {
  auto t1 = paddle::Tensor(paddle::PlaceType::kGPU);
  auto t2 = paddle::Tensor(paddle::PlaceType::kCPU);
  CHECK((paddle::PlaceType::kGPU == t1.place()));
  CHECK((paddle::PlaceType::kCPU == t2.place()));
}

void TestAPISizeAndShape() {
  std::vector<int> tensor_shape = {5, 5};
  auto t1 = paddle::Tensor(paddle::PlaceType::kCPU);
  t1.reshape(tensor_shape);
  CHECK_EQ(t1.size(), 25);
  CHECK(t1.shape() == tensor_shape);
}
template <typename T>
paddle::DataType TestDtype() {
  std::vector<int> tensor_shape = {5, 5};
  auto t1 = paddle::Tensor(paddle::PlaceType::kCPU);
  t1.reshape(tensor_shape);
  t1.template mutable_data<T>();
  return t1.type();
}

void GroupTestCopy() {
  TestCopyToCpuFromGpuTensor<float>();
  TestCopyToCpuFromGpuTensor<double>();
  TestCopyToGPUFromCpuTensor<float>();
  TestCopyToGPUFromCpuTensor<double>();
}
void GroupTestDtype() {
  CHECK(TestDtype<float>() == paddle::DataType::FLOAT32);
  CHECK(TestDtype<double>() == paddle::DataType::FLOAT64);
  CHECK(TestDtype<int>() == paddle::DataType::INT32);
  CHECK(TestDtype<int64_t>() == paddle::DataType::INT64);
  CHECK(TestDtype<int16_t>() == paddle::DataType::INT16);
  CHECK(TestDtype<int8_t>() == paddle::DataType::INT8);
}

TEST(CustomTensor, copyTest) {
  GroupTestCopy();
  GroupTestDtype();
  TestAPISizeAndShape();
  TestAPIPlace();
}
