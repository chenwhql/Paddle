/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#ifdef PADDLE_WITH_CUDA

#include "paddle/pten/core/base_tensor.h"
#include "paddle/pten/module/sign.h"

// See Note [ Why still include the fluid headers? ]
#include "paddle/fluid/platform/device_context.h"

namespace pt {

using CUDADeviceContext = paddle::platform::CUDADeviceContext;

template <typename T>
void Sign(const CUDADeviceContext& dev_ctx,
          const BaseTensor& x,
          BaseTensor* out) {
  module::Sign<CUDADeviceContext, T>(dev_ctx, x, out);
}

// TODO(chenweihang): Perhaps the Kernel call should not be implemented by
// calling functions, but by finding the Kernel call method from the global
// KernelMap. For a kernel like cuda, if you have to call functions through
// include header files, there will be many more function declarations and
// redundant function call
template <typename T>
void MeanCUDA(const CUDADeviceContext& dev_ctx,
              const BaseTensor& x,
              BaseTensor* out);

template <typename T>
void Mean(const CUDADeviceContext& dev_ctx,
          const BaseTensor& x,
          BaseTensor* out) {
  MeanCUDA<T>(dev_ctx, x, out);
}

}  // namespace pt

#endif
