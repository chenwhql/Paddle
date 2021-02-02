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

#include "paddle/fluid/extension/include/all.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/memory/memcpy.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/platform/gpu_info.h"
namespace paddle {

#define GET_CASTED_TENSOR                               \
  if (!tensor_) {                                       \
    tensor_ = std::make_shared<framework::LoDTensor>(); \
  }                                                     \
  auto *tensor = static_cast<framework::LoDTensor *>(tensor_.get());

void CustomTensor::Reshape(const std::vector<int> &shape) {
    GET_CASTED_TENSOR
    tensor->Resize(framework::make_ddim(shape));
}

CustomTensor::CustomTensor(PaddlePlace place):
        tensor_(std::make_shared<framework::LoDTensor>()),
        place_(place){};

CustomTensor::CustomTensor(void* raw_tensor) :
        tensor_(static_cast<framework::LoDTensor*>(raw_tensor)),
        place_(PlaceType::kUNK){}

template <typename T>
T *CustomTensor::mutable_data(const PaddlePlace& place) {
    place_ = place;
    return mutable_data<T>();
}

template <typename T>
T *CustomTensor::mutable_data() {
    GET_CASTED_TENSOR
    PADDLE_ENFORCE_GT(
            tensor->numel(), 0,
            platform::errors::PreconditionNotMet(
                    "You should call ZeroCopyTensor::Reshape(const std::vector<int> "
                    "&shape)"
                    "function before retrieving mutable_data from input tensor."));
    switch (static_cast<int>(place_.GetPlace())) {
        case static_cast<int>(PlaceType::kCPU): {
            return tensor->mutable_data<T>(platform::CPUPlace());
        }
        case static_cast<int>(PlaceType::kGPU): {
#ifdef PADDLE_WITH_CUDA
            int device_num = platform::GetCurrentDeviceId();
            return tensor->mutable_data<T>(platform::CUDAPlace(device_num));
#endif
        }
        default:
            PADDLE_THROW(platform::errors::Unavailable("CustomOp unsupported place: %d",
                                                   static_cast<int>(place_.GetPlace())));
    }
}

template <typename T>
T *CustomTensor::data() const {
    GET_CASTED_TENSOR;
    auto *res = tensor->data<T>();
    return res;
}

PaddleDType CustomTensor::type() const {
    GET_CASTED_TENSOR;
    auto type = tensor->type();
    if (type == framework::proto::VarType::FP32) {
        return PaddleDType::FLOAT32;
    } else if (type == framework::proto::VarType::INT64) {
        return PaddleDType::INT64;
    } else if (type == framework::proto::VarType::INT32) {
        return PaddleDType::INT32;
    } else if (type == framework::proto::VarType::UINT8) {
        return PaddleDType::UINT8;
    }
    return PaddleDType::FLOAT32;
}

template <typename T>
void CustomTensor::copy_from_cpu(const T *data) {
    GET_CASTED_TENSOR;
    PADDLE_ENFORCE_GE(tensor->numel(), 0,
                      platform::errors::PreconditionNotMet(
                              "You should call ZeroCopyTensor::Reshape(const "
                              "std::vector<int> &shape)"
                              "function before copying data from cpu."));
    size_t ele_size = tensor->numel() * sizeof(T);

    if (place_.GetPlace() == PlaceType::kCPU) {
        auto *t_data = tensor->mutable_data<T>(platform::CPUPlace());
        std::memcpy(static_cast<void *>(t_data), data, ele_size);
    } else {
#ifdef PADDLE_WITH_CUDA
    platform::DeviceContextPool &pool = platform::DeviceContextPool::Instance();
    int device_num = platform::GetCurrentDeviceId();
    platform::CUDAPlace gpu_place(device_num);
    auto *t_data = tensor->mutable_data<T>(gpu_place);
    auto *dev_ctx =
        static_cast<const platform::CUDADeviceContext *>(pool.Get(gpu_place));

memory::Copy(gpu_place, static_cast<void *>(t_data), platform::CPUPlace(),
             data, ele_size, dev_ctx->stream());
#else
        PADDLE_THROW(platform::errors::Unavailable(
                "Not compiled with CUDA, should not reach here."));
#endif
    }
}

template <typename T>
void CustomTensor::copy_to_cpu(T *data) {
    GET_CASTED_TENSOR;
    auto ele_num = tensor->numel();
    auto *t_data = tensor->data<T>();
    auto t_place = tensor->place();

    if (platform::is_cpu_place(t_place)) {
        std::memcpy(static_cast<void *>(data), t_data, ele_num * sizeof(T));
    } else {
#ifdef PADDLE_WITH_CUDA
        platform::DeviceContextPool &pool = platform::DeviceContextPool::Instance();
auto gpu_place = BOOST_GET_CONST(platform::CUDAPlace, t_place);
auto *dev_ctx =
    static_cast<const platform::CUDADeviceContext *>(pool.Get(gpu_place));
memory::Copy(platform::CPUPlace(), static_cast<void *>(data), gpu_place,
             t_data, ele_num * sizeof(T), dev_ctx->stream());

cudaStreamSynchronize(dev_ctx->stream());
#else
        PADDLE_THROW(platform::errors::Unavailable(
                "Not compile with CUDA, should not reach here."));
#endif
    }
}

std::vector<int> CustomTensor::shape() const {
    GET_CASTED_TENSOR
    return framework::vectorize<int>(tensor->dims());
}

void CustomTensor::SetLoD(const std::vector<std::vector<size_t>> &x) {
    GET_CASTED_TENSOR;
    framework::LoD lod;
    for (auto &level : x) {
        lod.emplace_back(level);
    }
    tensor->set_lod(lod);
}

std::vector<std::vector<size_t>> CustomTensor::lod() const {
    GET_CASTED_TENSOR;
    std::vector<std::vector<size_t>> res;
    for (auto &level : tensor->lod()) {
        res.emplace_back(level);
    }
    return res;
}

const PaddlePlace& CustomTensor::place() {
    GET_CASTED_TENSOR;
    if(platform::is_cpu_place(tensor->place())){
        place_ = PaddlePlace(PlaceType::kCPU);
    }else if(platform::is_gpu_place(tensor->place())){
        place_ = PaddlePlace(PlaceType::kGPU);
    }else{
        PADDLE_THROW("Current CustomTensor hold unsupported Place Type, Please Init it"
                     "using CustomTensor::mutable_data<T>(PaddlePlace) which T is"
                     "either Place::kCPU or Place::kGPU");
    }
    return place_;
}

void CustomTensor::ShareDataWith(void* out_data){
    static_cast<framework::LoDTensor*>(out_data)
    ->ShareDataWith(
            *static_cast<framework::LoDTensor*>(tensor_.get()));
}

int64_t CustomTensor::size() const{
    GET_CASTED_TENSOR;
    return tensor->numel();
}
}  // namespace paddle

