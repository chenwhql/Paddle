/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

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
#include <algorithm>
#include <string>
#include <utility>
#include <vector>

#include "paddle/fluid/framework/data_type.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/math/selected_rows_functor.h"

#if defined(PADDLE_WITH_CUDA) && !defined(_WIN32)
#include "paddle/fluid/operators/distributed/collective_client.h"
#include "paddle/fluid/operators/distributed/collective_server.h"
#include "paddle/fluid/operators/distributed/request_handler.h"
#include "paddle/fluid/platform/nccl_helper.h"
#endif

namespace paddle {
namespace operators {

template <typename DeviceContext, typename T>
class AllReduceOpKernel : public framework::OpKernel<T> {
 public:
  static inline std::string GetRemoteVarName(const std::string &var_name,
                                             int trainer_id) {
    return string::Sprintf("%s_merged_tmp@trainer_%d", var_name, trainer_id);
  }

  void Compute(const framework::ExecutionContext &ctx) const override {
    auto place = ctx.GetPlace();
    PADDLE_ENFORCE(is_gpu_place(place),
                   "AllReduce op can run on gpu place only for now.");
#if defined(PADDLE_WITH_CUDA) && !defined(_WIN32)
    auto &dev_ctx = ctx.template device_context<platform::CUDADeviceContext>();
    const auto *in_var = ctx.InputVar("X");
    if (in_var->IsType<framework::LoDTensor>()) {
      auto in = ctx.Input<framework::Tensor>("X");
      auto out = ctx.Output<framework::Tensor>("Out");

      int dtype = platform::ToNCCLDataType(in->type());
      int64_t numel = in->numel();
      auto *sendbuff = in->data<void>();
      out->Resize(in->dims());
      void *recvbuff = out->mutable_data<T>(place);

      auto *comm = dev_ctx.nccl_comm();
      // FIXME(typhoonzero): should use nccl stream here.
      auto stream = dev_ctx.stream();
      PADDLE_ENFORCE_NOT_NULL(stream, "Should initialize NCCL firstly.");

      int reduce_type = ctx.Attr<int>("reduce_type");
      ncclRedOp_t red_type = ncclSum;
      switch (reduce_type) {
        case 0:
          red_type = ncclSum;
          break;
        case 1:
          red_type = ncclProd;
          break;
        case 2:
          red_type = ncclMax;
          break;
        case 3:
          red_type = ncclMin;
          break;
      }
      PADDLE_ENFORCE(platform::dynload::ncclAllReduce(
          sendbuff, recvbuff, numel, static_cast<ncclDataType_t>(dtype),
          red_type, comm, stream));
      if (ctx.Attr<bool>("sync_mode")) {
        cudaError_t e_sync = cudaStreamSynchronize(stream);
        if (e_sync != 0) {
          LOG(FATAL) << "cudaStreamSynchronize " << cudaGetErrorString(e_sync);
        }
      }
    } else if (in_var->IsType<framework::SelectedRows>()) {
      // 1. Data check
      auto *out_var = ctx.OutputVar("Out");
      PADDLE_ENFORCE_EQ(out_var->IsType<framework::SelectedRows>(), true,
                        platform::errors::InvalidArgument(
                            "Output and input variable types do not match. "
                            "Expected output variable type is SellectedRows, "
                            "but received %s.",
                            framework::ToTypeName(out_var->Type())));

      const auto &in_selected_rows = in_var->Get<framework::SelectedRows>();
      // auto *out = out_var->GetMutable<framework::SelectedRows>();

      PADDLE_ENFORCE_EQ(in_selected_rows.value().IsInitialized(), true,
                        platform::errors::InvalidArgument(
                            "The SelectedRows is not initialized."));

      // all input tensor must be all on CPU or GPU

      // only support float and double

      VLOG(0) << "all reduce get valid SelectedRows.";

      // trainer endpoints and trainer id need to be set
      auto endpoints = ctx.Attr<std::vector<std::string>>("trainer_endpoints");
      auto trainer_id = ctx.Attr<int>("trainer_id");
      std::stringstream ss;
      ss << "trainer endpoints: ";
      for (auto ep : endpoints) {
        ss << ep << ", ";
      }
      ss << "trainer id: " << trainer_id;
      VLOG(0) << ss.str();

      PADDLE_ENFORCE_EQ(
          endpoints.empty(), false,
          platform::errors::InvalidArgument("trainer endpoints is null."));

      // 2. Gather SelectedRows
      std::string out_var_name = ctx.Outputs("Out").front();
      operators::math::scatter::MergeAdd<platform::CUDADeviceContext, T>
          merge_func;
      framework::Scope *local_scope = &ctx.scope().NewScope();
      platform::DeviceContextPool &pool =
          platform::DeviceContextPool::Instance();
      auto local_dev_ctx = pool.Get(place);
      auto merged_dev_ctx =
          dynamic_cast<platform::CUDADeviceContext *>(local_dev_ctx);

      // Create temp var to save reduce result
      std::string merged_var_name = GetRemoteVarName(out_var_name, trainer_id);
      auto merged_selected_rows = local_scope->Var(merged_var_name)
                                      ->GetMutable<framework::SelectedRows>();
      merge_func(*merged_dev_ctx, in_selected_rows, merged_selected_rows);

      // Start collective server and set variable
      operators::distributed::CollectiveServer *server =
          operators::distributed::CollectiveServer::GetInstance(
              endpoints[trainer_id], endpoints.size() - 1);

      auto rpc_server = server->GetRPCServer();
      rpc_server->RegisterVar(
          merged_var_name, operators::distributed::kRequestGetMonomerVariable,
          local_scope, merged_dev_ctx);

      // Gather vars from all remote nodes
      std::vector<const framework::SelectedRows *> remote;
      operators::distributed::CollectiveClient *client =
          operators::distributed::CollectiveClient::GetInstance();

      std::vector<operators::distributed::RemoteVar> vars;
      for (size_t i = 0; i < endpoints.size(); i++) {
        if (i == (unsigned)trainer_id) continue;

        operators::distributed::RemoteVar var;
        var.trainer_id_ = i;
        var.var_name_ = GetRemoteVarName(out_var_name, i);
        var.ep_ = endpoints[i];

        vars.push_back(var);
        VLOG(0) << "Gather from:" << var.String();
      }

      PADDLE_ENFORCE(
          client->Gather(vars, &remote, *merged_dev_ctx, local_scope));
      PADDLE_ENFORCE(remote.size() == vars.size());

      // Merge selectedrows
      std::vector<const framework::SelectedRows *> all;
      all.resize(endpoints.size());
      for (auto v : vars) {
        all[v.trainer_id_] = local_scope->FindVar(v.var_name_)
                                 ->GetMutable<framework::SelectedRows>();
      }
      all[trainer_id] = merged_selected_rows;
      merge_func(*merged_dev_ctx, all, merged_selected_rows);

      rpc_server->WaitVarBarrier(merged_var_name);
      rpc_server->ClearVar(merged_var_name);

      // Clear temp var
      std::vector<std::string> tmp_vars{merged_var_name};
      for (auto v : vars) {
        tmp_vars.push_back(v.var_name_);
      }
      local_scope->EraseVars(tmp_vars);

      // 3. Broadcast SelectedRows

    } else {
      PADDLE_THROW(platform::errors::InvalidArgument(
          "Unsupported variable type for allreduce."));
    }
#else
    PADDLE_THROW("PaddlePaddle should compile with GPU.");
#endif
  }
};

}  // namespace operators
}  // namespace paddle
