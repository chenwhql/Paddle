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

#include "paddle/fluid/operators/instance_norm_op.h"
#include <string>
#include "paddle/fluid/framework/data_layout.h"

namespace paddle {
namespace operators {

class InstanceNormOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("X"),
                   "Input(X) of Instance Normalization Op should not be null.");
    PADDLE_ENFORCE(
        ctx->HasInput("Scale"),
        "Input(Scale) of Instance Normalization Op should not be null.");
    PADDLE_ENFORCE(
        ctx->HasInput("Bias"),
        "Input(Bias) of Instance Normalization Op should not be null.");
    PADDLE_ENFORCE(ctx->HasOutput("Y"),
                   "Output(Y) of Instance Normalization Op should not be null");
    PADDLE_ENFORCE(ctx->HasOutput("SavedMean"), "");
    PADDLE_ENFORCE(ctx->HasOutput("SavedVariance"), "");

    const auto x_dims = ctx->GetInputDim("X");
    const DataLayout data_layout = framework::StringToDataLayout(
        ctx->Attrs().Get<std::string>("data_layout"));

    // TODO(chenweihang): x_dims.size() <= 6?
    PADDLE_ENFORCE(x_dims.size() >= 2 && x_dims.size() <= 5,
                   "Input X must have 2 to 5 dimensions.");

    const int64_t N = x_dims[0];
    const int64_t C =
        (data_layout == DataLayout::kNCHW ? x_dims[1]
                                          : x_dims[x_dims.size() - 1]);

    PADDLE_ENFORCE_EQ(ctx->GetInputDim("Scale").size(), 1UL);
    PADDLE_ENFORCE_EQ(ctx->GetInputDim("Scale")[0], C);
    PADDLE_ENFORCE_EQ(ctx->GetInputDim("Bias").size(), 1UL);
    PADDLE_ENFORCE_EQ(ctx->GetInputDim("Bias")[0], C);

    ctx->SetOutputDim("Y", x_dims);
    ctx->SetOutputDim("SavedMean", {C, N});
    ctx->SetOutputDim("SavedVariance", {C, N});
    ctx->ShareLoD("X", "Y");
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    auto input_data_type =
        framework::ToDataType(ctx.Input<Tensor>("X")->type());
    auto in_param_type = framework::proto::VarType::FP32;
    PADDLE_ENFORCE_EQ(in_param_type,
                      framework::ToDataType(ctx.Input<Tensor>("Scale")->type()),
                      "Scale input should be of float type");
    PADDLE_ENFORCE_EQ(in_param_type,
                      framework::ToDataType(ctx.Input<Tensor>("Bias")->type()),
                      "Bias input should be of float type");

    framework::LibraryType library = framework::LibraryType::kPlain;
    framework::DataLayout layout = framework::DataLayout::kAnyLayout;

    return framework::OpKernelType(input_data_type, ctx.GetPlace(), layout,
                                   library);
  }
};

class InstanceNormOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddAttr<float>("epsilon", "")
        .SetDefault(1e-5)
        .AddCustomChecker([](const float &epslion) {
          PADDLE_ENFORCE(epslion >= 0.0f && epslion <= 0x001f,
                         "'epslion' should between 0.0 and 0.001");
        });
    AddAttr<std::string>("data_layout", "").SetDefault("NCHW");
    AddInput("X", "The input tensor.");
    AddInput("Scale",
             "Scale is a 1-dimensional tensor of size C "
             "that is applied to the output");
    AddInput("Bias",
             "Bias is a 1-dimensional tensor of size C "
             "that is applied to the output");
    AddOutput("Y", "Result after normalization").Reuse("X");
    AddOutput("SavedMean",
              "Mean of the per instance of current mini "
              "batch, will apply to output when training")
        .AsIntermediate();
    AddOutput("SavedVariance",
              "Variance of the per instance of current mini "
              "batch, will apply to output when training")
        .AsIntermediate();
    AddComment(R"DOC(
Instance Normalization.

Carries out instance normalization as described in the paper:
https://arxiv.org/abs/1607.08022.
y = scale * (x - mean) / sqrt(variance + epsilon) + B, 
where mean and variance are computed per instance per channel.
    )DOC");
  }
};

template <typename T>
class InstanceNormKernel<platform::CPUDeviceContext, T>
    : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    const float epsilon = ctx.Attr<float>("epsilon");
    const std::string data_layout_str = ctx.Attr<std::string>("data_layout");
    const DataLayout data_layout =
        framework::StringToDataLayout(data_layout_str);

    const auto *x = ctx.Input<Tensor>("X");
    const auto &x_dims = x->dims();
    const int N = x_dims[0];
    const int C =
        (data_layout == DataLayout::kNCHW ? x_dims[1]
                                          : x_dims[x_dims.size() - 1]);
    const int sample_size = x->numel() / N / C;

    auto *y = ctx.Output<Tensor>("Y");
    auto *saved_mean = ctx.Output<Tensor>("SavedMean");
    auto *saved_variance = ctx.Output<Tensor>("SavedVariance");

    VLOG(3) << "compute start.";

    // Alloc memory and init
    y->mutable_data<T>(ctx.GetPlace());
    saved_mean->mutable_data<T>(ctx.GetPlace());
    saved_variance->mutable_data<T>(ctx.GetPlace());

    EigenArrayMap<T> saved_mean_e(saved_mean->mutable_data<T>(ctx.GetPlace()),
                                  C, N);
    EigenArrayMap<T> saved_variance_e(
        saved_variance->mutable_data<T>(ctx.GetPlace()), C, N);
    saved_mean_e.setZero();
    saved_variance_e.setZero();

    VLOG(3) << "alloc complete.";

    // Compute mean and variance
    switch (data_layout) {
      case DataLayout::kNCHW: {
        ConstEigenArrayMap<T> x_arr(x->data<T>(), sample_size, N * C);
        for (int n = 0; n < N; ++n) {
          for (int c = 0; c < C; ++c) {
            saved_mean_e(c, n) += x_arr.col(n * C + c).sum();
          }
        }
        saved_mean_e /= sample_size;
        VLOG(3) << "mean calc complete.";
        for (int n = 0; n < N; ++n) {
          for (int c = 0; c < C; ++c) {
            saved_variance_e(c, n) +=
                (x_arr.col(n * C + c) - saved_mean_e(c, n))
                    .matrix()
                    .squaredNorm();
          }
        }
        saved_variance_e /= sample_size;
        VLOG(3) << "HCHW, compute mean and var end.";
        break;
      }
      case DataLayout::kNHWC: {
        ConstEigenArrayMap<T> x_arr(x->data<T>(), C, N * sample_size);
        for (int n = 0; n < N; ++n) {
          for (int i = 0; i < sample_size; ++i) {
            saved_mean_e.col(n) += x_arr.col(n * sample_size + i);
          }
        }
        saved_mean_e /= sample_size;
        for (int n = 0; n < N; ++n) {
          for (int i = 0; i < sample_size; ++i) {
            saved_variance_e.col(n) +=
                (x_arr.col(n * sample_size + i) - saved_mean_e.col(n)) *
                (x_arr.col(n * sample_size + i) - saved_mean_e.col(n));
          }
        }
        saved_variance_e /= sample_size;
        VLOG(3) << "HHWC, compute mean and var end.";
        break;
      }
      default:
        PADDLE_THROW("Unknown storage order: %s", data_layout_str);
    }

    // Use SavedMean and SavedVariance to do normalize
    Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic> inv_std(C, N);
    EigenArrayMap<T> saved_inv_std(
        ctx.Output<Tensor>("SavedVariance")->data<T>(), C, N);
    for (int n = 0; n < N; ++n) {
      saved_inv_std.col(n) = (saved_inv_std.col(n) + epsilon).inverse().sqrt();
    }
    inv_std = saved_inv_std;
    ConstEigenVectorArrayMap<T> mean_arr(
        ctx.Output<Tensor>("SavedMean")->data<T>(), C, N);

    VLOG(3) << "normalization end.";

    // ((x * est_mean) * inv_var) * scale + bias
    // = (x * inv_var * scale) + (bias - est_mean * inv_var * scale)
    const auto *scale = ctx.Input<Tensor>("Scale");
    const auto *bias = ctx.Input<Tensor>("Bias");
    ConstEigenVectorArrayMap<T> scale_arr(scale->data<T>(), C);
    ConstEigenVectorArrayMap<T> bias_arr(bias->data<T>(), C);
    Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic> new_scale(C, N);
    for (int n = 0; n < N; ++n) {
      new_scale.col(n) = inv_std.col(n) * scale_arr;
    }
    Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic> new_bias(C, N);
    for (int n = 0; n < N; ++n) {
      new_bias.col(n) = bias_arr - mean_arr.col(n) * inv_std.col(n) * scale_arr;
    }

    VLOG(3) << "new scale and new bias end.";

    switch (data_layout) {
      case DataLayout::kNCHW: {
        EigenArrayMap<T> y_arr(y->mutable_data<T>(ctx.GetPlace()), sample_size,
                               N * C);
        ConstEigenArrayMap<T> x_arr(x->data<T>(), sample_size, N * C);
        for (int n = 0; n < N; ++n) {
          for (int c = 0; c < C; ++c) {
            y_arr.col(n * C + c) =
                x_arr.col(n * C + c) * new_scale(c, n) + new_bias(c, n);
          }
        }
        VLOG(3) << "NCHW, y calc end.";
        break;
      }
      case DataLayout::kNHWC: {
        EigenArrayMap<T> y_arr(y->mutable_data<T>(ctx.GetPlace()), C,
                               N * sample_size);
        ConstEigenArrayMap<T> x_arr(x->data<T>(), C, N * sample_size);
        for (int n = 0; n < N; ++n) {
          y_arr.middleCols(n * sample_size, (n + 1) * sample_size) =
              (x_arr.middleCols(n * sample_size, (n + 1) * sample_size)
                   .colwise() *
               new_scale.col(n))
                  .colwise() +
              new_bias.col(n);
        }
        VLOG(3) << "NHWC, y calc end.";
        break;
      }
      default:
        PADDLE_THROW("Unknown storage order: %d", data_layout);
    }
  }
};

class InstanceNormGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {
    // check input
    PADDLE_ENFORCE(ctx->HasInput("X"), "");
    PADDLE_ENFORCE(ctx->HasInput("Scale"), "");
    PADDLE_ENFORCE(ctx->HasInput(framework::GradVarName("Y")), "");
    PADDLE_ENFORCE(ctx->HasInput("SavedMean"), "");
    PADDLE_ENFORCE(ctx->HasInput("SavedVariance"), "");

    // check output
    PADDLE_ENFORCE(ctx->HasOutput(framework::GradVarName("X")), "");
    PADDLE_ENFORCE(ctx->HasOutput(framework::GradVarName("Scale")), "");
    PADDLE_ENFORCE(ctx->HasOutput(framework::GradVarName("Bias")), "");

    const auto x_dims = ctx->GetInputDim("X");
    const DataLayout data_layout = framework::StringToDataLayout(
        ctx->Attrs().Get<std::string>("data_layout"));
    const int C =
        (data_layout == DataLayout::kNCHW ? x_dims[1]
                                          : x_dims[x_dims.size() - 1]);

    ctx->SetOutputDim(framework::GradVarName("X"), x_dims);
    ctx->SetOutputDim(framework::GradVarName("Scale"), {C});
    ctx->SetOutputDim(framework::GradVarName("Bias"), {C});
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    const auto *var = ctx.InputVar(framework::GradVarName("Y"));
    if (var == nullptr) {
      PADDLE_THROW("can't find Y@GRAD");
    }
    const Tensor *t = nullptr;
    if (var->IsType<Tensor>()) {
      t = &var->Get<Tensor>();
    } else if (var->IsType<LoDTensor>()) {
      t = &var->Get<LoDTensor>();
    }
    if (t == nullptr) {
      PADDLE_THROW("can't find Y@GRAD");
    }

    framework::LibraryType library = framework::LibraryType::kPlain;
    framework::DataLayout layout = framework::DataLayout::kAnyLayout;

    return framework::OpKernelType(
        framework::ToDataType(ctx.Input<Tensor>("X")->type()), ctx.GetPlace(),
        layout, library);
  }
};

template <typename T>
class InstanceNormGradKernel<platform::CPUDeviceContext, T>
    : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    const auto *x = ctx.Input<Tensor>("X");
    const auto *d_y = ctx.Input<Tensor>(framework::GradVarName("Y"));
    const auto *scale = ctx.Input<Tensor>("Scale");
    const auto *saved_mean = ctx.Input<Tensor>("SavedMean");
    // SavedVariance have been reverted in forward operator
    const auto *saved_inv_variance = ctx.Input<Tensor>("SavedVariance");
    const std::string data_layout_str = ctx.Attr<std::string>("data_layout");
    const DataLayout data_layout =
        framework::StringToDataLayout(data_layout_str);

    // Get the size for each dimension.
    // NCHW [batch_size, in_channels, in_height, in_width]
    const auto &x_dims = x->dims();
    const int N = x_dims[0];
    const int C =
        (data_layout == DataLayout::kNCHW ? x_dims[1]
                                          : x_dims[x_dims.size() - 1]);
    const int sample_size = x->numel() / N / C;

    ConstEigenVectorArrayMap<T> scale_arr(scale->data<T>(), C);
    ConstEigenArrayMap<T> mean_arr(saved_mean->data<T>(), C, N);
    ConstEigenArrayMap<T> inv_var_arr(saved_inv_variance->data<T>(), C, N);

    // init output
    auto *d_x = ctx.Output<Tensor>(framework::GradVarName("X"));
    auto *d_scale = ctx.Output<Tensor>(framework::GradVarName("Scale"));
    auto *d_bias = ctx.Output<Tensor>(framework::GradVarName("Bias"));

    d_x->mutable_data<T>(ctx.GetPlace());
    d_scale->mutable_data<T>(ctx.GetPlace());
    d_bias->mutable_data<T>(ctx.GetPlace());

    // d_bias = np.sum(d_y, axis=0)
    // d_scale = np.sum((X - mean) / inv_std * dy, axis=0)
    // d_x = scale * inv_var * (d_y - np.sum(d_y, axis=0)
    //   - (X - mean) * inv_var * inv_var * np.sum(d_y * (X - mean), axis=0))

    Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic> d_bias_arr(C, N);
    Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic> d_scale_arr(C, N);

    d_bias_arr.setZero();
    d_scale_arr.setZero();

    Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic> scale_inv_var_hw(C, N);
    for (int n = 0; n < N; ++n) {
      scale_inv_var_hw.col(n) = scale_arr * inv_var_arr.col(n) / sample_size;
    }

    VLOG(3) << "scale inv var hw end.";

    switch (data_layout) {
      case DataLayout::kNCHW: {
        ConstEigenArrayMap<T> x_arr(x->data<T>(), sample_size, N * C);
        ConstEigenArrayMap<T> d_y_arr(d_y->data<T>(), sample_size, N * C);
        EigenArrayMap<T> d_x_arr(d_x->mutable_data<T>(ctx.GetPlace()),
                                 sample_size, N * C);
        d_x_arr.setZero();

        for (int n = 0; n < N; ++n) {
          for (int c = 0; c < C; ++c) {
            d_bias_arr(c, n) += d_y_arr.col(n * C + c).sum();
            d_scale_arr(c, n) += ((x_arr.col(n * C + c) - mean_arr(c)) *
                                  inv_var_arr(c, n) * d_y_arr.col(n * C + c))
                                     .sum();
          }
        }
        for (int n = 0; n < N; ++n) {
          for (int c = 0; c < C; ++c) {
            d_x_arr.col(n * C + c) +=
                scale_inv_var_hw(c, n) *
                (d_y_arr.col(n * C + c) * sample_size - d_bias_arr(c, n) -
                 (x_arr.col(n * C + c) - mean_arr(c)) * d_scale_arr(c, n) *
                     inv_var_arr(c, n));
          }
        }
        VLOG(3) << "NCHW, grad end.";
        break;
      }
      case DataLayout::kNHWC: {
        ConstEigenArrayMap<T> x_arr(x->data<T>(), C, N * sample_size);
        ConstEigenArrayMap<T> d_y_arr(d_y->data<T>(), C, N * sample_size);
        EigenArrayMap<T> d_x_arr(d_x->mutable_data<T>(ctx.GetPlace()), C,
                                 N * sample_size);
        d_x_arr.setZero();

        Eigen::Array<T, Eigen::Dynamic, 1> d_y_row_sum(N);
        Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic> x_minus_mean(
            sample_size, N);
        Eigen::Array<T, Eigen::Dynamic, 1> d_y_mul_x_minus_mean_row_sum(N);
        for (int n = 0; n < N; ++n) {
          d_y_row_sum.row(n) =
              d_y_arr.middleCols(n * sample_size, (n + 1) * sample_size)
                  .rowwise()
                  .sum();
          x_minus_mean.col(n) =
              x_arr.middleCols(n * sample_size, (n + 1) * sample_size)
                  .colwise() -
              mean_arr.col(n);
          d_y_mul_x_minus_mean_row_sum.row(n) =
              (d_y_arr.middleCols(n * sample_size, (n + 1) * sample_size) *
               x_minus_mean.col(n))
                  .rowwise()
                  .sum();
        }
        const auto inv_var_sqr = inv_var_arr * inv_var_arr;
        for (int n = 0; n < N; ++n) {
          for (int hw = 0; hw < sample_size; ++hw) {
            d_bias_arr.col(n) +=
                d_y_arr.middleCols(n * sample_size, (n + 1) * sample_size)
                    .col(hw);
            d_scale_arr.col(n) +=
                (x_arr.middleCols(n * sample_size, (n + 1) * sample_size)
                     .col(hw) -
                 mean_arr) *
                inv_var_arr.col(n) *
                d_y_arr.middleCols(n * sample_size, (n + 1) * sample_size)
                    .col(hw);
            d_x_arr.middleCols(n * sample_size, (n + 1) * sample_size)
                .col(hw) +=
                scale_inv_var_hw.col(n) *
                (d_y_arr.middleCols(n * sample_size, (n + 1) * sample_size)
                         .col(hw) *
                     sample_size -
                 d_y_row_sum(n) -
                 x_minus_mean.col(hw) * inv_var_sqr.col(n) *
                     d_y_mul_x_minus_mean_row_sum(n));
          }
        }
        VLOG(3) << "NHWC, grad end.";
        break;
      }
      default:
        PADDLE_THROW("Unknown storage order: %s", data_layout_str);
    }

    // Calc the the sum d_bias and d_scale
    EigenVectorArrayMap<T> d_bias_arr_sum(
        d_bias->mutable_data<T>(ctx.GetPlace()), C);
    EigenVectorArrayMap<T> d_scale_arr_sum(
        d_scale->mutable_data<T>(ctx.GetPlace()), C);
    d_bias_arr_sum.setZero();
    d_scale_arr_sum.setZero();

    for (int c = 0; c < C; ++c) {
      d_bias_arr_sum(c) += d_bias_arr.row(c).sum();
      d_scale_arr_sum(c) += d_scale_arr.row(c).sum();
    }

    VLOG(3) << "sum end.";
  }
};

class InstanceNormGradMaker : public framework::SingleGradOpDescMaker {
 public:
  using framework::SingleGradOpDescMaker::SingleGradOpDescMaker;

 protected:
  std::unique_ptr<framework::OpDesc> Apply() const override {
    auto *op = new framework::OpDesc();
    op->SetType("instance_norm_grad");
    op->SetInput("X", Input("X"));
    op->SetInput(framework::GradVarName("Y"), OutputGrad("Y"));

    op->SetInput("Scale", Input("Scale"));
    op->SetInput("Bias", Input("Bias"));
    op->SetInput("SavedMean", Output("SavedMean"));
    op->SetInput("SavedVariance", Output("SavedVariance"));

    op->SetAttrMap(Attrs());

    op->SetOutput(framework::GradVarName("X"), InputGrad("X"));
    op->SetOutput(framework::GradVarName("Scale"), InputGrad("Scale"));
    op->SetOutput(framework::GradVarName("Bias"), InputGrad("Bias"));

    return std::unique_ptr<framework::OpDesc>(op);
  }
};

};  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(instance_norm, ops::InstanceNormOp, ops::InstanceNormOpMaker,
                  ops::InstanceNormGradMaker);
REGISTER_OPERATOR(instance_norm_grad, ops::InstanceNormGradOp);

REGISTER_OP_CPU_KERNEL(
    instance_norm,
    ops::InstanceNormKernel<paddle::platform::CPUDeviceContext, float>,
    ops::InstanceNormKernel<paddle::platform::CPUDeviceContext, double>);
REGISTER_OP_CPU_KERNEL(
    instance_norm_grad,
    ops::InstanceNormGradKernel<paddle::platform::CPUDeviceContext, float>,
    ops::InstanceNormGradKernel<paddle::platform::CPUDeviceContext, double>);
