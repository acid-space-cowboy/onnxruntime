// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "softmax.h"

#include "core/framework/op_kernel.h"
#include "core/providers/cpu/math/softmax_shared.h"
#include "core/providers/xnnpack/detail/utils.h"

#include <xnnpack.h>

namespace onnxruntime {
namespace xnnpack {

bool IsQuantSoftmaxSupported(const onnxruntime::NodeUnit& node_unit, const onnxruntime::GraphViewer& graph) {
  bool supported = false;
  do {
    xnn_datatype x_input_type, output_type;
    const auto& inputs = node_unit.Inputs();
    // only one input for softmax
    if (inputs.size() != 1) {
      break;
    }
    x_input_type = GetDtypeInXnnpack(node_unit, 0, false, graph);
    output_type = GetDtypeInXnnpack(node_unit, 0, true, graph);
    if (x_input_type != xnn_datatype_quint8 ||
        output_type != xnn_datatype_quint8) {
      break;
    }
    supported = true;
  } while (false);

  return supported;
}

bool Softmax::IsSoftmaxOnnxNodeSupported(const onnxruntime::NodeUnit& nodeunit, const onnxruntime::GraphViewer& graph) {
  bool supported = false;
  if (IsQuantizedSoftmax(GetQuantizedOpType(nodeunit)) && IsQuantSoftmaxSupported(nodeunit, graph) == false) {
    return supported;
  }
  const onnxruntime::Node& node = nodeunit.GetNode();
  // use do {} while(false) so it's easier to set a breakpoint on the return
  do {
    // SoftMax has 1 input.
    const auto& inputs = nodeunit.Inputs();
    const auto& x_arg = inputs[0].node_arg;

    const auto* x_shape = x_arg.Shape();
    // require C to be known so we can construct the xnnpack kernel prior to Compute
    if (!x_shape || !x_shape->dim(x_shape->dim_size() - 1).has_dim_value()) {
      break;
    }
    onnxruntime::ProtoHelperNodeContext nc(node);
    onnxruntime::OpNodeProtoHelper info(&nc);

    // since version 13, softmax support any dimensions to be reduced
    if (node.SinceVersion() > 12) {
      // axis could be any dim, but we want it to be the last one right now.
      // otherwise, just leave it to CPU_EP
      int64_t axis = 0;
      info.GetAttrOrDefault<int64_t>("axis", &axis, -1);
      if (axis != -1 && axis != x_shape->dim_size() - 1) {
        break;
      }
    }
    supported = true;
  } while (false);

  return supported;
}

Softmax::Softmax(const OpKernelInfo& info) : OpKernel{info} {
  const auto& node = info.node();
  auto opset = node.SinceVersion();

  int64_t axis;
  Status status = info.GetAttr<int64_t>("axis", &axis);

  if (status.IsOK()) {
    axis_ = gsl::narrow_cast<int>(axis);
  } else {
    if (opset < 13) {
      axis_ = 1;  // opset-12 and below, the default axis value is 1
    } else {
      axis_ = -1;  // opset-13, the default axis value is -1
    }
  }
  // we have check it in GetCapability
  auto input_defs = node.InputDefs();
  int kernel_dtype = 0;
  ORT_ENFORCE(GetType(*input_defs[0], kernel_dtype));
  if (kernel_dtype == ONNX_NAMESPACE::TensorProto_DataType_FLOAT) {
    op_type_ = OpComputeType::op_compute_type_fp32;
  } else if (kernel_dtype == ONNX_NAMESPACE::TensorProto_DataType_UINT8) {
    op_type_ = OpComputeType::op_compute_type_qu8;
  }
  const auto& x_shape = input_defs[0]->Shape();
  if (axis_ == -1) {
    axis_ = x_shape->dim_size() - 1;
  }
  uint32_t channels = gsl::narrow_cast<uint32_t>(x_shape->dim(axis_).dim_value());
  xnn_status xstatus;
  struct xnn_operator* p;
  if (op_type_ == OpComputeType::op_compute_type_qu8) {
    InputTensorOrder tensor_index = {-1, 1, 2, -1, -1, -1, 3, 4, -1};
    ParseQuantParamFromInfoByOrder(info, tensor_index, quant_param_);

    /*
    const Tensor* X_zero_point = nullptr;
    const Tensor* Y_zero_point = nullptr;
    const Tensor* X_scale = nullptr;
    const Tensor* Y_scale = nullptr;
    info.TryGetConstantInput(InputTensors::IN_X_SCALE, &X_scale);
    info.TryGetConstantInput(InputTensors::IN_X_ZERO_POINT, &X_zero_point);
    info.TryGetConstantInput(InputTensors::IN_Y_SCALE, &Y_scale);
    info.TryGetConstantInput(InputTensors::IN_Y_ZERO_POINT, &Y_zero_point);

    quant_param_.X_zero_point_value = *(X_zero_point->template Data<uint8_t>());
    quant_param_.X_scale_value = *(X_scale->template Data<float>());
    quant_param_.Y_zero_point_value = *(Y_zero_point->template Data<uint8_t>());
    quant_param_.Y_scale_value = *(Y_scale->template Data<float>());
    */

    /*
    IsScalarOr1ElementVector(X_scale);
    X_zero_point == nullptr || IsScalarOr1ElementVector(X_zero_point);
    IsScalarOr1ElementVector(Y_scale);
    Y_zero_point == nullptr || IsScalarOr1ElementVector(Y_zero_point);
    */
    xstatus = xnn_create_softmax_nc_qu8(
        channels,
        channels,
        channels,
        quant_param_.X_scale_value,
        gsl::narrow_cast<uint8_t>(quant_param_.Y_zero_point_value),
        quant_param_.Y_scale_value,
        0,  // flags,
        &p);
  } else if (op_type_ == OpComputeType::op_compute_type_fp32) {
    xstatus = xnn_create_softmax_nc_f32(
        channels,
        channels,
        channels,
        0,  // flags,
        &p);
  } else {
    ORT_ENFORCE(0, "error kernel type input, expected uint8|float");
  }
  ORT_ENFORCE(xstatus == xnn_status_success, "xnn_create_softmax_nc_f32 failed. Status:", xstatus);
  op0_.reset(p);
}

// compute method of Softmax
Status Softmax::Compute(OpKernelContext* ctx) const {
  const auto* X = ctx->Input<Tensor>(0);
  const auto& X_shape = X->Shape();
  auto* Y = ctx->Output(0, X_shape);

  // edge case. one or more dims with value of 0. nothing to do
  if (X_shape.Size() == 0) {
    return Status::OK();
  }

  concurrency::ThreadPool* thread_pool = ctx->GetOperatorThreadPool();
  return ComputeImpl(*X, *Y, axis_, thread_pool);
}

Status Softmax::ComputeImpl(const Tensor& input, Tensor& output, size_t axis,
                            concurrency::ThreadPool* /*thread_pool*/) const {
  const auto& X_shape = input.Shape();
  const size_t N = X_shape.SizeToDimension(axis);
  xnn_status status = xnn_status_invalid_state;
  if (op_type_ == OpComputeType::op_compute_type_qu8) {
    status = xnn_setup_softmax_nc_qu8(
        op0_.get(),
        N,
        input.template Data<uint8_t>(),
        output.template MutableData<uint8_t>(),
        nullptr);
  } else {
    status = xnn_setup_softmax_nc_f32(
        op0_.get(),
        N,
        input.template Data<float>(),
        output.template MutableData<float>(),
        nullptr);
  }
  ORT_ENFORCE(status == xnn_status_success, "xnn_setup_softmax_nc_type failed. Status:", status);
  status = xnn_run_operator(op0_.get(), nullptr);
  if (status != xnn_status_success) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "xnn_run_operator returned ", status);
  }
  return Status::OK();
}

ONNX_OPERATOR_VERSIONED_KERNEL_EX(Softmax, kOnnxDomain, 1, 12, kXnnpackExecutionProvider,
                                  KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
                                  Softmax);
ONNX_OPERATOR_VERSIONED_KERNEL_EX(QLinearSoftmax, kMSDomain, 1, 12, kXnnpackExecutionProvider,
                                  KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<uint8_t>()),
                                  Softmax);

}  // namespace xnnpack
}  // namespace onnxruntime
