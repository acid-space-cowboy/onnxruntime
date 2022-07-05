// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "softmax.h"

#include "core/framework/op_kernel.h"
#include "core/providers/cpu/math/softmax_shared.h"
#include "core/providers/xnnpack/detail/utils.h"

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

    // axis could be any dim, but we want it to be the last one right now.
    // otherwise, just leave it to CPU_EP
    int64_t axis = 1;
    info.GetAttrOrDefault<int64_t>("axis", &axis, -1);
    if (node.SinceVersion() <= 12 && axis == -1) {
      axis = 1;  // default 1 for op-version less than 12
    }
    if (axis != -1 && axis != x_shape->dim_size() - 1) {
      break;
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
  // our op checker function has ensured that axis must be the last dim
  // The "semantic" meaning of axis has changed in opset-13.
  // Please compare: https://github.com/onnx/onnx/blob/master/docs/Operators.md#Softmax
  // with https://github.com/onnx/onnx/blob/master/docs/Changelog.md#Softmax-11 for detailed explanations
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
  const auto& x_shape = input_defs[0]->Shape();
  if (x_shape->dim_size() == 0) {
    return;
  }
  if (axis_ < 0) {
    axis_ = static_cast<int>(HandleNegativeAxis(axis_, x_shape->dim_size()));
  }

  int kernel_dtype = 0;
  ORT_ENFORCE(GetType(*input_defs[0], kernel_dtype));
  if (kernel_dtype == ONNX_NAMESPACE::TensorProto_DataType_FLOAT) {
    op_type_ = OpComputeType::op_compute_type_fp32;
  } else if (kernel_dtype == ONNX_NAMESPACE::TensorProto_DataType_UINT8) {
    op_type_ = OpComputeType::op_compute_type_qu8;
  }

  uint32_t channels = gsl::narrow_cast<uint32_t>(x_shape->dim(axis_).dim_value());
  xnn_status xstatus;
  struct xnn_operator* p;
  if (op_type_ == OpComputeType::op_compute_type_qu8) {
    // the order of input tensor, x,x_scale, x_zp, y_scale, y_zp
    InputTensorOrder tensor_index = {-1, 1, 2, -1, -1, -1, 3, 4, -1};
    ParseQuantParamFromInfoByOrder(info, tensor_index, quant_param_);
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

  const size_t N = X_shape.SizeToDimension(axis_);
  // const size_t D = X_shape.SizeFromDimension(axis_); // the step D is 1
  xnn_status status = xnn_status_invalid_state;
  if (op_type_ == OpComputeType::op_compute_type_qu8) {
    status = xnn_setup_softmax_nc_qu8(
        op0_.get(),
        N,
        X->template Data<uint8_t>(),
        Y->template MutableData<uint8_t>(),
        nullptr);
  } else {
    status = xnn_setup_softmax_nc_f32(
        op0_.get(),
        N,
        X->template Data<float>(),
        Y->template MutableData<float>(),
        nullptr);
  }
  ORT_ENFORCE(status == xnn_status_success, "xnn_setup_softmax_nc_type failed. Status:", status);
  ORT_ENFORCE(xnn_run_operator(op0_.get(), nullptr) == xnn_status_success, "xnn_run_operator returned ", status);
  return Status::OK();
}

ONNX_OPERATOR_VERSIONED_KERNEL_EX(Softmax, kOnnxDomain, 1, 13, kXnnpackExecutionProvider,
                                  KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
                                  Softmax);
ONNX_OPERATOR_VERSIONED_KERNEL_EX(QLinearSoftmax, kMSInternalNHWCDomain, 1, 13, kXnnpackExecutionProvider,
                                  KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<uint8_t>()),
                                  Softmax);

}  // namespace xnnpack
}  // namespace onnxruntime
