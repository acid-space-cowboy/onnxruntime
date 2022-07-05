// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "utils.h"
#include <unordered_map>

#include "core/framework/tensorprotoutils.h"
#include "core/graph/indexed_sub_graph.h"
#include "core/graph/node_attr_utils.h"

#include "core/providers/shared/node_unit/node_unit.h"
#include "onnx/defs/attr_proto_util.h"
#include "core/common/safeint.h"
namespace onnxruntime {
namespace xnnpack {

bool GetType(const NodeArg& node_arg, int32_t& type) {
  type = ONNX_NAMESPACE::TensorProto_DataType_UNDEFINED;
  const auto* type_proto = node_arg.TypeAsProto();
  if (!type_proto || !type_proto->has_tensor_type() || !type_proto->tensor_type().has_elem_type()) {
    LOGS_DEFAULT(WARNING) << "NodeArg [" << node_arg.Name() << "] has no input type";
    return false;
  }

  type = type_proto->tensor_type().elem_type();
  return true;
}

bool GetShape(const NodeArg& node_arg, Shape& shape) {
  shape.clear();
  const auto* shape_proto = node_arg.Shape();

  if (!shape_proto) {
    LOGS_DEFAULT(WARNING) << "NodeArg [" << node_arg.Name() << "] has no shape info";
    return false;
  }

  // uses 0 for dynamic dimension, which is the default value for dim.dim_value()
  for (const auto& dim : shape_proto->dim())
    shape.push_back(SafeInt<uint32_t>(dim.dim_value()));

  return true;
}

QuantizedOpType GetQuantizedOpType(const NodeUnit& node_unit) {
  const auto& op_type = node_unit.OpType();
  if (node_unit.UnitType() == NodeUnit::Type::QDQGroup) {
    if (op_type == "Conv")
      return QuantizedOpType::QDQConv;
    else if (op_type == "MaxPool")
      return QuantizedOpType::QDQMaxPool;
    else if (op_type == "AveragePool")
      return QuantizedOpType::QDQAvgPool;
    else if (op_type == "Softmax")
      return QuantizedOpType::QDQSoftmax;
  } else if (node_unit.OpType() == "QLinearConv") {
    return QuantizedOpType::QLinearConv;
  }
  return QuantizedOpType::Unknown;
}

bool IsPaddingTypeSupported(AutoPadType auto_pad) {
  return auto_pad == AutoPadType::NOTSET ||
         auto_pad == AutoPadType::VALID ||
         auto_pad == AutoPadType::SAME_UPPER;
}

typedef std::string ONNXOpType;

static std::unordered_map<QuantizedOpType, ONNXOpType> qdq_to_onnx_type_map = {
    {QuantizedOpType::QDQConv, "QLinearConv"},
    {QuantizedOpType::QDQAvgPool, "QLinearAveragePool"},
    {QuantizedOpType::QDQSoftmax, "QLinearSoftmax"},
    {QuantizedOpType::QDQMaxPool, "QLinearMaxPool"},
};

std::unique_ptr<IndexedSubGraph::MetaDef> FuseQDQGroup(const NodeUnit& unit_node) {
  QuantizedOpType qtype = GetQuantizedOpType(unit_node);
  // create a ComputeCapability for QDQ node.
  std::unique_ptr<IndexedSubGraph::MetaDef> metadef = std::make_unique<IndexedSubGraph::MetaDef>();
  IndexedSubGraph::MetaDef& def = *metadef;
  // It shouldn't happen if this unit_node passed the check function for each op
  if (qdq_to_onnx_type_map.count(qtype) == 0) {
    return {};
  }
  // inputs
  const auto& inputs = unit_node.Inputs();
  def.name = qdq_to_onnx_type_map[qtype];
  if (qtype == QuantizedOpType::QDQConv) {
    // registration
    def.domain = kMSInternalNHWCDomain;  // should always be kMSInternalNHWCDomain
    def.since_version = unit_node.GetNode().SinceVersion();
    def.inputs.reserve(9);

    // x x-scale x-zp w w-scale w-zp
    std::for_each(inputs.cbegin(), inputs.cbegin() + 2,
                  [&def](const NodeUnitIODef& arg) {
                    // keep the number of inputs the same by inserting an empty string for a missing optional input
                    def.inputs.push_back(arg.node_arg.Name());
                    const auto& quant_param = arg.quant_param.value();
                    def.inputs.push_back(quant_param.scale.Name());
                    def.inputs.push_back(quant_param.zero_point ? quant_param.zero_point->Name() : "");
                  });
    // y-scale y-zeropoint
    const auto& y_quant_param = unit_node.Outputs()[0].quant_param.value();
    def.inputs.push_back(y_quant_param.scale.Name());
    def.inputs.push_back(y_quant_param.zero_point ? y_quant_param.zero_point->Name() : "");
    // bias
    if (inputs.size() > 2) {
      def.inputs.push_back(inputs[2].node_arg.Name());
    }
  } else if (qtype == QuantizedOpType::QDQAvgPool || qtype == QuantizedOpType::QDQSoftmax) {
    // registration
    def.domain = kMSInternalNHWCDomain;
    def.since_version = unit_node.GetNode().SinceVersion();
    // x xsc xzp ysc yzp
    def.inputs.reserve(5);
    // x x-scale x-zp
    std::for_each(inputs.cbegin(), inputs.cend(),
                  [&def](const NodeUnitIODef& arg) {
                    // keep the number of inputs the same by inserting an empty string for a missing optional input
                    def.inputs.push_back(arg.node_arg.Name());
                    const auto& quant_param = arg.quant_param.value();
                    def.inputs.push_back(quant_param.scale.Name());
                    def.inputs.push_back(quant_param.zero_point ? quant_param.zero_point->Name() : "");
                  });
    // y-scale y-zeropoint
    const auto& y_quant_param = unit_node.Outputs()[0].quant_param.value();
    def.inputs.push_back(y_quant_param.scale.Name());
    def.inputs.push_back(y_quant_param.zero_point ? y_quant_param.zero_point->Name() : "");
  } else if (qtype == QuantizedOpType::QDQMaxPool) {
    // QDQMaxPool, do nothing, QDQMaxPool doesn't require dq node or q node.
  } else {
    // all qdq-types are enumerated
  }
  // outputs
  for (const auto& out : unit_node.Outputs()) {
    def.outputs.push_back(out.node_arg.Name());
  }

  // attributes
  // copy existing and add the activation info
  def.attributes.insert(unit_node.GetNode().GetAttributes().begin(), unit_node.GetNode().GetAttributes().end());
  return metadef;
}

// Fuse activation with node. Currently Conv and MaxPool are supported.
std::unique_ptr<IndexedSubGraph::MetaDef> FuseActivation(const Node& node, const Node& activation,
                                                         const GraphViewer& graph) {
  std::unique_ptr<IndexedSubGraph::MetaDef> metadef = std::make_unique<IndexedSubGraph::MetaDef>();
  IndexedSubGraph::MetaDef& def = *metadef;

  // we use the op type/domain to match the static xnnpack Conv or MaxPool kernel
  // registration
  def.name = node.OpType();
  def.domain = node.Domain();  // should always be kMSInternalNHWCDomain
  def.since_version = node.SinceVersion();

  // inputs
  const auto& inputs = node.InputDefs();
  def.inputs.reserve(inputs.size());
  std::for_each(inputs.cbegin(), inputs.cend(),
                [&def](const NodeArg* arg) {
                  // keep the number of inputs the same by inserting an empty string for a missing optional input
                  def.inputs.push_back(arg ? arg->Name() : "");
                });

  // outputs
  def.outputs.push_back(activation.OutputDefs()[0]->Name());

  // attributes
  // copy existing and add the activation info
  def.attributes = node.GetAttributes();

  // use infinity as the default as that's what xnnpack uses if min/max are not set
  float min = -INFINITY;
  float max = INFINITY;

  const auto& activation_type = activation.OpType();
  if (activation_type == "Clip") {
    min = std::numeric_limits<float>::min();
    max = std::numeric_limits<float>::max();
    bool min_max_are_attributes = activation.SinceVersion() == 1 || activation.SinceVersion() == 6;

    if (min_max_are_attributes) {
      ProtoHelperNodeContext nc(activation);
      OpNodeProtoHelper info(&nc);
      min = info.GetAttrOrDefault<float>("min", min);
      max = info.GetAttrOrDefault<float>("max", max);
    } else {
      const auto& clip_inputs = activation.InputDefs();
      const auto num_inputs = clip_inputs.size();

      const auto update_value = [&](size_t idx, float& value_to_set) {
        if (num_inputs > idx) {
          const NodeArg& arg = *clip_inputs[idx];
          if (arg.Exists()) {
            const auto& value = *graph.GetConstantInitializer(arg.Name(), true);
            // these should never be in external data as it makes no sense to put scalars there.
            ORT_ENFORCE(utils::HasExternalData(value) == false,
                        "External data is not supported for the scalar min/max Clip values");

            value_to_set = utils::HasRawData(value)
                               ? *reinterpret_cast<const float*>(value.raw_data().data())
                               : value.float_data()[0];
          }
        }
      };

      update_value(1, min);
      update_value(2, max);
    }
  } else if (activation_type == "Relu") {
    min = 0.f;
  } else {
    ORT_NOT_IMPLEMENTED("No support for fusion of ", node.OpType(), " with ", activation_type);
  }

  InlinedVector<float> activation_params{min, max};
  def.attributes.insert({"activation", utils::MakeAttribute("activation", activation_type)});
  def.attributes.insert({"activation_params", utils::MakeAttribute("activation_params", activation_params)});

  return metadef;
}

bool IsQuantizedConv(QuantizedOpType quant_op_type) {
  return (quant_op_type == QuantizedOpType::QLinearConv) ||
         (quant_op_type == QuantizedOpType::QDQConv);
}

bool IsQuantizedMaxPool(QuantizedOpType quant_op_type) {
  return (quant_op_type == QuantizedOpType::QLinearMaxPool) ||
         (quant_op_type == QuantizedOpType::QDQMaxPool);
}

bool IsQuantizedAvgPool(QuantizedOpType quant_op_type) {
  return (quant_op_type == QuantizedOpType::QlinearAvgPool) ||
         (quant_op_type == QuantizedOpType::QDQAvgPool);
}

bool IsQuantizedSoftmax(QuantizedOpType quant_op_type) {
  return (quant_op_type == QuantizedOpType::QDQSoftmax);
}

const onnx::TensorProto* GetQuantizationScale(const InitializedTensorSet& initializers,
                                              const NodeUnitIODef& io_def) {
  if (io_def.quant_param.has_value() == false) {
    return nullptr;
  }
  onnx::TensorProto tensor_proto_ret;
  const auto scale_name = io_def.quant_param->scale.Name();
  auto it = initializers.find(scale_name);
  if (it == initializers.cend()) {
    return nullptr;
  }
  return it->second;
}

const onnx::TensorProto* GetQuantizationZeroPoint(const InitializedTensorSet& initializers,
                                                  const NodeUnitIODef& io_def) {
  if (!io_def.quant_param.has_value() || !io_def.quant_param->zero_point)
    return nullptr;

  const auto& zero_point_name = io_def.quant_param->zero_point->Name();
  if (!Contains(initializers, zero_point_name)) {
    return nullptr;
  }

  return initializers.at(zero_point_name);
}

// XNNPACK defined a few dtypes for quantized tensor, hence we can easily check if XNNPACK support it
xnn_datatype GetDtypeInXnnpack(const onnxruntime::NodeUnit& node_unit, int32_t io_index,
                               bool is_output, const onnxruntime::GraphViewer& graph_viewer) {
  // we do not check the legality of io_index here
  const NodeUnitIODef& iodef = is_output ? node_unit.Outputs()[io_index] : node_unit.Inputs()[io_index];
  xnn_datatype datatype = xnn_datatype_invalid;
  int32_t input_type = 0;
  if (!GetType(iodef.node_arg, input_type) || iodef.quant_param.has_value() == false) {
    return datatype;
  }

  const InitializedTensorSet& initializers = graph_viewer.GetAllInitializedTensors();
  auto* zero_tensor = GetQuantizationZeroPoint(initializers, iodef);
  auto* scale_tensor = GetQuantizationScale(initializers, iodef);
  int64_t scales_dim = !scale_tensor ? 0 : (scale_tensor->dims().empty() ? 1 : scale_tensor->dims()[0]);
  int64_t zero_dim = !zero_tensor ? 0 : (zero_tensor->dims().empty() ? 1 : zero_tensor->dims()[0]);
  const auto& quantization_params = iodef.quant_param.value();
  Shape tensor_shape;
  if (!GetShape(iodef.node_arg, tensor_shape)) {
    return datatype;
  }

  std::vector<uint8_t> unpacked_tensor;
  // we have process float-type in the beginning
  // we do not handle u8s8
  switch (input_type) {
    case ONNX_NAMESPACE::TensorProto_DataType_UINT8:
      if (quantization_params.zero_point == nullptr) {
        LOGS_DEFAULT(VERBOSE) << "missing zero point quantization parameters for "
                                 "UINT8 tensor";
        break;
      }
      if (scales_dim != 1 || zero_dim != 1) {
        LOGS_DEFAULT(VERBOSE) << "unsupported number " << scales_dim
                              << " of scale quantization parameters for UINT8 tensor"
                                 "per-channel uint8 quantization isn't supported";
        break;
      }
      datatype = xnn_datatype_quint8;
      break;
    case ONNX_NAMESPACE::TensorProto_DataType_INT8:
      // symmetry conv? when zero_dim == 0
      if (scales_dim != zero_dim && zero_dim != 0) {
        LOGS_DEFAULT(VERBOSE) << "mismatching number of scale " << scales_dim
                              << " and zero-point " << zero_dim << " quantization parameters for INT8";
        break;
      }

      if (scales_dim == 1) {
        datatype = xnn_datatype_qint8;
        // layout keeps NCHW, check channel dim
      } else if (scales_dim == tensor_shape[1]) {
        // default 0 for zero-point if zero_dim == 0
        if (zero_tensor != nullptr) {
          auto status = onnxruntime::utils::UnpackInitializerData(*zero_tensor, node_unit.ModelPath(), unpacked_tensor);
          if (!status.IsOK()) {
            LOGS_DEFAULT(ERROR) << "error when unpack zero tensor: "
                                << ", error msg: " << status.ErrorMessage();
            break;
          }
          const int8_t* zero_points = reinterpret_cast<const int8_t*>(unpacked_tensor.data());
          for (size_t i = 0; i < unpacked_tensor.size(); i++) {
            if (zero_points[i] != 0) {
              LOGS_DEFAULT(VERBOSE) << "only support 0 as zero point, "
                                    << "zero_points[" << i << "] has value: " << zero_points[i];
              break;
            }
          }
        }
        datatype = xnn_datatype_qcint8;
      } else {
        LOGS_DEFAULT(VERBOSE) << "mismatching number of quantization parameters  " << scales_dim
                              << " and outer dimension " << tensor_shape[1];
      }
      break;
      // TODO(Jicwen)
    /* case ONNX_NAMESPACE::TensorProto_DataType_FLOAT16:
      break;
      */
    default:
      break;
  }
  return datatype;
}

bool ParseQuantParamFromInfoByOrder(const OpKernelInfo& info,
                                    const InputTensorOrder& scale_zp_indexs,
                                    QuantParam& quant_param_) {
  // quant param, which used in create xnnpack_conv_kernel
  // we do not check the error here, as we have done it in op_checker
  // if this input tensor is not exists, its value is -1;
  if (scale_zp_indexs.X_ZERO_POINT >= 0) {
    const onnxruntime::Tensor* X_zero_point = nullptr;
    info.TryGetConstantInput(scale_zp_indexs.X_ZERO_POINT, &X_zero_point);
    if (X_zero_point == nullptr) {
      quant_param_.X_zero_point_value = 0;
    } else {
      quant_param_.X_zero_point_value = *(X_zero_point->template Data<uint8_t>());
    }
  }
  if (scale_zp_indexs.W_ZERO_POINT >= 0) {
    const onnxruntime::Tensor* W_zero_point = nullptr;
    info.TryGetConstantInput(scale_zp_indexs.W_ZERO_POINT, &W_zero_point);
    if (W_zero_point == nullptr) {
      quant_param_.W_zero_point_value = 0;
    } else {
      quant_param_.W_zero_point_value = *(W_zero_point->template Data<uint8_t>());
    }
  }
  if (scale_zp_indexs.Y_ZERO_POINT >= 0) {
    const onnxruntime::Tensor* Y_zero_point = nullptr;
    info.TryGetConstantInput(scale_zp_indexs.Y_ZERO_POINT, &Y_zero_point);
    if (Y_zero_point == nullptr) {
      quant_param_.Y_zero_point_value = 0;
    } else {
      quant_param_.Y_zero_point_value = *(Y_zero_point->template Data<uint8_t>());
    }
  }
  if (scale_zp_indexs.X_SCALE >= 0) {
    const onnxruntime::Tensor* X_scale = nullptr;
    info.TryGetConstantInput(scale_zp_indexs.X_SCALE, &X_scale);
    quant_param_.X_scale_value = *(X_scale->template Data<float>());
  }
  if (scale_zp_indexs.W_SCALE >= 0) {
    const onnxruntime::Tensor* W_scale = nullptr;
    info.TryGetConstantInput(scale_zp_indexs.W_SCALE, &W_scale);
    quant_param_.W_scale_value = *(W_scale->template Data<float>());
    if (!IsScalarOr1ElementVector(W_scale)) {
      quant_param_.W_scale_tensor = W_scale;
    }
  }
  if (scale_zp_indexs.Y_SCALE >= 0) {
    const onnxruntime::Tensor* Y_scale = nullptr;
    info.TryGetConstantInput(scale_zp_indexs.Y_SCALE, &Y_scale);
    quant_param_.Y_scale_value = *(Y_scale->template Data<float>());
  }
  return true;
}

}  // namespace xnnpack
}  // namespace onnxruntime
