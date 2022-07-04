// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <memory>
#include <vector>
#include <string>
#include <utility>

#include "core/framework/op_kernel.h"
#include "core/graph/indexed_sub_graph.h"
#include "core/providers/common.h"
#include "core/providers/shared/node_unit/node_unit.h"

#include "xnnpack.h"

namespace onnxruntime {
class GraphViewer;
class NodeUnit;
namespace xnnpack {

enum OpComputeType : uint8_t {
  op_compute_type_invalid = 0,
  op_compute_type_fp32,
  op_compute_type_fp16,
  op_compute_type_qs8_per_channel,
  op_compute_type_qs8,
  op_compute_type_qu8,
};

struct InputTensorOrder {
  int X_IN = -1;
  int X_SCALE = -1;
  int X_ZERO_POINT = -1;
  int W_CONST = -1;
  int W_SCALE = -1;
  int W_ZERO_POINT = -1;
  int Y_SCALE = -1;
  int Y_ZERO_POINT = -1;
  int BIAS = -1;
};

struct QuantParam {
  uint8_t X_zero_point_value = 0;
  uint8_t W_zero_point_value = 0;
  uint8_t Y_zero_point_value = 0;

  float X_scale_value = 0;
  float W_scale_value = 0;
  const Tensor* W_scale_tensor = nullptr;
  float Y_scale_value = 0;
};

using Shape = std::vector<uint32_t>;
enum class QuantizedOpType : uint8_t {
  QLinearConv,
  QLinearMaxPool,
  QlinearAvgPool,
  // QDQ operator
  QDQConv,
  QDQMaxPool,
  QDQAvgPool,
  QDQSoftmax,
  Unknown,
};

QuantizedOpType GetQuantizedOpType(const NodeUnit& node_unit);

// forward declaration for this EP's namespace.
template <typename T>
KernelCreateInfo BuildKernelCreateInfo();

struct XnnpackOperatorDeleter {
  void operator()(struct xnn_operator* p) const {
    if (p != nullptr) {
      // Ignore returned value because it fails only when xnnpack wasn't initialized
      xnn_delete_operator(p);
    }
  }
};

bool IsPaddingTypeSupported(AutoPadType auto_pad);

using XnnpackOperator = std::unique_ptr<struct xnn_operator, XnnpackOperatorDeleter>;

std::unique_ptr<IndexedSubGraph::MetaDef> FuseActivation(const Node& conv, const Node& activation,
                                                         const GraphViewer& graph);
std::unique_ptr<IndexedSubGraph::MetaDef> FuseQDQGroup(const NodeUnit& unit_node);

bool GetType(const NodeArg& node_arg, int32_t& type);
bool GetShape(const NodeArg& node_arg, Shape& shape);
bool ParseQuantParamFromInfoByOrder(const OpKernelInfo& info,
                                    const InputTensorOrder& scale_zp_indexs,
                                    QuantParam& quant_param_);

bool IsQuantizedConv(QuantizedOpType quant_op_type);

bool IsQuantizedMaxPool(QuantizedOpType quant_op_type);

bool IsQuantizedAvgPool(QuantizedOpType quant_op_type);

bool IsQuantizedSoftmax(QuantizedOpType quant_op_type);
xnn_datatype GetDtypeInXnnpack(const onnxruntime::NodeUnit& node_unit, int32_t io_index,
                               bool is_output, const onnxruntime::GraphViewer& graph_viewer);
const onnx::TensorProto* GetQuantizationScale(const InitializedTensorSet& initializers,
                                              const NodeUnitIODef& io_def);

const onnx::TensorProto* GetQuantizationZeroPoint(const InitializedTensorSet& initializers,
                                                  const NodeUnitIODef& io_def);
}  // namespace xnnpack
}  // namespace onnxruntime
