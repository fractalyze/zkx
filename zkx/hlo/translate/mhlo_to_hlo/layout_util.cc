/* Copyright 2017 The OpenXLA Authors.
Copyright 2025 The ZKX Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "zkx/hlo/translate/mhlo_to_hlo/layout_util.h"

#include <cstdint>
#include <vector>

#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "zkx/shape_util.h"

namespace mlir {

// Rewrites the layout of zkx_shape if there is tiled sharding.
absl::Status RewriteLayoutWithShardedShape(
    const std::optional<zkx::HloSharding>& sharding, bool use_fast_memory,
    const LayoutPreferenceFn& layout_preference_fn,
    const ShapeRepresentationFn& shape_representation_fn,
    zkx::Shape* zkx_shape) {
  if (sharding && !sharding->IsTileMaximal() && !sharding->IsManual()) {
    // After sharding, per core shape might have different layout. For example,
    // before sharding, a shape [128, 128] will be assigned default
    // minor-to-major {1, 0}. But after we shard this shape to [128, 64] * 2,
    // the sharded shapes will have minor-to-major {0, 1}.
    //
    // As a result, for sharded shapes, we set their layout to per core shape's
    // layout.
    //
    // TODO(endlessroad): for variable input & update, we might have
    // different layouts which will prevent input output aliasing and
    // increase memory usage. Investigate such cases.
    int64_t device = sharding->tile_assignment().first();
    std::vector<int64_t> offset =
        sharding->TileOffsetForDevice(*zkx_shape, device);
    std::vector<int64_t> limit =
        sharding->TileLimitForDevice(*zkx_shape, device);
    std::vector<int64_t> dimensions(zkx_shape->rank());
    for (int64_t i = 0; i < zkx_shape->rank(); ++i) {
      dimensions[i] = limit[i] - offset[i];
    }
    zkx::Shape per_device_zkx_shape =
        zkx::ShapeUtil::MakeShape(zkx_shape->element_type(), dimensions);
    TF_ASSIGN_OR_RETURN(auto layout_preference,
                        layout_preference_fn
                            ? layout_preference_fn(per_device_zkx_shape)
                            : ZkxLayoutPreference::kNoPreference);
    TF_ASSIGN_OR_RETURN(
        per_device_zkx_shape,
        shape_representation_fn
            ? shape_representation_fn(per_device_zkx_shape, use_fast_memory,
                                      layout_preference)
            : per_device_zkx_shape);
    *zkx_shape->mutable_layout() = per_device_zkx_shape.layout();
  }
  return absl::OkStatus();
}

// There is a shape_representation_fn or sharding for an output, this function
// uses a reshape to fix the layout.
absl::StatusOr<zkx::ZkxOp> ReshapeWithCorrectRepresentationAndSharding(
    zkx::ZkxBuilder* builder, zkx::ZkxOp original, zkx::Shape original_shape,
    const LayoutPreferenceFn& layout_preference_fn,
    const ShapeRepresentationFn& shape_representation_fn,
    std::optional<zkx::OpSharding> sharding, bool fast_mem) {
  if (original_shape.IsTuple()) {
    std::vector<zkx::ZkxOp> elements;
    for (int i = 0; i < original_shape.tuple_shapes_size(); ++i) {
      auto subsharding = sharding ? sharding->tuple_shardings(i) : sharding;
      TF_ASSIGN_OR_RETURN(
          auto element,
          ReshapeWithCorrectRepresentationAndSharding(
              builder, zkx::GetTupleElement(original, i),
              original_shape.tuple_shapes(i), layout_preference_fn,
              shape_representation_fn, subsharding, fast_mem));
      elements.push_back(element);
    }
    return zkx::Tuple(builder, elements);
  }
  if (!original_shape.IsArray()) return original;
  TF_ASSIGN_OR_RETURN(auto layout_preference,
                      layout_preference_fn
                          ? layout_preference_fn(original_shape)
                          : ZkxLayoutPreference::kNoPreference);
  TF_ASSIGN_OR_RETURN(
      auto to_shape,
      shape_representation_fn
          ? shape_representation_fn(original_shape, fast_mem, layout_preference)
          : original_shape);
  if (sharding) {
    TF_ASSIGN_OR_RETURN(auto hlo_sharding,
                        zkx::HloSharding::FromProto(*sharding));

    TF_RETURN_IF_ERROR(RewriteLayoutWithShardedShape(
        hlo_sharding, fast_mem, layout_preference_fn, shape_representation_fn,
        &to_shape));
  }
  if (zkx::ShapeUtil::Compatible(original_shape, to_shape)) {
    for (int64_t i = 0; i < original_shape.rank(); ++i) {
      to_shape.set_dynamic_dimension(i, original_shape.is_dynamic_dimension(i));
    }
  }
  zkx::ZkxScopedShardingAssignment scoped_sharding(builder, sharding);
  return zkx::Reshape(to_shape, original);
}

}  // namespace mlir
