#ifndef ZKX_HLO_IR_HLO_INSTRUCTION_H_
#define ZKX_HLO_IR_HLO_INSTRUCTION_H_

#include <string>

#include "zkx/shape.h"

namespace zkx {

// TODO(chokobole): Rename the name of this class.
class HloInstruction {
 public:
  // Returns the result shape of this instruction.
  const Shape& shape() const { return shape_; }

  // Returns the (mutable) result shape of this instruction.
  Shape* mutable_shape() { return &shape_; }

  // Gets the string identifier for this instruction.
  std::string_view name() const { return name_; }

 private:
  // Result shape of this instruction.
  Shape shape_;

  std::string name_;
};

}  // namespace zkx

#endif  // ZKX_HLO_IR_HLO_INSTRUCTION_H_
