#ifndef ZKX_HLO_IR_HLO_INSTRUCTION_H_
#define ZKX_HLO_IR_HLO_INSTRUCTION_H_

#include <string>

namespace zkx {

// TODO(chokobole): Rename the name of this class.
class HloInstruction {
 public:
  // Gets the string identifier for this instruction.
  std::string_view name() const { return name_; }

 private:
  std::string name_;
};

}  // namespace zkx

#endif  // ZKX_HLO_IR_HLO_INSTRUCTION_H_
