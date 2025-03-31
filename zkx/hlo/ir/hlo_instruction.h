#ifndef ZKX_HLO_IR_HLO_INSTRUCTION_H_
#define ZKX_HLO_IR_HLO_INSTRUCTION_H_

#include <stdint.h>

#include <functional>
#include <iterator>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/inlined_vector.h"
#include "absl/functional/function_ref.h"
#include "absl/log/check.h"

#include "zkx/hlo/ir/hlo_opcode.h"
#include "zkx/hlo/ir/hlo_original_value.h"
#include "zkx/hlo/ir/hlo_print_options.h"
#include "zkx/hlo/ir/ptr_vec.h"
#include "zkx/printer.h"
#include "zkx/service/mapped_ptr_container_sorter.h"
#include "zkx/service/name_uniquer.h"
#include "zkx/shape.h"

namespace zkx {

class HloComputation;
class HloModule;
class HloInstruction;

// A small holder that is used to keep some immutable info alongside an
// instruction pointer in an HloComputation's list of instructions
class HloInstructionInfo {
 public:
  HloInstruction* get() const { return inst_; }
  HloInstruction& operator*() { return *inst_; }
  HloInstruction* operator->() { return inst_; }
  const HloInstruction& operator*() const { return *inst_; }
  const HloInstruction* operator->() const { return inst_; }

  HloOpcode opcode() const { return opcode_; }
  HloInstruction* inst() const { return inst_; }

 private:
  friend class HloComputation;
  HloOpcode opcode_;
  HloInstruction* inst_;
};

namespace mapped_ptr_container_sorter_internal {

template <typename T>
struct PtrGetter<const HloInstructionInfo&, const T*> {
  static const T* Get(const HloInstructionInfo& p) { return p.get(); }
};

}  // namespace mapped_ptr_container_sorter_internal

using HloInstructionList = std::vector<HloInstructionInfo>;

template <typename UnderlyingList>
class HloInstructionIteratorBase {
 public:
  using iterator_category = std::input_iterator_tag;
  using value_type = HloInstructionInfo;
  using difference_type = ptrdiff_t;
  using pointer = value_type*;
  using reference = value_type&;

  HloInstructionIteratorBase(UnderlyingList* list, int begin_index,
                             int end_index)
      : list_(list), current_(begin_index), end_index_(end_index) {
    if (current_ < end_index_ && (*list_)[current_].inst() == nullptr) {
      ++*this;
    }
  }

  HloInstruction* get() const { return (*list_)[current_].inst(); }

  auto operator*() -> HloInstructionInfo { return (*list_)[current_]; }
  HloInstructionIteratorBase& operator++() {
    int next = current_;
    do {
      ++next;
    } while (next < end_index_ && (*list_)[next].inst() == nullptr);
    current_ = next;
    return *this;
  }
  HloInstructionIteratorBase operator++(int) {
    HloInstructionIteratorBase temp(list_, current_, end_index_);
    operator++();
    return temp;
  }

  friend bool operator==(const HloInstructionIteratorBase& a,
                         const HloInstructionIteratorBase& b) {
    return a.current_ == b.current_;
  }

  friend bool operator!=(const HloInstructionIteratorBase& a,
                         const HloInstructionIteratorBase& b) {
    return !(a == b);
  }

 private:
  UnderlyingList* list_;
  int current_;
  int end_index_;
};
using HloInstructionIterator = HloInstructionIteratorBase<HloInstructionList>;
using HloInstructionConstIterator =
    HloInstructionIteratorBase<const HloInstructionList>;

template <typename WrappedIter>
class HloInstructionUnwrappingIteratorBase {
 public:
  using iterator_category = std::input_iterator_tag;
  using value_type = HloInstruction*;
  using difference_type = ptrdiff_t;
  using pointer = value_type*;
  using reference = value_type&;

  explicit HloInstructionUnwrappingIteratorBase(WrappedIter iter)
      : iter_(std::move(iter)) {}

  auto operator*() -> value_type { return iter_.get(); }
  HloInstructionUnwrappingIteratorBase& operator++() {
    ++iter_;
    return *this;
  }
  HloInstructionUnwrappingIteratorBase operator++(int) {
    HloInstructionUnwrappingIteratorBase temp(iter_);
    operator++();
    return temp;
  }

  friend bool operator==(const HloInstructionUnwrappingIteratorBase& a,
                         const HloInstructionUnwrappingIteratorBase& b) {
    return a.iter_ == b.iter_;
  }

  friend bool operator!=(const HloInstructionUnwrappingIteratorBase& a,
                         const HloInstructionUnwrappingIteratorBase& b) {
    return !(a == b);
  }

 private:
  WrappedIter iter_;
};
using HloInstructionUnwrappingIterator =
    HloInstructionUnwrappingIteratorBase<HloInstructionIterator>;
using HloInstructionUnwrappingConstIterator =
    HloInstructionUnwrappingIteratorBase<HloInstructionConstIterator>;

// HLO instructions are the atomic unit of the high-level compiler's IR.
//
// HloInstructions live inside of an HloComputation, which is analogous to a
// function in other programming languages. Nodes have no total order within
// their computation. Instead, they have a partial ordering determined by their
// data and control dependencies.
//
// HLO does not have basic blocks or explicit "branch" instructions. Instead,
// certain HloInstructions -- namely, kWhile, kConditional, and kCall -- encode
// control flow. For example, the kConditional HLO executes one of two possible
// computations, depending on the runtime value of a predicate.
//
// HLO is pure (mostly). It has no concept of mutable state. Instead, data
// values are produced by one HLO and flow into consumers across dependency
// edges.
class HloInstruction {
 public:
  using InstructionVector = absl::InlinedVector<HloInstruction*, 2>;

  inline static constexpr char kMainExecutionThread[] = "main";
  inline static constexpr char kHostThread[] = "host";

  virtual ~HloInstruction() { DetachFromOperandsAndUsers(); }

  // Detaches an instruction from its operands and users. That is, remove the
  // instruction from each operand's user set and user's operand set.
  void DetachFromOperandsAndUsers();

  // Returns the opcode for this instruction.
  HloOpcode opcode() const { return opcode_; }
  HloOpcode* mutable_opcode() { return &opcode_; }

  // Returns whether this instruction is the root of its parent computation.
  bool IsRoot() const { return is_root_; }
  void MarkAsRoot() { is_root_ = true; }
  void MarkAsNonRoot() { is_root_ = false; }

  // Returns the result shape of this instruction.
  const Shape& shape() const { return shape_; }

  // Returns the (mutable) result shape of this instruction.
  Shape* mutable_shape() { return &shape_; }

  // Returns the ith operand to this instruction.
  const HloInstruction* operand(int64_t i) const { return operands_[i]; }

  // Returns the ith operand to this instruction.
  HloInstruction* mutable_operand(int64_t i) {
    CHECK(operands_[i] != nullptr);
    return operands_[i];
  }

  // Returns the number of operands to this instruction.
  int64_t operand_count() const { return operands_.size(); }

  const InstructionVector& operands() const { return operands_; }
  InstructionVector mutable_operands() { return operands_; }

  // Returns the vector of unique operands, in the same order they are found
  // within the operand vector.
  InstructionVector unique_operands() const;

  // Returns the first index of 'target' that occurs in the operands sequence.
  // Precondition: target must be an operand (or a fatal error will occur).
  int64_t operand_index(const HloInstruction* target) const;

  // Returns all indices of 'target' that occur in the operands sequence.
  // Precondition: target must be an operand (or a fatal error will occur).
  std::vector<int64_t> operand_indices(const HloInstruction* target) const;

  // Returns the number of users of this instruction.
  int64_t user_count() const { return users_.size(); }

  // Returns the users of this instruction.
  const PtrVec<HloInstruction*>& users() const { return users_.vec(); }

  // Returns the index of the user in the users() vector.
  //
  // Precondition: `user` is a user of the instruction.
  int64_t UserId(HloInstruction* user) { return users_.UserId(user); }

  // Returns true if this instruction is a user of 'instruction'.
  bool IsUserOf(const HloInstruction* instruction) const {
    return instruction->users_.Contains(this);
  }

  // Prints a debugging string that represents this instruction.
  void Print(Printer* printer) const {
    return Print(printer, HloPrintOptions::Default());
  }
  void Print(Printer* printer, const HloPrintOptions& options) const;

  // Returns a debugging string that represents this instruction.
  //
  // (We express the default options using an overload rather than a default
  // param because gdb ignores default params, but does resolve overloads.)
  //
  // TODO(b/73348663): Make ToString() adaptive to the size of the string by
  // default, backing off on providing full information for very large strings,
  // or provide a different name for a ToString-like function that does that.
  std::string ToString() const;
  std::string ToString(const HloPrintOptions& options) const;

  // Prints an instruction to a string.
  //
  // The canonical string representation needs to name operands and instruction
  // names in a consistent way. This is implemented through the
  // canonical_name_map.
  void PrintWithCanonicalNameMap(Printer* printer,
                                 const HloPrintOptions& options,
                                 CanonicalNameMap* canonical_name_map) const;

  // Gets the string identifier for this instruction.
  std::string_view name() const { return name_; }

  // Sets the string identifier for this instruction. Name will be sanitized to
  // match the regexp "[a-zA-Z_][a-zA-Z0-9_.-]*".
  //
  // See also HloModule::SetAndUniquifyInstrName(), which does this plus
  // UniquifyName().
  void SetAndSanitizeName(std::string_view name) {
    name_ = NameUniquer::GetSanitizedName(name);
  }

  // Clear the unique ID of the instruction so that it can be re-assigned, such
  // as for the purpose of compacting the instruction unique IDs.
  void ClearUniqueIdInternal() { unique_id_ = -1; }

  // Set the unique id for this instruction to "id"
  void SetUniqueId(int id) {
    CHECK_EQ(unique_id_, -1);  // Should not be assigned already
    CHECK_GE(id, 0);
    unique_id_ = id;
  }

  // Return the unique ID assigned to this nod`e via SetUniqueId (or -1
  // if no id has been assigned yet).
  int unique_id() const { return unique_id_; }

  std::shared_ptr<OriginalValue> original_value() const {
    return original_value_;
  }
  void set_original_value(std::shared_ptr<OriginalValue> original_value) {
    original_value_ = original_value;
  }

 private:
  // Prints an operand to a string. Accessed by friend class HloInstruction.
  virtual void PrintOperandsWithCanonicalNameMap(
      Printer* printer, const HloPrintOptions& options,
      CanonicalNameMap* canonical_name_map) const;

  // Users holds the list of users of an HloInstruction, plus it provides a fast
  // way for checking for presence of a potential user.
  class Users {
   public:
    Users() = default;
    ~Users() = default;

    // No copying allowed
    Users(const Users&) = delete;
    Users& operator=(const Users&) = delete;

    bool empty() const { return users_.empty(); }
    int64_t size() const { return users_.size(); }
    const PtrVec<HloInstruction*>& vec() const { return users_; }

    void Clear();
    bool Contains(const HloInstruction* instruction) const;
    void AddUser(HloInstruction* user);
    void MaybeRemoveUser(HloInstruction* user);  // Remove user if present
    void RemoveUser(HloInstruction* user);       // REQUIRES: Contains(user)
    int64_t UserId(HloInstruction* user);
    void SortInstructionUsers(
        const MappedPtrContainerSorter<HloInstruction>::MapPtrFn& map_fn,
        const Users& sorted_instruction_users);
    bool CheckInvariants();

   private:
    void RebuildMap();

    PtrVec<HloInstruction*> users_;

    // If users_ is big, we also maintain a copy of the elements of users_
    // in a hash map to enable fast membership tests. The value in the map
    // contains the index of the instruction in the vector what enables fast
    // removal.
    static constexpr size_t kMapThreshold = 16;
    std::unique_ptr<absl::flat_hash_map<const HloInstruction*, int64_t>>
        user_map_;
  };

  int unique_id_;  // Unique to this HloInstruction within a HloModule

  // Opcode for this instruction.
  HloOpcode opcode_;

  // True if this instruction has already been detached from its user and
  // operands.
  bool cleaned_up_ : 1;

  // True if this instruction is the root of a computation.
  bool is_root_ : 1;

  // Instruction operands.
  InstructionVector operands_;

  // The users of this instruction. Users are HLOs where this instruction is an
  // operand.
  Users users_;

  // Result shape of this instruction.
  Shape shape_;

  // String identifier for instruction.
  std::string name_;

  // Original value this instruction corresponds to in the unoptimized HLO
  // graph.
  std::shared_ptr<OriginalValue> original_value_ = nullptr;
};

}  // namespace zkx

#endif  // ZKX_HLO_IR_HLO_INSTRUCTION_H_
