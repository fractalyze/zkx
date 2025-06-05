#include "zkx/backends/cpu/codegen/cpu_kernel_emitter_test.h"

#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/statusor.h"
#include "zkx/backends/cpu/runtime/thread_pool_task_runner.h"
#include "zkx/hlo/parser/hlo_parser.h"
#include "zkx/literal_util.h"
#include "zkx/permutation_util.h"
#include "zkx/service/cpu/cpu_compiler.h"

namespace zkx::cpu {

namespace {

BufferAllocations CreateBufferAllocations(absl::Span<Literal*> literals) {
  std::vector<se::DeviceMemoryBase> buffers;
  buffers.reserve(literals.size());

  for (auto* literal : literals) {
    size_t size_in_bytes = literal->size_bytes();
    buffers.emplace_back(literal->untyped_data(), size_in_bytes);
  }

  return BufferAllocations(buffers);
}

}  // namespace

void CpuKernelEmitterTest::RunHlo(std::string_view hlo_string,
                                  absl::Span<Literal*> literals) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnUnverifiedModule(hlo_string));

  CpuCompiler compiler;
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Executable> executable,
                          compiler.RunBackend(std::move(module), nullptr,
                                              Compiler::CompileOptions()));

  ThreadPoolTaskRunner task_runner(nullptr);

  auto cpu_executable = static_cast<CpuExecutable*>(executable.get());

  const BufferAssignment& buffer_assignment =
      cpu_executable->buffer_assignment();
  const HloInstruction* root_instruction =
      cpu_executable->module().entry_computation()->root_instruction();
  std::vector<int64_t> permutations;
  for (int64_t i = 0; i < root_instruction->operand_count(); ++i) {
    TF_ASSERT_OK_AND_ASSIGN(
        BufferAllocation::Slice slice,
        buffer_assignment.GetUniqueTopLevelSlice(root_instruction->operand(i)));
    if (std::find(permutations.begin(), permutations.end(), slice.index()) ==
        permutations.end()) {
      permutations.push_back(slice.index());
    }
  }
  TF_ASSERT_OK_AND_ASSIGN(
      BufferAllocation::Slice slice,
      buffer_assignment.GetUniqueTopLevelSlice(root_instruction));
  if (std::find(permutations.begin(), permutations.end(), slice.index()) ==
      permutations.end()) {
    permutations.push_back(slice.index());
  }
  std::vector<Literal*> permuted_literals =
      PermuteInverse(literals, permutations);
  BufferAllocations allocations =
      CreateBufferAllocations(absl::MakeSpan(permuted_literals));

  Thunk::ExecuteParams params;
  params.function_library = cpu_executable->function_library();
  params.buffer_allocations = &allocations;
  params.task_runner = &task_runner;
  params.session =
      Thunk::ExecuteSession(/*max_workers=*/1, /*split_threshold=*/0);

  auto execute_event = cpu_executable->thunks().Execute(params);

  tsl::BlockUntilReady(execute_event);
  ASSERT_TRUE(execute_event.IsConcrete());
}

}  // namespace zkx::cpu
