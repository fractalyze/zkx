#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"

#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "zkx/hlo/parser/hlo_parser.h"
#include "zkx/literal_util.h"
#include "zkx/service/cpu/cpu_compiler.h"

namespace zkx::cpu {

// An adaptor from a lambda that runs tasks and a TaskRunner API.
template <typename Runner, typename WorkerId>
class TaskRunnerAdaptor : public Thunk::TaskRunner {
 public:
  TaskRunnerAdaptor(Runner runner, WorkerId worker_id)
      : runner_(std::move(runner)), worker_id_(std::move(worker_id)) {}

  void operator()(Thunk::Task task) final { runner_(std::move(task)); }

  std::optional<int64_t> current_worker_id() const final {
    return worker_id_();
  }

 private:
  Runner runner_;
  WorkerId worker_id_;
};

BufferAllocations CreateBufferAllocations(absl::Span<Literal*> literals) {
  std::vector<se::DeviceMemoryBase> buffers;
  buffers.reserve(literals.size());

  for (auto* literal : literals) {
    size_t size_in_bytes = literal->size_bytes();
    buffers.emplace_back(literal->untyped_data(), size_in_bytes);
  }

  return BufferAllocations(buffers);
}

template <typename Runner>
auto MakeTaskRunnerFrom(Runner&& runner) {
  auto no_id = []() { return std::nullopt; };
  return TaskRunnerAdaptor<Runner, decltype(no_id)>(
      std::forward<Runner>(runner), no_id);
}

template <typename Runner, typename WorkerId>
auto MakeTaskRunnerFrom(Runner&& runner, WorkerId&& worker_id) {
  return TaskRunnerAdaptor<Runner, WorkerId>(std::forward<Runner>(runner),
                                             std::forward<WorkerId>(worker_id));
}

enum class Testcase { kAdd, kAdd2, kAdd3 };

std::string GetFilename(Testcase test_case) {
  switch (test_case) {
    case Testcase::kAdd: {
      return "add.hlo";
    }
    case Testcase::kAdd2: {
      return "add2.hlo";
    }
    case Testcase::kAdd3: {
      return "add3.hlo";
    }
  }
}

std::tuple<Literal, Literal, Literal> CreateLiterals(Testcase test_case) {
  Literal input1, input2, output;
  switch (test_case) {
    case Testcase::kAdd: {
      input1 = LiteralUtil::CreateR0(1);
      input2 = LiteralUtil::CreateR0(5);
      output = LiteralUtil::CreateR0(0);
      break;
    }
    case Testcase::kAdd2: {
      input1 = LiteralUtil::CreateR2({{6, 3}, {5, 2}});
      input2 = LiteralUtil::CreateR2({{9, 6}, {10, 4}});
      output = LiteralUtil::CreateR2({{0, 0}, {0, 0}});
      break;
    }
    case Testcase::kAdd3: {
      input1 = LiteralUtil::CreateR3(
          {{{1, 2}, {3, 4}, {5, 6}}, {{7, 8}, {9, 10}, {11, 12}}});
      input2 = LiteralUtil::CreateR3(
          {{{5, 7}, {4, 10}, {9, 11}}, {{2, 3}, {6, 5}, {1, 0}}});
      output = LiteralUtil::CreateR3(
          {{{0, 0}, {0, 0}, {0, 0}}, {{0, 0}, {0, 0}, {0, 0}}});
      break;
    }
  }
  return std::make_tuple(std::move(input1), std::move(input2),
                         std::move(output));
}

absl::Status RealMain(int argc, char** argv) {
  mlir::DialectRegistry registry;

  auto mlir_context = std::make_unique<mlir::MLIRContext>(registry);

  Testcase test_case;
  std::string test_case_str = argv[1];
  if (test_case_str == "add") {
    test_case = Testcase::kAdd;
  } else if (test_case_str == "add2") {
    test_case = Testcase::kAdd2;
  } else {
    test_case = Testcase::kAdd3;
  }

  std::string filename =
      std::string("zkx/service/cpu/testdata/") + GetFilename(test_case);
  std::string hlo_string;
  TF_RETURN_IF_ERROR(
      tsl::ReadFileToString(tsl::Env::Default(), filename, &hlo_string));

  TF_ASSIGN_OR_RETURN(std::unique_ptr<HloModule> module,
                      ParseAndReturnUnverifiedModule(hlo_string));

  CpuCompiler compiler;
  TF_ASSIGN_OR_RETURN(std::unique_ptr<Executable> executable,
                      compiler.RunBackend(std::move(module), nullptr,
                                          Compiler::CompileOptions()));

  auto [input1, input2, output] = CreateLiterals(test_case);
  std::vector<Literal*> literals_ptrs = {&output, &input1, &input2};
  BufferAllocations allocations =
      CreateBufferAllocations(absl::MakeSpan(literals_ptrs));

  auto task_runner = MakeTaskRunnerFrom([&](Thunk::Task task) { task(); },
                                        // Always return current worker id as 0.
                                        [] { return 0; });

  auto cpu_executable = static_cast<CpuExecutable*>(executable.get());
  Thunk::ExecuteParams params;
  params.function_library = cpu_executable->function_library();
  params.buffer_allocations = &allocations;
  params.task_runner = &task_runner;
  params.session =
      Thunk::ExecuteSession(/*max_workers=*/1, /*split_threshold=*/0);

  auto execute_event = cpu_executable->thunks().Execute(params);

  tsl::BlockUntilReady(execute_event);
  CHECK(execute_event.IsConcrete());
  std::cout << input1.ToString() << std::endl;
  std::cout << input2.ToString() << std::endl;
  std::cout << output.ToString() << std::endl;

  return absl::OkStatus();
}

}  // namespace zkx::cpu

int main(int argc, char** argv) {
  absl::Status s = zkx::cpu::RealMain(argc, argv);
  if (!s.ok()) {
    std::cerr << s << std::endl;
    return 1;
  }
  return 0;
}
