#include "zkx/backends/cpu/codegen/cpu_kernel_emitter_test.h"

#include "xla/tsl/platform/statusor.h"
#include "zkx/hlo/parser/hlo_parser.h"

#include "zkx/service/platform_util.h"

namespace zkx::cpu {

CpuKernelEmitterTest::CpuKernelEmitterTest()
    : runner_(PlatformUtil::GetPlatform("cpu").value()) {}

void CpuKernelEmitterTest::Compile(std::string_view hlo_string) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnUnverifiedModule(hlo_string));

  TF_ASSERT_OK_AND_ASSIGN(
      opaque_executable_,
      runner_.CreateExecutable(std::move(module), /*run_hlo_passes=*/false));
}

absl::StatusOr<Literal> CpuKernelEmitterTest::Run(
    absl::Span<Literal> arguments) {
  return runner_.ExecuteWithExecutable(opaque_executable_.get(), arguments,
                                       /*profile=*/nullptr);
}

}  // namespace zkx::cpu
