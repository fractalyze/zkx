#include "zkx/service/cpu/cpu_executable.h"

#include "xla/tsl/platform/statusor.h"
#include "zkx/base/logging.h"

namespace zkx::cpu {

// static
absl::StatusOr<std::unique_ptr<CpuExecutable>> CpuExecutable::Create(
    std::unique_ptr<FunctionLibrary> function_library,
    std::unique_ptr<const BufferAssignment> assignment,
    std::unique_ptr<HloModule> hlo_module, ThunkSequence thunks) {
  // TODO(chokboole): Uncomment this. Dependency: ConstantAllocation
  VLOG(2) << "Create CpuExecutable from a thunk sequence; module="
          << hlo_module->name();  // << ", constants=" << constants.size();

  std::unique_ptr<CpuExecutable> executable(
      new CpuExecutable(std::move(hlo_module), std::move(assignment)));
  executable->function_library_ = std::move(function_library);

  TF_ASSIGN_OR_RETURN(executable->thunks_,
                      ThunkExecutor::Create(std::move(thunks)));

  // Re-index constants by their allocation index to allow efficient lookup.
  // TODO(chokobole): Uncomment this. Dependency: ConstantAllocation
  // for (auto& constant : constants) {
  //   if (executable->constants_.size() <= constant.index) {
  //     executable->constants_.resize(constant.index + 1);
  //   }
  //   executable->constants_[constant.index] = std::move(constant);
  // }

  return executable;
}

CpuExecutable::CpuExecutable(std::unique_ptr<HloModule> hlo_module,
                             std::unique_ptr<const BufferAssignment> assignment)
    : hlo_module_(std::move(hlo_module)), assignment_(std::move(assignment)) {
  // TODO(chokobole): Uncomment this. Dependency: ZkxDebugInfoManager
  // if (assignment_ && has_module()) {
  //   ZkxDebugInfoManager::Get()->RegisterModule(shared_module(),
  //                                              assignment_->ToProto());
  // }
}

}  // namespace zkx::cpu
