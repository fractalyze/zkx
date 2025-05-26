/* Copyright 2018 The OpenXLA Authors.

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

#include "zkx/hlo/ir/hlo_module_group.h"

namespace zkx {

HloModuleGroup::HloModuleGroup(std::unique_ptr<HloModule> module)
    : name_(module->name()) {
  push_back(std::move(module));
}

HloModuleGroup::HloModuleGroup(std::string_view name,
                               absl::Span<std::unique_ptr<HloModule>> modules)
    : name_(name) {
  for (auto& module : modules) {
    push_back(std::move(module));
  }
}

HloModuleGroup::HloModuleGroup(
    std::string_view name, std::vector<std::unique_ptr<HloModule>>&& modules)
    : name_(name) {
  for (auto& module : modules) {
    push_back(std::move(module));
  }
}

std::vector<std::unique_ptr<HloModule>> HloModuleGroup::ConsumeModules() {
  std::vector<std::unique_ptr<HloModule>> ret_modules = std::move(modules_);

  // Clear everything so the object state is in a known (empty) state.
  modules_.clear();
  module_ptrs_.clear();
  return ret_modules;
}

std::string HloModuleGroup::ToString() const {
  std::ostringstream s;
  s << "HloModuleGroup " << name() << "\n\n";
  for (const HloModule* module : modules()) {
    s << module->ToString() << "\n";
  }
  return s.str();
}

void HloModuleGroup::push_back(std::unique_ptr<HloModule> module) {
  module->metadata()->set_module_group_name(name());
  modules_.push_back(std::move(module));
  module_ptrs_.push_back(modules_.back().get());
}

void HloModuleGroup::ReplaceModule(int index,
                                   std::unique_ptr<HloModule> module) {
  modules_.at(index)->MoveMetadataToModule(module.get());
  modules_.at(index) = std::move(module);
  module_ptrs_.at(index) = modules_.at(index).get();
}

std::ostream& operator<<(std::ostream& out, const HloModuleGroup& group) {
  out << group.ToString();
  return out;
}

#include <utility>

}  // namespace zkx
