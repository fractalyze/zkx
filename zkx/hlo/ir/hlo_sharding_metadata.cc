/* Copyright 2018 The OpenXLA Authors.
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

#include "zkx/hlo/ir/hlo_sharding_metadata.h"

#include "absl/log/log.h"

#include "zkx/hlo/ir/hlo_instruction.h"

namespace zkx {

namespace {

bool ShardingMatches(const HloSharding& sharding1,
                     const HloSharding& sharding2) {
  auto single_sharding1 = sharding1.ExtractSingleSharding();
  if (single_sharding1) {
    auto single_sharding2 = sharding2.ExtractSingleSharding();
    if (single_sharding2) {
      return *single_sharding1 == single_sharding2;
    }
  }
  // Anything which is not unique across all elements, gets a full sharding
  // compare.
  return sharding1 == sharding2;
}

// For tuple shardings if every element have the same sharding then we want to
// treat them as single element shardings to insert less domain separation as a
// domain can prevent some optimizations and we want to minimize that from
// happening.
std::shared_ptr<const HloSharding> CloneShardingForDomain(
    std::shared_ptr<const HloSharding> sharding) {
  auto single_sharding = sharding->ExtractSingleSharding();
  if (!single_sharding) {
    return sharding;
  }
  return std::make_shared<const HloSharding>(*single_sharding);
}

}  // namespace

std::unique_ptr<DomainMetadata> ShardingMetadata::Clone() const {
  std::unique_ptr<HloSharding> sharding;
  if (sharding_ != nullptr) {
    sharding = std::make_unique<HloSharding>(*sharding_);
  }
  return std::make_unique<ShardingMetadata>(std::move(sharding));
}

bool ShardingMetadata::Matches(const DomainMetadata& other) const {
  const ShardingMetadata* other_ptr =
      dynamic_cast<const ShardingMetadata*>(&other);
  if (other_ptr == nullptr) {
    // If other is not a ShardingMetadata, then it is clearly a no match.
    return false;
  }
  if (sharding_ == nullptr) {
    return other_ptr->sharding_ == nullptr;
  }
  return other_ptr->sharding_ != nullptr
             ? ShardingMatches(*sharding_, *other_ptr->sharding_)
             : false;
}

std::string ShardingMetadata::ToString() const {
  return sharding_ != nullptr ? sharding_->ToString() : "{}";
}

// static
absl::StatusOr<const ShardingMetadata*> ShardingMetadata::ToShardingMetadata(
    const DomainMetadata* metadata) {
  if (metadata->Kind() != ShardingMetadata::KindName()) {
    return absl::InvalidArgumentError(
        "ShardingMetadata normalizer called with incorrect domain metadata");
  }
  return static_cast<const ShardingMetadata*>(metadata);
}

absl::Status ShardingMetadata::NormalizeShardingDomain(
    const DomainMetadata::Domain& domain, const DomainMetadata* metadata) {
  // TODO(chokobole): Implement this.
  return absl::UnimplementedError("Not implemented");
}

// Creates a kDomain instruction to be placed between instruction and operand.
// The kDomain instruction will be created only if the sharding differ between
// the instruction and the operand.
HloInstruction* ShardingDomainCreator::operator()(HloInstruction* instruction,
                                                  HloInstruction* root,
                                                  HloInstruction* operand) {
  auto instruction_sharding = instruction->sharding_ptr();
  auto root_sharding = root->sharding_ptr();
  // No need for domain if they both have no sharding.
  if (instruction_sharding == nullptr && root_sharding == nullptr) {
    return nullptr;
  }
  // No need for domain if they match.
  if (instruction_sharding != nullptr && root_sharding != nullptr &&
      ShardingMatches(*instruction_sharding, *root_sharding)) {
    return nullptr;
  }

  if (instruction_sharding != nullptr) {
    instruction_sharding = CloneShardingForDomain(instruction_sharding);
  }
  if (root_sharding != nullptr) {
    root_sharding = CloneShardingForDomain(root_sharding);
  }

  auto it = domain_cse_map_.find({operand, instruction_sharding});
  if (it != domain_cse_map_.end()) {
    return it->second;
  }

  VLOG(3) << "Creating domain:";
  VLOG(3) << "  Instruction: " << instruction->name();
  VLOG(3) << "  Operand: " << operand->name();
  VLOG(3) << "    User side sharding: "
          << (instruction_sharding != nullptr ? instruction_sharding->ToString()
                                              : "None");
  VLOG(3) << "    Operand side sharding: "
          << (root_sharding != nullptr ? root_sharding->ToString() : "None");

  // TODO(chokobole): Uncomment this. Dependency: HloInstruction::CreateDomain
  // HloInstruction* domain =
  //     operand->parent()->AddInstruction(HloInstruction::CreateDomain(
  //         operand->shape(), operand,
  //         std::make_unique<ShardingMetadata>(root_sharding),
  //         std::make_unique<ShardingMetadata>(instruction_sharding)));
  // domain_cse_map_.emplace(DomainCseMapKey{operand, instruction_sharding},
  //                         domain);
  // return domain;
  return nullptr;
}

bool ShardingDomainCreator::DomainCseMapKey::operator==(
    const ShardingDomainCreator::DomainCseMapKey& other) const {
  if (instruction != other.instruction) {
    return false;
  }
  if (sharding == nullptr && other.sharding == nullptr) {
    return true;
  }
  if (sharding == nullptr || other.sharding == nullptr) {
    return false;
  }
  return *sharding == *other.sharding;
}

}  // namespace zkx
