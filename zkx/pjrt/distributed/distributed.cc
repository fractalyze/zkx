/* Copyright 2020 The OpenXLA Authors.
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

#include "zkx/pjrt/distributed/distributed.h"

#include "grpcpp/create_channel.h"

namespace zkx {

absl::StatusOr<std::unique_ptr<DistributedRuntimeService>>
GetDistributedRuntimeService(std::string address,
                             const CoordinationServiceImpl::Options& options) {
  // TODO(chokobole): support creating secure credentials.
  return DistributedRuntimeService::Get(
      address, grpc::InsecureServerCredentials(), options);
}

std::shared_ptr<DistributedRuntimeClient> GetDistributedRuntimeClient(
    std::string address, const DistributedRuntimeClient::Options& options,
    bool use_compression) {
  // TODO(chokobole): support creating secure credentials.
  std::shared_ptr<grpc::Channel> channel = GetDistributedRuntimeClientChannel(
      address, grpc::InsecureChannelCredentials(), use_compression);
  return GetDistributedRuntimeClient(channel, options);
}

std::shared_ptr<grpc::Channel> GetDistributedRuntimeClientChannel(
    std::string address, std::shared_ptr<grpc::ChannelCredentials> creds,
    bool use_compression) {
  grpc::ChannelArguments args;
  if (use_compression) {
    args.SetCompressionAlgorithm(GRPC_COMPRESS_GZIP);
  }
  return grpc::CreateCustomChannel(address, creds, args);
}

}  // namespace zkx
