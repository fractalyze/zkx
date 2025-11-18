# Copyright The OpenXLA Authors.
# Copyright 2025 The ZKX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""TSL rules."""

# These configs are used to determine whether we should use CUDA tools and libs in cc_libraries.
# They are intended for the OSS builds only.
def if_cuda_tools(if_true, if_false = []):  # buildifier: disable=unused-variable
    """Shorthand for select()'ing on whether we're building with hCUDA tools."""
    return select({"@local_config_cuda//cuda:cuda_tools": if_true, "//conditions:default": if_false})  # copybara:comment_replace return if_false

def if_cuda_libs(if_true, if_false = []):  # buildifier: disable=unused-variable
    """Shorthand for select()'ing on whether we need to include hermetic CUDA libraries."""
    return select({"@local_config_cuda//cuda:cuda_tools_and_libs": if_true, "//conditions:default": if_false})  # copybara:comment_replace return if_false
