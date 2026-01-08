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

"""Provides the repo macro to import zkir.

zkir provides mlir dialect for ZK(Zero Knowledge).
"""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    ZKIR_COMMIT = "a498f78630a2ca5e1263a6b5262044f2bb0cfe18"
    ZKIR_SHA256 = "f80f6172c6feb9f4628a169d7034b87678aa6b47eaa2c3f4d2f759f84893f6e9"
    tf_http_archive(
        name = "zkir",
        sha256 = ZKIR_SHA256,
        strip_prefix = "zkir-{commit}".format(commit = ZKIR_COMMIT),
        urls = tf_mirror_urls("https://github.com/fractalyze/zkir/archive/{commit}.tar.gz".format(commit = ZKIR_COMMIT)),
    )
    # Uncomment this for development!
    # native.local_repository(
    #     name = "zkir",
    #     path = "../zkir",
    # )
