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
    ZKIR_COMMIT = "e48fb953e7ae6dd44293567826e99b057147b7be"
    ZKIR_SHA256 = "8a01f95d3d4db77ca18218bbf71c31c26a55d1b55f94aef0183dda99ecf0800d"
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
