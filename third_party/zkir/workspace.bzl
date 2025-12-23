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
    ZKIR_COMMIT = "775c57412295e73ef1652d36a39058b7cec2e853"
    ZKIR_SHA256 = "6b4b288a80c6b0d6d319f9573375cea6d2f58c8e467267f6b0e46b1734cb146a"
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
