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

"""Provides the repo macro to import zk_dtypes.

zk_dtypes provides ZK-specific data-types like int4.
"""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    ZK_DTYPES_COMMIT = "b97c731474c33bf3d7ba6289ed53b2b166437c12"
    ZK_DTYPES_SHA256 = "4c29068a66734020e110bf4f6f8c32ec0a26cfde9a8bad918f0390595aa83e5c"
    tf_http_archive(
        name = "zk_dtypes",
        sha256 = ZK_DTYPES_SHA256,
        strip_prefix = "zk_dtypes-{commit}".format(commit = ZK_DTYPES_COMMIT),
        urls = tf_mirror_urls("https://github.com/fractalyze/zk_dtypes/archive/{commit}/zk_dtypes-{commit}.tar.gz".format(commit = ZK_DTYPES_COMMIT)),
    )
