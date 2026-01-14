# Copyright 2026 The ZKX Authors.
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

# buildifier: disable=module-docstring
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")
load("//bazel:zkx_cc.bzl", "zkx_cc_test")
load("//xla/tsl/platform:build_config_root.bzl", "tf_gpu_tests_tags")
load("//zkx/tests:plugin.bzl", "plugins")

NVIDIA_GPU_BACKENDS = [
    "gpu_any",
    "gpu_p100",
    "gpu_v100",
    "gpu_a100",
    "gpu_h100",
    "gpu_b200",
]

# The generic "gpu" backend includes the actual backends in this list.
NVIDIA_GPU_DEFAULT_BACKENDS = [
    "gpu_any",
    "gpu_a100",
    "gpu_h100",
    "gpu_b200",
]

AMD_GPU_DEFAULT_BACKENDS = ["gpu_amd_any"]

_DEFAULT_BACKENDS = ["cpu"] + NVIDIA_GPU_DEFAULT_BACKENDS + AMD_GPU_DEFAULT_BACKENDS

GPU_BACKENDS = NVIDIA_GPU_BACKENDS + AMD_GPU_DEFAULT_BACKENDS

GPU_DEFAULT_BACKENDS = NVIDIA_GPU_DEFAULT_BACKENDS

DEFAULT_DISABLED_BACKENDS = []

_ALL_BACKENDS = ["cpu", "interpreter"] + NVIDIA_GPU_BACKENDS + AMD_GPU_DEFAULT_BACKENDS + list(plugins.keys())

# buildifier: disable=function-docstring
def prepare_nvidia_gpu_backend_data(backends, disabled_backends, backend_tags, backend_args):
    # Expand "gpu" backend name into device specific backend names.
    new_backends = [name for name in backends if name != "gpu"]
    if len(new_backends) < len(backends):
        new_backends.extend(NVIDIA_GPU_DEFAULT_BACKENDS)

    new_disabled_backends = [name for name in disabled_backends if name != "gpu"]
    if len(new_disabled_backends) < len(disabled_backends):
        new_disabled_backends.extend(NVIDIA_GPU_BACKENDS)

    new_backend_tags = {key: value for key, value in backend_tags.items() if key != "gpu"}
    gpu_backend_tags = backend_tags.get("gpu", tf_gpu_tests_tags())
    for key in NVIDIA_GPU_BACKENDS:
        new_backend_tags.setdefault(key, gpu_backend_tags[:])

    new_backend_args = {key: value for key, value in backend_args.items() if key != "gpu"}
    if "gpu" in backend_args:
        for key in NVIDIA_GPU_BACKENDS:
            new_backend_args.setdefault(key, backend_args["gpu"])

    # Disable backends that don't meet the device requirements.
    sm_requirements = {
        "gpu_any": (0, 0),
        "gpu_p100": (6, 0),
        "gpu_v100": (7, 0),
        "gpu_a100": (8, 0),
        "gpu_h100": (9, 0),
        "gpu_b200": (10, 0),
    }
    for gpu_backend in NVIDIA_GPU_BACKENDS:
        all_tags = new_backend_tags[gpu_backend]
        requires_gpu = [t for t in all_tags if t.startswith("requires-gpu-")]
        requires_sm, only = None, False
        num_gpus = None
        for tag in requires_gpu:
            if ":" in tag:  # Multi-GPU tests are suffixed with colon and number of GPUs.
                tag, suffix = tag.split(":")  # Remove the suffix from the tag for further parsing.
                parsed_num_gpus = int(suffix)
                if num_gpus and num_gpus != parsed_num_gpus:
                    fail("Inconsistent number of GPUs: %d vs %d" % (num_gpus, parsed_num_gpus))
                num_gpus = parsed_num_gpus
            if tag.startswith("requires-gpu-sm"):
                version = tag.split("-")[2][2:]
                sm = (int(version[:-1]), int(version[-1]))
                if not requires_sm or sm < requires_sm:
                    requires_sm = sm
                if tag.endswith("-only"):
                    only = True
        if only:
            disable = requires_sm != sm_requirements[gpu_backend]
        else:
            disable = requires_sm and requires_sm > sm_requirements[gpu_backend]

        if disable:
            new_disabled_backends.append(gpu_backend)
        else:
            sm_major, sm_minor = sm_requirements[gpu_backend]
            sm_tag = "requires-gpu-nvidia" if sm_major == 0 else "requires-gpu-sm%s%s-only" % (sm_major, sm_minor)
            if num_gpus:
                sm_tag += ":%d" % num_gpus
            new_backend_tags[gpu_backend] = [t for t in all_tags if t not in requires_gpu]
            new_backend_tags[gpu_backend].append(sm_tag)
            new_backend_tags[gpu_backend].append("cuda-only")

    return new_backends, new_disabled_backends, new_backend_tags, new_backend_args

# buildifier: disable=function-docstring
def prepare_amd_gpu_backend_data(backends, disabled_backends, backend_tags, backend_args):
    new_backends = [name for name in backends if name != "gpu"]
    if len(new_backends) < len(backends):
        new_backends.extend(AMD_GPU_DEFAULT_BACKENDS)

    new_disabled_backends = [name for name in disabled_backends if name != "gpu"]
    if len(new_disabled_backends) < len(disabled_backends):
        new_disabled_backends.extend(AMD_GPU_DEFAULT_BACKENDS)

    new_backend_tags = {
        key: value
        for key, value in backend_tags.items()
        if key not in ["gpu"] + NVIDIA_GPU_BACKENDS
    }

    gpu_backend_tags = backend_tags.get("gpu", [])
    nvidia_tags = []
    for key in gpu_backend_tags:
        if key.startswith("requires-"):
            nvidia_tags.append(key)

    for key in nvidia_tags:
        gpu_backend_tags.remove(key)

    for key in AMD_GPU_DEFAULT_BACKENDS:
        new_backend_tags.setdefault(key, gpu_backend_tags[:])

    for backend in AMD_GPU_DEFAULT_BACKENDS:
        if "cuda-only" not in gpu_backend_tags:
            new_backend_tags[backend].append("requires-gpu-amd")
        new_backend_tags[backend].append("notap")
        new_backend_tags[backend].append("rocm-only")

    return new_backends, new_disabled_backends, new_backend_tags, backend_args

# buildifier: disable=function-docstring
def prepare_gpu_backend_data(backends, disabled_backends, backend_tags, backend_args):
    nvidia_backends = [
        backend
        for backend in backends
        if backend in ["gpu"] + NVIDIA_GPU_BACKENDS
    ]
    amd_backends = [
        backend
        for backend in backends
        if backend in ["gpu"] + AMD_GPU_DEFAULT_BACKENDS
    ]
    other_backends = [
        backend
        for backend in backends
        if backend not in ["gpu"] + NVIDIA_GPU_BACKENDS + AMD_GPU_DEFAULT_BACKENDS
    ]

    nvidia_backends, nvidia_disabled_backends, nvidia_backend_tags, nvidia_backend_args = \
        prepare_nvidia_gpu_backend_data(nvidia_backends, disabled_backends, backend_tags, backend_args)
    amd_backends, amd_disabled_backends, amd_backend_tags, amd_backend_args = \
        prepare_amd_gpu_backend_data(amd_backends, disabled_backends, backend_tags, {})

    new_backends = [
        backend
        for backend in nvidia_backends + amd_backends + other_backends
    ]

    disabled_backends = nvidia_disabled_backends + amd_disabled_backends

    backend_tags = nvidia_backend_tags | amd_backend_tags

    backend_args = nvidia_backend_args | amd_backend_args

    return new_backends, disabled_backends, backend_tags, backend_args

def zkx_test(
        name,
        srcs,
        deps,
        zkx_test_library_deps = [],
        backends = [],
        disabled_backends = DEFAULT_DISABLED_BACKENDS,
        args = [],
        tags = [],
        copts = [],
        data = [],
        backend_tags = {},
        backend_args = {},
        backend_kwargs = {},
        **kwargs):
    """Generates cc_test targets for the given ZKX backends.

    This rule generates a cc_test target for one or more ZKX backends. The arguments
    are identical to cc_test with two additions: 'backends' and 'backend_args'.
    'backends' specifies the backends to generate tests for ("cpu", "gpu"), and
    'backend_args'/'backend_tags' specifies backend-specific args parameters to use
    when generating the cc_test.

    The name of the cc_tests are the provided name argument with the backend name
    appended. For example, if name parameter is "foo_test", then the cpu
    test target will be "foo_test_cpu".

    The build rule also defines a test suite ${name} which includes the tests for
    each of the supported backends.

    Each generated cc_test target has a tag indicating which backend the test is
    for. This tag is of the form "zkx_${BACKEND}" (eg, "zkx_cpu"). These
    tags can be used to gather tests for a particular backend into a test_suite.

    Examples:

      # Generates the targets: foo_test_cpu and foo_test_gpu.
      zkx_test(
          name = "foo_test",
          srcs = ["foo_test.cc"],
          backends = ["cpu", "gpu"],
          deps = [...],
      )

      # Generates the targets: bar_test_cpu and bar_test_gpu. bar_test_cpu
      # includes the additional arg "--special_cpu_flag".
      zkx_test(
          name = "bar_test",
          srcs = ["bar_test.cc"],
          backends = ["cpu", "gpu"],
          backend_args = {"cpu": ["--special_cpu_flag"]}
          deps = [...],
      )

    The build rule defines the preprocessor macro ZKX_TEST_BACKEND_${BACKEND}
    to the value 1 where ${BACKEND} is the uppercase name of the backend.

    Args:
      name: Name of the target.
      srcs: Sources for the target.
      deps: Dependencies of the target.
      zkx_test_library_deps: If set, the generated test targets will depend on the
        respective cc_libraries generated by the zkx_test_library rule.
      backends: A list of backends to generate tests for. Supported values: "cpu",
        "gpu". If this list is empty, the test will be generated for all supported
        backends.
      disabled_backends: A list of backends to NOT generate tests for.
      args: Test arguments for the target.
      tags: Tags for the target.
      copts: Additional copts to pass to the build.
      data: Additional data to pass to the build.
      backend_tags: A dict mapping backend name to list of additional tags to
        use for that target.
      backend_args: A dict mapping backend name to list of additional args to
        use for that target.
      backend_kwargs: A dict mapping backend name to list of additional keyword
        arguments to pass to native.cc_test. Only use for kwargs that don't have a
        dedicated argument, like setting per-backend flaky or timeout attributes.
      **kwargs: Additional keyword arguments to pass to native.cc_test.
    """

    test_names = []
    if not backends:
        backends = _DEFAULT_BACKENDS

    # Expand "gpu" backend name to specific GPU backends and update tags.
    backends, disabled_backends, backend_tags, backend_args = \
        prepare_gpu_backend_data(backends, disabled_backends, backend_tags, backend_args)

    backends = [
        backend
        for backend in backends
        if backend not in disabled_backends
    ]

    for backend in backends:
        test_name = "%s_%s" % (name, backend)
        this_backend_tags = ["zkx_%s" % backend] + tags + backend_tags.get(backend, [])
        this_backend_copts = []
        this_backend_args = backend_args.get(backend, [])
        this_backend_kwargs = dict(kwargs) | backend_kwargs.get(backend, {})
        this_backend_data = []
        backend_deps = []
        if backend == "cpu":
            backend_deps += [
                "//zkx/service:cpu_plugin",
                "//zkx/tests:test_macros_cpu",
            ]

            # TODO: b/382779188 - Remove this when all tests are migrated to PjRt.
            # TODO(chokobole): Uncomment this. Dependency: pjrt_cpu_client_registry
            # if "test_migrated_to_hlo_runner_pjrt" in this_backend_tags:
            # backend_deps.append("//zkx/tests:pjrt_cpu_client_registry")

        elif backend in NVIDIA_GPU_BACKENDS + AMD_GPU_DEFAULT_BACKENDS:
            backend_deps += [
                "//zkx/service:gpu_plugin",
                "//zkx/tests:test_macros_%s" % backend,
            ]
            if backend in NVIDIA_GPU_BACKENDS:
                this_backend_tags += tf_gpu_tests_tags()
            if backend in AMD_GPU_DEFAULT_BACKENDS:
                this_backend_tags.append("gpu")
            this_backend_copts.append("-DZKX_TEST_BACKEND_GPU=1")

            # TODO: b/382779188 - Remove this when all tests are migrated to PjRt.
            # TODO(chokobole): Uncomment this. Dependency: pjrt_gpu_client_registry
            # if "test_migrated_to_hlo_runner_pjrt" in this_backend_tags:
            #     backend_deps.append("//zkx/tests:pjrt_gpu_client_registry")

        elif backend == "interpreter":
            backend_deps += [
                "//zkx/service:interpreter_plugin",
                "//zkx/tests:test_macros_interpreter",
            ]

            # TODO: b/382779188 - Remove this when all tests are migrated to PjRt.
            # TODO(chokobole): Uncomment this. Dependency: pjrt_interpreter_client_registry
            # if "test_migrated_to_hlo_runner_pjrt" in this_backend_tags:
            #     backend_deps.append("//zkx/tests:pjrt_interpreter_client_registry")

        elif backend in plugins:
            backend_deps += plugins[backend]["deps"]
            this_backend_copts += plugins[backend]["copts"]
            this_backend_tags += plugins[backend]["tags"]
            this_backend_args += plugins[backend]["args"]
            this_backend_data += plugins[backend]["data"]
        else:
            fail("Unknown backend %s" % backend)

        if zkx_test_library_deps:
            for lib_dep in zkx_test_library_deps:
                backend_deps += ["%s_%s" % (lib_dep, backend)]  # buildifier: disable=list-append

        zkx_cc_test(
            name = test_name,
            srcs = srcs,
            tags = this_backend_tags,
            copts = copts + ["-DZKX_TEST_BACKEND_%s=1" % backend.upper()] +
                    this_backend_copts,
            args = args + this_backend_args,
            deps = deps + backend_deps,
            data = data + this_backend_data,
            **this_backend_kwargs
        )

        test_names.append(test_name)

    # Notably, a test_suite with `tests = []` is not empty:
    # https://bazel.build/reference/be/general#test_suite_args and the default
    # `tests = []` behavior doesn't respect `--build_tag_filters` due to
    # b/317293391. For this reason, if we would create an empty `test_suite`,
    # instead create a `cc_test` with no srcs that links against `main` to have
    # more predictable behavior that avoids bugs.
    #
    # Due to b/317293391, we also mark the test suite `manual`, so that wild card builds
    # like in the ZKX CI won't try to build the test suite target. Instead the wild card
    # build will build the individual test targets and therefore respect the tags on each
    # individual test target.
    # Example: Assume we have an `zkx_test(name=my_test)` in `//zkx/service/gpu` with backends `cpu`
    # and `gpu`. This generates two test targets `//zkx/service/gpu:my_test_{cpu|gpu}`. The latter
    # has a tag `gpu`.
    #
    # - `bazel test --test_tag_filters=-gpu //zkx/service/gpu/...` will only run the cpu test.
    # - `bazel test //zkx/service/gpu/...` will run both tests.
    # - `bazel test //zkx/service/gpu:my_test` will run both tests.
    # Caveat:
    # - `bazel test --test_tag_filters=-gpu //zkx/service/gpu:my_test` will run both tests and
    #   not respect the tag filter - but it's way better than the previous behavior.
    if test_names:
        native.test_suite(name = name, tags = tags + ["manual"], tests = test_names)
    else:
        cc_test(name = name, deps = ["@com_google_googletest//:gtest_main"])

def generate_backend_test_macros(backends = []):  # buildifier: disable=unnamed-macro
    """Generates test_macro libraries for each backend with correct options.

    Args:
      backends: The list of backends to generate libraries for.
    """
    if not backends:
        backends = _ALL_BACKENDS
    for backend in backends:
        manifest = ""
        if backend in plugins:
            manifest = plugins[backend]["disabled_manifest"]
        cc_library(
            name = "test_macros_%s" % backend,
            testonly = True,
            srcs = ["test_macros.cc"],
            hdrs = ["test_macros.h"],
            copts = [
                "-DZKX_PLATFORM=\\\"%s\\\"" % backend.upper(),
                "-DZKX_DISABLED_MANIFEST=\\\"%s\\\"" % manifest,
            ],
            deps = [
                "@com_google_absl//absl/log",
            ],
        )
