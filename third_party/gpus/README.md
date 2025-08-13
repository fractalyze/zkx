# Gloo

This is taken and modified from [xla](https://github.com/openxla/xla/tree/8bac4a2/third_party/tsl/third_party/gpus).

```shell
> diff -r /path/to/openxla/xla/third_party/tsl/third_party/gpus third_party/gpus
Only in third_party/gpus: README.md
diff --color -r /path/to/openxla/xla/third_party/tsl/third_party/gpus/check_cuda_libs.py third_party/gpus/check_cuda_libs.py
0a1,2
> #!/usr/bin/env python3
>
68c70
<     args = [argv for argv in sys.argv[1:]]
---
>     args = sys.argv[1:]
76d77
<     # pylint: disable=superfluous-parens
78d78
<     # pylint: enable=superfluous-parens
diff --color -r /path/to/openxla/xla/third_party/tsl/third_party/gpus/compiler_common_tools.bzl third_party/gpus/compiler_common_tools.bzl
170,174c170,174
<     return includes_cpp + [
<         inc
<         for inc in includes_c
<         if inc not in includes_cpp
<     ]
---
>     includes_cpp_set = {i: True for i in includes_cpp}
>     for inc in includes_c:
>         if inc not in includes_cpp_set:
>             includes_cpp.append(inc)
>     return includes_cpp
diff --color -r /path/to/openxla/xla/third_party/tsl/third_party/gpus/crosstool/clang/bin/crosstool_wrapper_driver_is_not_gcc.tpl third_party/gpus/crosstool/clang/bin/crosstool_wrapper_driver_is_not_gcc.tpl
227c227
<                re.search('\.cpp$|\.cc$|\.c$|\.cxx$|\.C$', f)]
---
>                re.search(r'\.(cpp|cc|c|cxx|C)$', f)]
260c260
<   # NVCC+clang compilers.
---
>   # NVCC+clang compilers.
diff --color -r /path/to/openxla/xla/third_party/tsl/third_party/gpus/crosstool/clang/bin/crosstool_wrapper_driver_rocm.tpl third_party/gpus/crosstool/clang/bin/crosstool_wrapper_driver_rocm.tpl
1c1
< #!/usr/bin/env python
---
> #!/usr/bin/env python3
23c23,27
< import pipes
---
> # NOTE(chokobole): Gemini suggested replacing `pipes` (deprecated since Python 3.3,
> # removed in Python 3.9) with `shlex`, which is already imported in other scripts
> # in this PR.
> # See: https://github.com/zk-rabbit/zkx/pull/32#discussion_r2268367251
> import shlex
166c170
<                re.search('\.cpp$|\.cc$|\.c$|\.cxx$|\.C$', f)]
---
>                re.search(r'\.(cpp|cc|c|cxx|C)$', f)]
231c235
<     leftover = [pipes.quote(s) for s in leftover]
---
>     leftover = [shlex.quote(s) for s in leftover]
237c241
<     # with hipcc compiler invoked with -fno-gpu-rdc by default now, it's ok to
---
>     # with hipcc compiler invoked with -fno-gpu-rdc by default now, it's ok to
diff --color -r /path/to/openxla/xla/third_party/tsl/third_party/gpus/crosstool/clang/bin/crosstool_wrapper_driver_sycl.tpl third_party/gpus/crosstool/clang/bin/crosstool_wrapper_driver_sycl.tpl
1c1
< #!/usr/bin/env python
---
> #!/usr/bin/env python3
58c58,60
<   parser.add_argument('-sycl_compile', action='store_true')
---
>   # NOTE(chokobole): Gemini said that we don't use this flag.
>   # See https://github.com/zk-rabbit/zkx/pull/32#discussion_r2268367247.
>   # parser.add_argument('-sycl_compile', action='store_true')
diff --color -r /path/to/openxla/xla/third_party/tsl/third_party/gpus/crosstool/windows/msvc_wrapper_for_nvcc.py.tpl third_party/gpus/crosstool/windows/msvc_wrapper_for_nvcc.py.tpl
1c1
< #!/usr/bin/env python
---
> #!/usr/bin/env python3
105c105
<                re.search('\.cpp$|\.cc$|\.c$|\.cxx$|\.C$', f)]
---
>                re.search(r'\.(cpp|cc|c|cxx|C)$', f)]
107c107
<     raise Error('No source files found for cuda compilation.')
---
>     raise Exception('No source files found for cuda compilation.')
111c111
<     raise Error('Please specify exactly one output file for cuda compilation.')
---
>     raise Exception('Please specify exactly one output file for cuda compilation.')
199c199
<                           shell=True)
---
>                           shell=False)
256d255
<   return subprocess.call([CPU_COMPILER] + cpu_compiler_flags)
diff --color -r /path/to/openxla/xla/third_party/tsl/third_party/gpus/cuda/BUILD.tpl third_party/gpus/cuda/BUILD.tpl
162,167c162,168
< cc_library(
<     name = "cudnn",
<     srcs = ["cuda/lib/%{cudnn_lib}"],
<     data = ["cuda/lib/%{cudnn_lib}"],
<     linkstatic = 1,
< )
---
> # TODO(chokobole): Uncomment this when we need cuDNN.
> # cc_library(
> #     name = "cudnn",
> #     srcs = ["cuda/lib/%{cudnn_lib}"],
> #     data = ["cuda/lib/%{cudnn_lib}"],
> #     linkstatic = 1,
> # )
169,175c170,176
< cc_library(
<     name = "cudnn_header",
<     hdrs = [":cudnn-include"],
<     include_prefix = "third_party/gpus/cudnn",
<     strip_include_prefix = "cudnn/include",
<     deps = [":cuda_headers"],
< )
---
> # cc_library(
> #     name = "cudnn_header",
> #     hdrs = [":cudnn-include"],
> #     include_prefix = "third_party/gpus/cudnn",
> #     strip_include_prefix = "cudnn/include",
> #     deps = [":cuda_headers"],
> # )
198c199,200
<         ":cudnn",
---
>         # TODO(chokobole): Uncomment this when we need cuDNN.
>         # ":cudnn",
diff --color -r /path/to/openxla/xla/third_party/tsl/third_party/gpus/cuda/BUILD.windows.tpl third_party/gpus/cuda/BUILD.windows.tpl
152,156c152,157
< cc_import(
<     name = "cudnn",
<     interface_library = "cuda/lib/%{cudnn_lib}",
<     system_provided = 1,
< )
---
> # TODO(chokobole): Uncomment this when we need cuDNN.
> # cc_import(
> #     name = "cudnn",
> #     interface_library = "cuda/lib/%{cudnn_lib}",
> #     system_provided = 1,
> # )
158,164c159,165
< cc_library(
<     name = "cudnn_header",
<     hdrs = [":cudnn-include"],
<     include_prefix = "third_party/gpus/cudnn",
<     strip_include_prefix = "cudnn/include",
<     deps = [":cuda_headers"],
< )
---
> # cc_library(
> #     name = "cudnn_header",
> #     hdrs = [":cudnn-include"],
> #     include_prefix = "third_party/gpus/cudnn",
> #     strip_include_prefix = "cudnn/include",
> #     deps = [":cuda_headers"],
> # )
185c186,187
<         ":cudnn",
---
>         # TODO(chokobole): Uncomment this when we need cuDNN.
>         # ":cudnn",
diff --color -r /path/to/openxla/xla/third_party/tsl/third_party/gpus/cuda/hermetic/BUILD.tpl third_party/gpus/cuda/hermetic/BUILD.tpl
180,183c180,184
< alias(
<   name = "cudnn",
<   actual = "@cuda_cudnn//:cudnn",
< )
---
> # TODO(chokobole): Uncomment this when we need cuDNN.
> # alias(
> #   name = "cudnn",
> #   actual = "@cuda_cudnn//:cudnn",
> # )
185,188c186,189
< alias(
<   name = "cudnn_header",
<   actual = "@cuda_cudnn//:headers",
< )
---
> # alias(
> #   name = "cudnn_header",
> #   actual = "@cuda_cudnn//:headers",
> # )
207c208,209
<         ":cudnn",
---
>         # TODO(chokobole): Uncomment this when we need cuDNN.
>         # ":cudnn",
diff --color -r /path/to/openxla/xla/third_party/tsl/third_party/gpus/cuda/hermetic/cuda_configure.bzl third_party/gpus/cuda/hermetic/cuda_configure.bzl
16c16
<   * `HERMETIC_CUDA_COMPUTE_CAPABILITIES`: The CUDA compute capabilities. Default
---
>   * `HERMETIC_CUDA_COMPUTE_CAPABILITIES`: The CUDA compute capabilities. Default
219c219,220
<         cudnn_version = repository_ctx.read(repository_ctx.attr.cudnn_version),
---
>         # TODO(chokobole): Uncomment this when we need cuDNN.
>         # cudnn_version = repository_ctx.read(repository_ctx.attr.cudnn_version),
254c255,256
<         repository_ctx.attr.cudnn_version,
---
>         # TODO(chokobole): Uncomment this when we need cuDNN.
>         # repository_ctx.attr.cudnn_version,
299a302,307
>         # NOTE(chokobole):Extract the repo name from the label string, since .repo_name is not available in 7.0.2.
>         nvcc_label_str = str(repository_ctx.attr.nvcc_binary)
>         if nvcc_label_str.startswith("@"):
>             nvcc_repo_name = nvcc_label_str[1:].split("//", 1)[0]
>         else:
>             nvcc_repo_name = ""
305c313
<             nvcc_archive = repository_ctx.attr.nvcc_binary.repo_name,
---
>             nvcc_archive = nvcc_repo_name,
420c428,429
<             "%{cudnn_version}": "",
---
>             # TODO(chokobole): Uncomment this when we need cuDNN.
>             # "%{cudnn_version}": "",
493c502,503
<             "%{cudnn_version}": cuda_config.cudnn_version,
---
>             # TODO(chokobole): Uncomment this when we need cuDNN.
>             # "%{cudnn_version}": cuda_config.cudnn_version,
509c519,520
<             "cudnn_version": cuda_config.cudnn_version,
---
>             # TODO(chokobole): Uncomment this when we need cuDNN.
>             # "cudnn_version": cuda_config.cudnn_version,
552c563,564
<     "LOCAL_CUDNN_PATH",
---
>     # TODO(chokobole): Uncomment this when we need cuDNN.
>     # "LOCAL_CUDNN_PATH",
563c575,576
<         "cudnn_version": attr.label(default = Label("@cuda_cudnn//:version.txt")),
---
>         # TODO(chokobole): Uncomment this when we need cuDNN.
>         # "cudnn_version": attr.label(default = Label("@cuda_cudnn//:version.txt")),
diff --color -r /path/to/openxla/xla/third_party/tsl/third_party/gpus/cuda/hermetic/cuda_cublas.BUILD.tpl third_party/gpus/cuda/hermetic/cuda_cublas.BUILD.tpl
3c3
<     "@xla//xla/tsl/platform/default:cuda_build_defs.bzl",
---
>     "@zkx//xla/tsl/platform:cuda_build_defs.bzl",
diff --color -r /path/to/openxla/xla/third_party/tsl/third_party/gpus/cuda/hermetic/cuda_cudart.BUILD.tpl third_party/gpus/cuda/hermetic/cuda_cudart.BUILD.tpl
3c3
<     "@xla//xla/tsl/platform/default:cuda_build_defs.bzl",
---
>     "@zkx//xla/tsl/platform:cuda_build_defs.bzl",
diff --color -r /path/to/openxla/xla/third_party/tsl/third_party/gpus/cuda/hermetic/cuda_cudnn.BUILD.tpl third_party/gpus/cuda/hermetic/cuda_cudnn.BUILD.tpl
3c3
<     "@xla//xla/tsl/platform/default:cuda_build_defs.bzl",
---
>     "@zkx//xla/tsl/platform:cuda_build_defs.bzl",
diff --color -r /path/to/openxla/xla/third_party/tsl/third_party/gpus/cuda/hermetic/cuda_cudnn9.BUILD.tpl third_party/gpus/cuda/hermetic/cuda_cudnn9.BUILD.tpl
3c3
<     "@xla//xla/tsl/platform/default:cuda_build_defs.bzl",
---
>     "@zkx//xla/tsl/platform:cuda_build_defs.bzl",
11c11
< cc_import(
---
> cc_import(
17c17
< cc_import(
---
> cc_import(
23c23
< cc_import(
---
> cc_import(
29c29
< cc_import(
---
> cc_import(
diff --color -r /path/to/openxla/xla/third_party/tsl/third_party/gpus/cuda/hermetic/cuda_cufft.BUILD.tpl third_party/gpus/cuda/hermetic/cuda_cufft.BUILD.tpl
3c3
<     "@xla//xla/tsl/platform/default:cuda_build_defs.bzl",
---
>     "@zkx//xla/tsl/platform:cuda_build_defs.bzl",
27c27
<         %{comment}"include/cudalibxt.h",
---
>         %{comment}"include/cudalibxt.h",
diff --color -r /path/to/openxla/xla/third_party/tsl/third_party/gpus/cuda/hermetic/cuda_cupti.BUILD.tpl third_party/gpus/cuda/hermetic/cuda_cupti.BUILD.tpl
4c4
<     "@xla//xla/tsl/platform/default:cuda_build_defs.bzl",
---
>     "@zkx//xla/tsl/platform:cuda_build_defs.bzl",
diff --color -r /path/to/openxla/xla/third_party/tsl/third_party/gpus/cuda/hermetic/cuda_curand.BUILD.tpl third_party/gpus/cuda/hermetic/cuda_curand.BUILD.tpl
3c3
<     "@xla//xla/tsl/platform/default:cuda_build_defs.bzl",
---
>     "@zkx//xla/tsl/platform:cuda_build_defs.bzl",
diff --color -r /path/to/openxla/xla/third_party/tsl/third_party/gpus/cuda/hermetic/cuda_cusolver.BUILD.tpl third_party/gpus/cuda/hermetic/cuda_cusolver.BUILD.tpl
3c3
<     "@xla//xla/tsl/platform/default:cuda_build_defs.bzl",
---
>     "@zkx//xla/tsl/platform:cuda_build_defs.bzl",
diff --color -r /path/to/openxla/xla/third_party/tsl/third_party/gpus/cuda/hermetic/cuda_cusparse.BUILD.tpl third_party/gpus/cuda/hermetic/cuda_cusparse.BUILD.tpl
3c3
<     "@xla//xla/tsl/platform/default:cuda_build_defs.bzl",
---
>     "@zkx//xla/tsl/platform:cuda_build_defs.bzl",
diff --color -r /path/to/openxla/xla/third_party/tsl/third_party/gpus/cuda/hermetic/cuda_nvjitlink.BUILD.tpl third_party/gpus/cuda/hermetic/cuda_nvjitlink.BUILD.tpl
3c3
<     "@xla//xla/tsl/platform/default:cuda_build_defs.bzl",
---
>     "@zkx//xla/tsl/platform:cuda_build_defs.bzl",
32d31
<
diff --color -r /path/to/openxla/xla/third_party/tsl/third_party/gpus/cuda/hermetic/cuda_nvrtc.BUILD.tpl third_party/gpus/cuda/hermetic/cuda_nvrtc.BUILD.tpl
3c3
<     "@xla//xla/tsl/platform/default:cuda_build_defs.bzl",
---
>     "@zkx//xla/tsl/platform:cuda_build_defs.bzl",
Only in /path/to/openxla/xla/third_party/tsl/third_party/gpus: cuda_configure.bzl
diff --color -r /path/to/openxla/xla/third_party/tsl/third_party/gpus/find_cuda_config.py third_party/gpus/find_cuda_config.py
589a590
>
Only in /path/to/openxla/xla/third_party/tsl/third_party/gpus: find_rocm_config.py
Only in /path/to/openxla/xla/third_party/tsl/third_party/gpus: find_sycl_config.py
Only in /path/to/openxla/xla/third_party/tsl/third_party/gpus: rocm
Only in /path/to/openxla/xla/third_party/tsl/third_party/gpus: rocm_configure.bzl
Only in /path/to/openxla/xla/third_party/tsl/third_party/gpus: sycl
Only in /path/to/openxla/xla/third_party/tsl/third_party/gpus: sycl_configure.bzl
```
