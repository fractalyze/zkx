# Py

This is taken and modified from [xla](https://github.com/openxla/xla/tree/8bac4a2/third_party/py).

```shell
> diff -r /path/to/openxla/xla/third_party/py third_party/py
1,60d0
< load("@bazel_skylib//rules:common_settings.bzl", "bool_flag")
< load("@python//:defs.bzl", "compile_pip_requirements")
< load("@python_version_repo//:py_version.bzl", "REQUIREMENTS")
<
< compile_pip_requirements(
<     name = "requirements",
<     extra_args = [
<         "--allow-unsafe",
<         "--build-isolation",
<     ],
<     generate_hashes = True,
<     requirements_in = "requirements.in",
<     requirements_txt = REQUIREMENTS,
< )
<
< compile_pip_requirements(
<     name = "requirements_nightly",
<     data = ["test-requirements.txt"],
<     extra_args = [
<         "--allow-unsafe",
<         "--build-isolation",
<         "--extra-index-url=https://pypi.anaconda.org/scientific-python-nightly-wheels/simple",
<         "--pre",
<         "--upgrade",
<     ],
<     generate_hashes = False,
<     requirements_in = "requirements.in",
<     requirements_txt = REQUIREMENTS,
< )
<
< compile_pip_requirements(
<     name = "requirements_dev",
<     extra_args = [
<         "--allow-unsafe",
<         "--build-isolation",
<         "--upgrade",
<     ],
<     generate_hashes = False,
<     requirements_in = "requirements.in",
<     requirements_txt = REQUIREMENTS,
< )
<
< # Flag indicating if the target requires pre-built wheel.
< bool_flag(
<     name = "wheel_dependency",
<     build_setting_default = False,
< )
<
< config_setting(
<     name = "enable_wheel_dependency",
<     flag_values = {
<         ":wheel_dependency": "True",
<     },
< )
<
< filegroup(
<     name = "manylinux_compliance_test",
<     srcs = ["manylinux_compliance_test.py"],
<     visibility = ["//visibility:public"],
< )
diff --color -r /home/ryan/Workspace/xla/third_party/py/BUILD.tpl third_party/py/BUILD.tpl
39c39
< # This alias is exists for the use of targets in the @llvm-project dependency,
---
> # This alias exists for the use of targets in the @llvm-project dependency,
48d47
<
53c52
< )
\ No newline at end of file
---
> )
Only in /home/ryan/Workspace/xla/third_party/py: ml_dtypes
Only in /home/ryan/Workspace/xla/third_party/py: numpy
diff --color -r /home/ryan/Workspace/xla/third_party/py/python_configure.bzl third_party/py/python_configure.bzl
56,57c56
< )
< """Detects and configures the local Python.
---
>     doc = """Detects and configures the local Python.
67c66,67
< """
---
> """,
> )
diff --color -r /home/ryan/Workspace/xla/third_party/py/python_repo.bzl third_party/py/python_repo.bzl
375c375
<                 # By default we assume Linux x86_64 architecture, eplace with
---
>                 # By default we assume Linux x86_64 architecture, replace with
Only in third_party/py: README.md
```
