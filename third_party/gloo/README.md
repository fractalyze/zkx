# Gloo

This is taken and modified from [xla](https://github.com/openxla/xla/tree/8bac4a2/third_party/gloo).

```shell
> diff -r /path/to/openxla/xla/third_party/gloo third_party/gloo
Only in third_party/gloo: README.md
diff --color -r /path/to/openxla/xla/third_party/gloo/gloo.BUILD third_party/gloo/gloo.BUILD
60,61c60,61
<         "@xla//xla/tsl:macos": [],
<         "@xla//xla/tsl:windows": [],
---
>         "@platforms//os:macos": [],
>         "@platforms//os:windows": [],
```
