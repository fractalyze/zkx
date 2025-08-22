def if_libnvptxcompiler_support_enabled(a, b = []):
    return select({
        "//zkx/stream_executor/cuda:libnvptxcompiler_support_enabled": a,
        "//conditions:default": b,
    })
