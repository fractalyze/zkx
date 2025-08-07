def if_cuda_libs(if_true, if_false = []):  # buildifier: disable=unused-variable
    """Shorthand for select()'ing on whether we need to include hermetic CUDA libraries."""
    return select({"@local_config_cuda//cuda:cuda_tools_and_libs": if_true, "//conditions:default": if_false})  # copybara:comment_replace return if_false
