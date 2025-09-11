"""TSL rules."""

# These configs are used to determine whether we should use CUDA tools and libs in cc_libraries.
# They are intended for the OSS builds only.
def if_cuda_tools(if_true, if_false = []):  # buildifier: disable=unused-variable
    """Shorthand for select()'ing on whether we're building with hCUDA tools."""
    return select({"@local_config_cuda//cuda:cuda_tools": if_true, "//conditions:default": if_false})  # copybara:comment_replace return if_false

def if_cuda_libs(if_true, if_false = []):  # buildifier: disable=unused-variable
    """Shorthand for select()'ing on whether we need to include hermetic CUDA libraries."""
    return select({"@local_config_cuda//cuda:cuda_tools_and_libs": if_true, "//conditions:default": if_false})  # copybara:comment_replace return if_false
