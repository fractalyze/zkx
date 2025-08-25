"""Open source build configurations for CUDA."""

# Constructs rpath linker flags for use with nvidia wheel-packaged libs
# available from PyPI. Two paths are needed because symbols are used from
# both the root of the TensorFlow installation directory as well as from
# various pywrap libs within the 'python' subdir.
def cuda_rpath_flags(relpath):
    return [
        "-Wl,-rpath='$$ORIGIN/../../" + relpath + "'",
        "-Wl,-rpath='$$ORIGIN/../" + relpath + "'",
    ]
