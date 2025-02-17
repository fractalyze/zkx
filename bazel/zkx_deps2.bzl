load("@zkx//third_party/llvm:setup.bzl", "llvm_setup")

def zkx_deps2():
    # Load the raw llvm-project.  llvm does not have build rules set up by default,
    # but provides a script for setting up build rules via overlays.
    llvm_setup(name = "llvm-project")
