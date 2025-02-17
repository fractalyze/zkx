load("@zkx//third_party/absl:workspace.bzl", absl = "repo")
load("@zkx//third_party/llvm:workspace.bzl", llvm = "repo")

def zkx_deps():
    absl()

    # Load the raw llvm-project.  llvm does not have build rules set up by default,
    # but provides a script for setting up build rules via overlays.
    llvm("llvm-raw")
