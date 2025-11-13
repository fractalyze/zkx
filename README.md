# ZKX: Zero Knowledge Accelerator

ZKX is a computation framework inspired by
[XLA](https://github.com/openxla/xla). It compiles ZK-specific high-level
operations into efficient low-level code using
[ZKIR](https://github.com/fractalyze/zkir)￼as its intermediate representation.
ZKX is optimized for CPUs and GPUs, with plans to extend support to specialized
ZK hardware for greater performance and portability.

## Prerequisite

1. Follow the [bazel installation guide](https://bazel.build/install).

## Build instructions

1. Clone the ZKX repo

   ```sh
   git clone https://github.com/fractalyze/zkx
   ```

1. Build ZKX

   ```sh
   bazel build //...
   ```

1. Test ZKX

   ```sh
   bazel test //...
   ```

## Community

Building a substantial ZK compiler requires collaboration across the broader ZK
ecosystem — and we’d love your help in shaping ZKX. See
[CONTRIBUTING.md](https://github.com/fractalyze/.github/blob/main/CONTRIBUTING.md)
for more details.

We use GitHub Issues and Pull Requests to coordinate development, and
longer-form discussions take place in the
[zkx-discuss](https://github.com/fractalyze/zkx/discussions).
