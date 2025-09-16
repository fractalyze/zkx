# ZKX Style Guide

## Introduction

This document defines the coding standards for C++ code in ZKX. The base
guideline is the [Google C++ Style Guide], combined with the
[Angular Commit Convention], with explicit project-specific modifications. In
addition to code style, this guide incorporates our rules for commit messages,
pull requests, and IDE/editor setup.

_Note: Imported code from XLA may not fully follow these rules and should
generally remain unchanged._

______________________________________________________________________

## Core Principles

- **Readability:** Both code and commits should be immediately understandable.
- **Maintainability:** Code should be easy to refactor and extend.
- **Consistency:** Apply the same conventions across files and modules, except
  where external code (e.g., XLA) is imported.
- **Performance:** Prioritize clarity, but optimize carefully where latency and
  cost are critical.

______________________________________________________________________

## C++ Coding Style

The following are project-specific deviations and clarifications from the
[Google C++ Style Guide].

### Static Methods

- For **static methods** implemented in `.cc` files, explicitly annotate with
  `// static`.

  ```c++
  // static
  uint64_t EnvTime::NowNanos() {
    // ...
  }
  ```

### File-Scoped Symbols

- Wrap **file-scoped functions, constants, and variables** inside an **anonymous
  namespace**.

  ```c++
  namespace {

  constexpr int kBufferSize = 1024;

  void HelperFunction() {
    // ...
  }

  }  // namespace
  ```

### Abseil

- Prefer **`std::string_view`** instead of `absl::string_view`.

### Header Inclusion

- **Avoid redundant includes**: Do not repeat headers in `.cc` files that are
  already included in the corresponding `.h`.

  ```c++
  // in a.h
  #include <stdint.h>

  // in a.cc
  #include "a.h"
  // #include <stdint.h>  // ❌ redundant
  ```

- **Include only required headers**. Remove unused includes.

### Raw Pointer Ownership

- When using a **raw pointer** (`T*`) in **class or struct members**, explicitly
  document ownership by adding an inline comment `// not owned` or `// owned`.
- Prefer `std::unique_ptr` or `std::shared_ptr` for owned resources.

Example:

```c++
class Prover {
 public:
  explicit Prover(Context* ctx) : ctx_(ctx) {}

 private:
  Context* ctx_; // not owned
  std::unique_ptr<Engine> engine_;
};
```

### Static Lifetime (Teardown-Sensitive Resources)

Some runtimes (e.g., **CUDA drivers**) are **sensitive to destruction order** at
program shutdown. If their destructors run after or before certain subsystems,
this can cause **deadlocks or crashes**.

In these cases, we intentionally avoid destruction and use **static lifetime
patterns**.

#### Allowed Patterns

- **Trivially destructible types** (per [Google C++ Style Guide]).

  - Examples: pointers, integers, arrays of trivially destructible types,
    `constexpr` objects.

- **Function-local static references to heap objects**

  - Pattern: `static T& t = *new T(...);`
  - This avoids destruction at shutdown by intentionally leaking the object.
  - Use **`absl::IgnoreLeak`** to clearly communicate intentional
    process-lifetime leaks and to prevent false positives in leak detectors
    (e.g., LeakSanitizer / ASan’s leak detection, Valgrind memcheck).
  - This ensures CI runs with sanitizers (e.g., ASAN_OPTIONS=detect_leaks=1) do
    not fail on approved, process-lifetime singletons.

  ```c++
  static Foo& GetFoo() {
    static Foo* foo = absl::IgnoreLeak(new Foo(...));
    return *foo;
  }
  ```

- **absl::NoDestructor<T>**

  - For cases where the object must live for the lifetime of the process.
  - Guarantees thread-safe initialization and avoids teardown-order issues.

  ```c++
  #include "absl/base/no_destructor.h"

  // A teardown-sensitive background executor.
  // Destruction order at process exit is hard to control and can crash/hang
  // if other subsystems (logging, metrics, OS handles) go away first.
  class BackgroundExecutor {
  public:
    BackgroundExecutor();
    ~BackgroundExecutor();                 // non-trivial destructor (threads, queues)
    void Post(std::function<void()> fn);   // schedules work
    // ...
  };

  // Use a process-lifetime singleton to avoid destructor at shutdown.
  BackgroundExecutor* GetBackgroundExecutor() {
    // NOTE: Avoid destruction to prevent teardown-order issues at process exit.
    static absl::NoDestructor<BackgroundExecutor> exec;
    return &*exec;
  }
  ```

#### Forbidden Patterns

- **Static/global objects with non-trivial destructors**

  ```c++
  // ❌ Bad: destructor will run at shutdown, order is undefined.
  static std::map<int, int> kData = {{1, 0}, {2, 0}};
  ```

- **Lifetime-extended temporaries with non-trivial destructors**

  ```c++
  // ❌ Bad
  const std::string kFoo = "foo";
  const std::string& kBar = StrCat("a", "b", "c");
  ```

## Naming & Migration Hygiene (XLA → ZKX)

- All **new identifiers** (namespaces, classes, variables, include guards, Bazel
  packages) must use **ZKX**, not `XLA`.
- Code migrated from `XLA` should be renamed where practical, except where it
  breaks compatibility.
- PR checklist: ensure no new identifiers introduce `XLA`.

### XLA Compatibility Exceptions

The following usages of `XLA` are explicitly allowed:

1. **License headers**

   - Example:

     ```c++
     // Copyright 2017 The OpenXLA Authors.
     ```

1. **Includes from XLA/TSL paths**

   ```c++
   #include "xla/tsl/platform/env.h"
   #include "xla/tsl/platform/env_time.h"
   ```

1. **Header guard**

   ```c++
   #ifndef XLA_TSL_PLATFORM_CUDA_ROOT_PATH_H_
   #define XLA_TSL_PLATFORM_CUDA_ROOT_PATH_H_

   #endif XLA_TSL_PLATFORM_CUDA_ROOT_PATH_H_
   ```

1. **Bazel dependencies/labels**

   ```bazel
   deps = [
       "//xla/tsl/platform:env",
   ]
   ```

1. **Widely used macros**: Macros starting with the `TF_` prefix are allowed
   as-is, since they are part of the established XLA/TSL API surface.

   - `TF_RETURN_IF_ERROR`
   - `TF_ASSIGN_OR_RETURN`
   - ...

1. **Explicit external references** in comments or documentation (e.g., “ported
   from XLA …”).

_Disallowed:_ Introducing new identifiers with `XLA` inside ZKX code, except in
vendored or third-party files that keep upstream names.

______________________________________________________________________

## Comment Style

- Non-trivial code changes must be accompanied by comments.

- Comments explain **why** a change or design decision was made or explain the
  code for better readability.

- Use full sentences with proper punctuation.

- Add the lint type to `NOLINT` comments

  ```c++
  #include "farmhash.h"  // NOLINT(build/include_subdir)
  ```

- Do not use **double spaces** in comments. Always use a **single space** after
  periods.

  ```c++
  // ✅ Correct: This is a proper comment. It follows the rule.
  // ❌ Wrong: This is an improper comment.  It has double spaces.
  ```

### Dependency TODO Comments

- TODO comments that include **`Dependency:`** (used to mark code temporarily
  disabled or pending due to upstream XLA dependencies) must remain **inline on
  a single line**.
- If `clang-format` attempts to wrap such a comment, enforce inline formatting
  with `clang-format off/on`.

```c++
// clang-format off
// TODO(chokobole): Uncomment this. Dependency: Something
// clang-format on
```

______________________________________________________________________

## Bazel Style

- Every header included in a Bazel target must also be declared as a Bazel
  dependency.

______________________________________________________________________

## Testing

- **Framework**: Use gtest/gmock.
- **Coverage**: New features must include tests whenever applicable.
- **Completeness**: Always include boundary cases and error paths.
- **Determinism**: Tests must be deterministic and runnable independently (no
  hidden state dependencies).
- **Performance**: Add benchmarks for performance-critical code paths when
  appropriate.

______________________________________________________________________

## Collaboration Rules

### Commits (Angular Commit Convention)

- Must follow the [Commit Message Guideline].

- Format:

  ```
  <type>(<scope>): <summary>
  ```

  where `type` ∈ {build, chore, ci, docs, feat, fix, perf, refactor, style,
  test}.

- Commit body: explain **why** the change was made (minimum 20 characters).

- Footer: record breaking changes, deprecations, and related issues/PRs.

- Each commit must include only **minimal, logically related changes**. Avoid
  mixing style fixes with functional changes.

### Pull Requests

- Follow the [Pull Request Guideline].
- Commits must be **atomic** and independently buildable/testable.
- Provide context and links (short SHA for external references).

### File Formatting

- Every file must end with a single newline.
- No trailing whitespace.
- No extra blank lines at EOF.

______________________________________________________________________

## Tooling

- **Formatter:** `clang-format` (Google preset with project overrides). Refer to
  the [.clang-format] file in the repo.
- **Linter:** `clang-tidy`.
- **Pre-commit hooks:** Recommended for enforcing format and lint locally.
- **CI:** All PRs must pass lint, format, and tests before merge.

[.clang-format]: /.clang-format
[angular commit convention]: https://github.com/angular/angular/blob/main/contributing-docs/commit-message-guidelines.md
[commit message guideline]: https://github.com/zk-rabbit/.github/blob/main/COMMIT_MESSAGE_GUIDELINE.md
[google c++ style guide]: https://google.github.io/styleguide/cppguide.html
[pull request guideline]: https://github.com/zk-rabbit/.github/blob/main/PULL_REQUEST_GUIDELINE.md
