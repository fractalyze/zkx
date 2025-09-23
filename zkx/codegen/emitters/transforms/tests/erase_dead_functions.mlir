// RUN: emitters_opt %s -zkx-erase-dead-functions | FileCheck %s

// Test: Only live (reachable) functions are kept.

// Entry function, should be kept.
func.func @main() -> i32 {
  %0 = call @live_func() : () -> i32
  return %0 : i32
}

// Called by @main, should be kept.
func.func private @live_func() -> i32 {
  %c = arith.constant 42 : i32
  return %c : i32
}

// Not called by anyone, should be erased.
func.func private @dead_func() -> i32 {
  %c = arith.constant 0 : i32
  return %c : i32
}

// Not called by anyone, should be erased.
func.func private @another_dead_func() -> i32 {
  %c = arith.constant 1 : i32
  return %c : i32
}

// CHECK-LABEL: func.func @main()
// CHECK: call @live_func
// CHECK-NOT: @dead_func
// CHECK-NOT: @another_dead_func
// CHECK: func.func private @live_func

// -----

// Test: Public functions are considered live even if not called.

func.func @public_func() -> i32 {
  %c = arith.constant 7 : i32
  return %c : i32
}

// Not called, private, should be erased.
func.func private @private_dead() -> i32 {
  %c = arith.constant 8 : i32
  return %c : i32
}

// CHECK-LABEL: func.func @public_func
// CHECK-NOT: @private_dead

// -----

// Test: Chain of calls, all reachable should be kept.

func.func @entry() -> i32 {
  %0 = call @mid() : () -> i32
  return %0 : i32
}

func.func private @mid() -> i32 {
  %0 = call @leaf() : () -> i32
  return %0 : i32
}

func.func private @leaf() -> i32 {
  %c = arith.constant 99 : i32
  return %c : i32
}

// Unreachable, should be erased.
func.func private @unreachable() -> i32 {
  %c = arith.constant 123 : i32
  return %c : i32
}

// CHECK-LABEL: func.func @entry
// CHECK: call @mid
// CHECK: func.func private @mid
// CHECK: call @leaf
// CHECK: func.func private @leaf
// CHECK-NOT: @unreachable
