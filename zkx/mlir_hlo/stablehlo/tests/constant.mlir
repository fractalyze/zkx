// RUN: stablehlo-opt %s -split-input-file | FileCheck %s

func.func private @constant_with_i32(%x: tensor<15xi32>) -> tensor<i32> {
  %0 = stablehlo.constant dense<0> : tensor<i32>
  return %0 : tensor<i32>
}

// CHECK: @constant_with_i32

// -----

!pf_babybear = !field.pf<2013265921 : i32, true>

func.func @constant_with_babybear() -> tensor<!pf_babybear> {
  %0 = stablehlo.constant dense<0> : tensor<!pf_babybear>
  return %0 : tensor<!pf_babybear>
}

// CHECK: @constant_with_babybear
