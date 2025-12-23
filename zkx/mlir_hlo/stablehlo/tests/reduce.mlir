// RUN: stablehlo-opt %s | FileCheck %s

!pf_babybear = !field.pf<2013265921 : i32, true>

func.func @reduce_sum(%x: tensor<15x!pf_babybear>) -> tensor<!pf_babybear> {
  %init = stablehlo.constant dense<0> : tensor<!pf_babybear>
  %0 = stablehlo.reduce(%x init: %init) applies stablehlo.add across dimensions = [0] : (tensor<15x!pf_babybear>, tensor<!pf_babybear>) -> tensor<!pf_babybear>
  return %0 : tensor<!pf_babybear>
}

// CHECK: @reduce_sum
