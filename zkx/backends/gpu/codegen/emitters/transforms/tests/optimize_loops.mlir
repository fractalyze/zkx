// RUN: emitters_opt %s -split-input-file -zkx-gpu-optimize-loops | FileCheck %s

module {
  func.func @unroll_by_factor(%arg0: i32) -> i32 {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c256 = arith.constant 256 : index
    %ret = scf.for %i = %c0 to %c256 step %c1 iter_args (%v = %arg0) -> (i32) {
      %add = arith.addi %v, %v : i32
      scf.yield %add : i32
    }
    return %ret : i32
  }
}

// CHECK-LABEL: @unroll_by_factor
// CHECK: %[[C128:.*]] = arith.constant 128 : index
// CHECK: scf.for {{.*}} step %[[C128]]

// -----

module {
  func.func private @exp(%arg: i32) -> i32

  func.func @do_not_unroll(%arg0: i32) -> i32 {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c31 = arith.constant 31 : index
    %ret = scf.for %i = %c0 to %c31 step %c1 iter_args (%v = %arg0) -> (i32) {
      %add = arith.addi %v, %v : i32
      %exp = func.call @exp(%add) : (i32) -> i32
      scf.yield %exp : i32
    }
    return %ret : i32
  }
}

// CHECK-LABEL: @do_not_unroll
// CHECK: %[[C1:.*]] = arith.constant 1 : index
// CHECK: scf.for {{.*}} step %[[C1]]

// -----

module {
  func.func private @exp(%arg: i32) -> i32

  func.func @pipeline_extract(%arg: tensor<31xi32>) -> i32 {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c31 = arith.constant 31 : index
    %cst = arith.constant 0 : i32
    %ret = scf.for %i = %c0 to %c31 step %c1 iter_args (%iter = %cst) -> (i32) {
      %val = tensor.extract %arg[%i] : tensor<31xi32>
      %exp = func.call @exp(%val) : (i32) -> i32
      %add = arith.addi %exp, %iter : i32
      scf.yield %add : i32
    }
    return %ret : i32
  }
}

// CHECK: #[[$MAP:.*]] = #zkx.indexing_map<"(d0) -> (d0 + 1),
// CHECK-LABEL: @pipeline_extract
// CHECK-DAG:  %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:  %[[C30:.*]] = arith.constant 30 : index
// CHECK:      %[[VAL0:.*]] = tensor.extract %[[ARG0:.*]][%[[C0]]]
// CHECK:      scf.for %[[I:.*]] = %[[C0]] {{.*}} iter_args(%[[ITER:.*]] = {{.*}}, %[[VAL:.*]] = %[[VAL0]])
// CHECK-DAG:  %[[NEXT_I_EXISTS:.*]] = arith.cmpi ult, %[[I]], %[[C30]]
// CHECK-DAG:  %[[NEXT_I:.*]] = zkx.apply_indexing #[[$MAP]](%[[I]]
// CHECK:      %[[NEXT_VAL:.*]] = scf.if %[[NEXT_I_EXISTS]]
// CHECK-NEXT:   tensor.extract %[[ARG0]][%[[NEXT_I]]]
// CHECK-NEXT:   yield
// CHECK-NEXT: else
// CHECK-NEXT:   yield %[[VAL]]
// CHECK:     func.call @exp(%[[VAL]]) : (i32) -> i32
// CHECK-NEXT:   %[[ADD:.*]] = arith.addi
// CHECK-NEXT:   yield %[[ADD]], %[[NEXT_VAL]]

// -----

module {
  func.func private @exp(%arg: vector<2xi32>) -> vector<2xi32>

  func.func @pipeline_transfer(%arg: tensor<34xi32>) -> vector<2xi32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c17 = arith.constant 17 : index
    %cst = arith.constant dense<[0, 0]> : vector<2xi32>
    %cst0  = arith.constant 0 : i32
    %ret = scf.for %i = %c0 to %c17 step %c1 iter_args (%iter = %cst) -> (vector<2xi32>) {
      %base = zkx.apply_indexing #zkx.indexing_map<"(d0) -> (d0 * 2), domain: d0 in [0, 15]">(%i)
      %val = vector.transfer_read %arg[%base], %cst0 : tensor<34xi32>, vector<2xi32>
      %exp = func.call @exp(%val) : (vector<2xi32>) -> vector<2xi32>
      %add = arith.addi %exp, %iter : vector<2xi32>
      scf.yield %add : vector<2xi32>
    }
    return %ret : vector<2xi32>
  }
}

// CHECK-DAG: #[[$MAP0:.*]] = #zkx.indexing_map<"(d0) -> (d0 * 2),
// CHECK-DAG: #[[$MAP1:.*]] = #zkx.indexing_map<"(d0) -> (d0 + 1),
// CHECK-LABEL: @pipeline_transfer
// CHECK-DAG:  %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:  %[[C16:.*]] = arith.constant 16 : index
// CHECK:      %[[BASE0:.*]] = zkx.apply_indexing #[[$MAP0]](%[[C0]]
// CHECK:      %[[VAL0:.*]] = vector.transfer_read %[[ARG0:.*]][%[[BASE0]]]
// CHECK:      scf.for %[[I:.*]] = %[[C0]] {{.*}} iter_args(%[[ITER:.*]] = {{.*}}, %[[VAL:.*]] = %[[VAL0]])
// CHECK-DAG:  %[[NEXT_I_EXISTS:.*]] = arith.cmpi ult, %[[I]], %[[C16]]
// CHECK-DAG:  %[[NEXT_I:.*]] = zkx.apply_indexing #[[$MAP1]](%[[I]]
// CHECK-DAG:  %[[NEXT_BASE:.*]] = zkx.apply_indexing #[[$MAP0]](%[[NEXT_I]]
// CHECK:      %[[NEXT_VAL:.*]] = scf.if %[[NEXT_I_EXISTS]]
// CHECK-NEXT:   vector.transfer_read %[[ARG0]][%[[NEXT_BASE]]]
// CHECK-NEXT:   yield
// CHECK-NEXT: else
// CHECK-NEXT:   yield %[[VAL]]
// CHECK:     func.call @exp(%[[VAL]]) : (vector<2xi32>) -> vector<2xi32>
// CHECK-NEXT:     %[[ADD:.*]] = arith.addi
// CHECK-NEXT:     yield %[[ADD]], %[[NEXT_VAL]]

// -----

module {
  func.func @sequential_extract(%arg0: tensor<6xindex>, %arg1: tensor<22xindex>) -> (index) {
    %c1 = arith.constant 1 : index
    %c733 = arith.constant 733 : index
    %c0 = arith.constant 0 : index
    %2 = scf.for %i = %c0 to %c733 step %c1 iter_args(%x = %c1) -> (index) {
      %extracted = tensor.extract %arg0[%i] : tensor<6xindex>
      %extracted_1 = tensor.extract %arg1[%extracted] : tensor<22xindex>
      scf.yield %extracted_1 : index
    }
    return %2 : index
  }
}

// Once `extracted` is pipelined, it becomes an iter arg, so `extracted_1` is
// extract %arg1[%arg]. While it is possible to pipeline this in principle, we
// do not currently do this.

// CHECK-LABEL: @sequential_extract
// CHECK-SAME: (%[[ARG0:.*]]: tensor<6xindex>, %[[ARG1:.*]]: tensor<22xindex>)
// CHECK: tensor.extract %[[ARG0]]
// CHECK-NOT: tensor.extract
// CHECK: scf.for
