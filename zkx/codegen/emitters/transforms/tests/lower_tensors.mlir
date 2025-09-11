// RUN: emitters_opt %s --allow-unregistered-dialect -split-input-file \
// RUN: -zkx-lower-tensors="gpu_device_info='cuda_compute_capability {major: 6}'" \
// RUN: | FileCheck %s

// RUN: emitters_opt %s --allow-unregistered-dialect -split-input-file \
// RUN: -zkx-lower-tensors="target_type=cpu" \
// RUN: | FileCheck %s

// RUN: emitters_opt %s --allow-unregistered-dialect -split-input-file \
// RUN: -zkx-lower-tensors="gpu_device_info='cuda_compute_capability {major: 6}'" \
// RUN: | FileCheck %s --check-prefix=CHECK-PASCAL

module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<index, 32 : i32>>} {
  func.func private @add(%arg0: f32, %arg1: f32) -> f32 {
    %sum = arith.addf %arg0, %arg1 : f32
    func.return %sum : f32
  }

  func.func private @tensorarg(%arg0: tensor<43xf32> {zkx.invariant, zkx.slice_index = 0}, %arg1: index) -> f32 {
    %v1 = arith.constant 2.0 : f32
    %v2 = tensor.extract %arg0[%arg1] : tensor<43xf32>
    %sum = func.call @add(%v1, %v2) : (f32, f32) -> f32
    func.return %sum : f32
  }

  func.func @tensorcall(%arg0: tensor<43xf32> {zkx.slice_index = 0}, %arg1: index) -> f32 {
    %call = func.call @tensorarg(%arg0, %arg1) : (tensor<43xf32>, index) -> f32
    func.return %call : f32
  }

  func.func @stores(%arg0: tensor<17xf32> {zkx.slice_index = 0}, %arg1: tensor<43xf32> {zkx.slice_index = 1}) -> tensor<43xf32> {
    %c17 = arith.constant 17 : index
    %c23 = arith.constant 23 : index
    %cst = arith.constant 3.0 : f32
    %out = tensor.insert %cst into %arg1[%c17] : tensor<43xf32>
    %out2 = tensor.insert %cst into %out[%c23] : tensor<43xf32>
    func.return %out2 : tensor<43xf32>
  }
}

// CHECK:        func.func private @add(%{{.*}}: f32, %{{.*}}: f32) -> f32 {
// CHECK-NEXT:     arith.addf
// CHECK-NEXT:     return

// CHECK:        func.func private @tensorarg(%[[ARG0:.*]]: !llvm.ptr
// CHECK-SAME:        {zkx.invariant, zkx.slice_index = 0 : i64}, %[[ARG1:.*]]: index) -> f32 {
// CHECK-DAG:       %[[C2:.*]] = arith.constant 2.000000e+00
// CHECK-DAG:       %[[IDX:.*]] = arith.index_castui %[[ARG1]] : index to i32
// CHECK-DAG:       %[[PTR:.*]] = llvm.getelementptr inbounds %[[ARG0]][%[[IDX]]]
// CHECK-DAG:       %[[V2:.*]] = llvm.load %[[PTR]] invariant
// CHECK:           %[[RET:.*]] = call @add(%[[C2]], %[[V2]])
// CHECK:           return %[[RET]]

// CHECK:        func.func @tensorcall(%[[ARG0:.*]]: !llvm.ptr
// CHECK-SAME:        {zkx.slice_index = 0 : i64}, %[[ARG1:.*]]: index)
// CHECK:           %[[RET:.*]] = call @tensorarg(%[[ARG0]], %[[ARG1]])
// CHECK:           return %[[RET]]

// CHECK:        func.func @stores(
// CHECK-SAME:        %[[ARG0:.*]]: !llvm.ptr {zkx.slice_index = 0 : i64},
// CHECK-SAME:        %[[ARG1:.*]]: !llvm.ptr {zkx.slice_index = 1 : i64})
// CHECK-NEXT:      %[[CST:.*]] = arith.constant 3.000000e+00 : f32
// CHECK-NEXT:      %[[PTR1:.*]] = llvm.getelementptr inbounds %[[ARG1]][17]
// CHECK-NEXT:      llvm.store %[[CST]], %[[PTR1]]
// CHECK-NEXT:      %[[PTR2:.*]] = llvm.getelementptr inbounds %[[ARG1]][23]
// CHECK-NEXT:      llvm.store %[[CST]], %[[PTR2]]
// CHECK-NEXT:      return

// -----

func.func @store_control_flow( %arg0: tensor<2xf32>, %arg1: index)
    -> tensor<2xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %cst = arith.constant 0.0 : f32
  %cst2 = arith.constant 1.0 : f32

  %for = scf.for %i = %c0 to %c2 step %c1 iter_args(%arg2 = %arg0) -> tensor<2xf32> {
    %new_out = tensor.insert %cst into %arg2[%i] : tensor<2xf32>
    scf.yield %new_out : tensor<2xf32>
  }

  %inbounds = arith.cmpi sle, %arg1, %c1 : index
  %result = scf.if %inbounds -> tensor<2xf32> {
    %if = tensor.insert %cst2 into %for[%arg1] : tensor<2xf32>
    scf.yield %if : tensor<2xf32>
  } else {
    scf.yield %for : tensor<2xf32>
  }
  func.return %result : tensor<2xf32>
}
// CHECK-LABEL:     @store_control_flow(
// CHECK-SAME:  %[[ARG0:.*]]: !llvm.ptr, %[[X:.*]]: index) {
// CHECK-DAG:   %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:   %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG:   %[[C2:.*]] = arith.constant 2 : index
// CHECK:       scf.for %[[I:.*]] = %[[C0]] to %[[C2]] step %[[C1]] {
// CHECK:         %[[CAST:.*]] = arith.index_castui %[[I]] : index to i64
// CHECK:         %[[PTR:.*]] = llvm.getelementptr inbounds %[[ARG0]][%[[CAST]]]
// CHECK:         llvm.store {{.*}}, %[[PTR]]
// CHECK:       %[[INBOUNDS:.*]] = arith.cmpi
// CHECK:       scf.if %[[INBOUNDS]] {
// CHECK:         llvm.store
// CHECK-NEXT:  }
// CHECK-NEXT:  return

// -----

func.func @large_tensor(%arg0: tensor<8000000000xf32>, %arg1: index) -> f32 {
  %v = tensor.extract %arg0[%arg1] : tensor<8000000000xf32>
  func.return %v : f32
}
// CHECK-LABEL: @large_tensor
// CHECK: arith.index_castui {{.*}} : index to i64

// -----

func.func @extract_from_constant(%arg0: tensor<2xf32>, %arg1: index) -> f32 {
  %cst = arith.constant dense<[1.000000e+00, 2.000000e+00]> : tensor<2xf32>
  %extracted = tensor.extract %arg0[%arg1] : tensor<2xf32>
  %extracted_0 = tensor.extract %cst[%arg1] : tensor<2xf32>
  %0 = arith.addf %extracted, %extracted_0 : f32
  return %0 : f32
}
// CHECK: llvm.mlir.global private constant @global_cst_0(dense<
// CHECK-SAME: [1.000000e+00, 2.000000e+00]> : tensor<2xf32>) {addr_space = 0 : i32} : !llvm.array<2 x f32>
// CHECK-LABEL: @extract_from_constant
// CHECK: %[[ADDR_OF:.*]] = llvm.mlir.addressof @global_cst_0 : !llvm.ptr
// CHECK: %[[GEP:.*]] = llvm.getelementptr inbounds %[[ADDR_OF]][%{{.*}}] : (!llvm.ptr, i64) -> !llvm.ptr, f32
// CHECK: %[[LOAD:.*]] = llvm.load %[[GEP]] : !llvm.ptr -> f32
// CHECK: %[[ADD:.*]] = arith.addf %{{.*}}, %[[LOAD]] : f32
// CHECK: return %[[ADD]] : f32

// -----

func.func @vector_constant() -> vector<2xindex> {
  %c1 = arith.constant dense<[1, 2]> : vector<2xindex>
  func.return %c1 : vector<2xindex>
}
// vector constants should not be rewritten.
// CHECK-LABEL: @vector_constant
// CHECK-NEXT: arith.constant

// -----

// This example is a bit silly, in real life there wouldn't be a loop (the
// loop body would be executed by different threads). We're just doing it this
// way so control flow with shared memory is tested as well.
func.func @transpose_shared(%in: tensor<1024xf32>,
                            %out: tensor<1024xf32>) -> tensor<1024xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c1024 = arith.constant 1024 : index

  %shared = zkx_gpu.allocate_shared : tensor<1024xf32>
  %loaded_tile = scf.for %i = %c0 to %c1024 step %c1
      iter_args(%tile = %shared) -> tensor<1024xf32> {
    %v = tensor.extract %in[%i] : tensor<1024xf32>
    %inserted = tensor.insert %v into %tile[%i] : tensor<1024xf32>
    scf.yield %inserted : tensor<1024xf32>
  }

  %synced = zkx_gpu.sync_threads %shared : tensor<1024xf32>
  %written_tile = scf.for %i = %c0 to %c1024 step %c1
      iter_args(%written = %out) -> tensor<1024xf32> {
    %v = tensor.extract %shared[%i] : tensor<1024xf32>
    %inserted = tensor.insert %v into %written[%i] : tensor<1024xf32>
    scf.yield %inserted : tensor<1024xf32>
  }

  return %written_tile : tensor<1024xf32>
}
// CHECK:      llvm.mlir.global private @[[SHARED:shared_.*]]()
// CHECK-SAME:     {addr_space = 3 : i32} : !llvm.array<1024 x f32>
// CHECK:      @transpose_shared
// CHECK:        %[[ADDR:.*]] = llvm.mlir.addressof @[[SHARED]] : !llvm.ptr<3>
// CHECK:        %[[CAST:.*]] = llvm.addrspacecast %[[ADDR]]
// CHECK-SAME:       : !llvm.ptr<3> to !llvm.ptr
// CHECK:        scf.for
// CHECK:          %[[ELEM_ADDR:.*]] = llvm.getelementptr inbounds %[[CAST]]
// CHECK:          llvm.store {{.*}} %[[ELEM_ADDR]]
// CHECK:        gpu.barrier
// CHECK:        scf.for
// CHECK:          %[[ELEM_ADDR:.*]] = llvm.getelementptr inbounds %[[CAST]]
// CHECK:          llvm.load %[[ELEM_ADDR]]

// -----

func.func @atomic_rmw_i32(%in: tensor<8xi32>, %i: index) -> (tensor<8xi32>) {
  %ret = zkx.atomic_rmw %in[%i] : tensor<8xi32> {
    ^bb0(%current : i32):
      %c1 = arith.constant 1 : i32
      %add = arith.minui %current, %c1 : i32
      zkx.yield %add : i32
  }
  return %ret : tensor<8xi32>
}

// CHECK-PASCAL-LABEL: @atomic_rmw_i32
// CHECK-PASCAL: %[[C1:.*]] = arith.constant 1
// CHECK-PASCAL: %[[ADDR:.*]] = llvm.getelementptr
// CHECK-PASCAL-NEXT: llvm.atomicrmw umin %[[ADDR]], %[[C1]] seq_cst : !llvm.ptr, i32

// -----

func.func @atomic_rmw_i16(%in: tensor<8xi16>, %i: index)
    -> (tensor<8xi16>) {
  %ret = zkx.atomic_rmw %in[%i] : tensor<8xi16> {
    ^bb0(%current : i16):
      %c1 = arith.constant 1 : i16
      %add = arith.addi %current, %c1 : i16
      zkx.yield %add : i16
  }
  return %ret : tensor<8xi16>
}
// CHECK-LABEL: @atomic_rmw_i16
// CHECK: %[[ADDR:.*]] = llvm.getelementptr
// CHECK-NEXT: %[[ADDR_INT:.*]] = llvm.ptrtoint %[[ADDR]]
// CHECK-NEXT: %[[OFFSET:.*]] = llvm.and %[[ADDR_INT]], %{{.*}}
// CHECK-NEXT: %[[INDEX:.*]] = llvm.mul %[[OFFSET]], %{{.*}}
// CHECK-NEXT: %[[BASE:.*]] = llvm.getelementptr inbounds %[[ADDR]][%[[INDEX]]]
// CHECK: %[[INIT:.*]] = llvm.load %[[BASE]]
// CHECK-NEXT: scf.while (%[[VAR:.*]] = %[[INIT]])
// CHECK-NEXT: %[[VAR_SHIFT:.*]] = llvm.lshr %[[VAR]], %{{.*}}
// CHECK-NEXT: %[[VAR_TRUNC:.*]] = llvm.trunc %[[VAR_SHIFT]]
// CHECK-NEXT: %[[RES:.*]] = arith.addi %[[VAR_TRUNC]], %{{.*}}
// CHECK-NEXT: %[[RES_WIDE:.*]] = llvm.zext %[[RES]]
// CHECK-NEXT: %[[NEW_MASKED:.*]] = llvm.and %[[VAR]], %{{.*}}
// CHECK-NEXT: %[[RES_SHIFT:.*]] = llvm.shl %[[RES_WIDE]], %{{.*}}
// CHECK-NEXT: %[[NEW:.*]] = llvm.or %[[NEW_MASKED]], %[[RES_SHIFT]]
// CHECK-NEXT: llvm.cmpxchg %[[BASE]], %[[VAR]], %[[NEW]]

// -----

func.func @atomic_rmw_overwrite(%in: tensor<i16>, %i: index)
    -> (tensor<i16>) {
  %c1 = arith.constant 1 : i16
  %ret = zkx.atomic_rmw %in[%i] : tensor<i16> {
    ^bb0(%current : i16):
      zkx.yield %c1 : i16
  }
  return %ret : tensor<i16>
}
// CHECK-LABEL: @atomic_rmw_overwrite
// CHECK: %[[C1:.*]] = llvm.mlir.constant(1 : i32)
// CHECK: %[[ADDR:.*]] = llvm.getelementptr
// CHECK-NEXT: %[[ADDR_INT:.*]] = llvm.ptrtoint %[[ADDR]]
// CHECK-NEXT: %[[OFFSET:.*]] = llvm.and %[[ADDR_INT]], %{{.*}}
// CHECK-NEXT: %[[INDEX:.*]] = llvm.mul %[[OFFSET]], %{{.*}}
// CHECK-NEXT: %[[BASE:.*]] = llvm.getelementptr inbounds %[[ADDR]][%[[INDEX]]]
// CHECK: %[[INIT:.*]] = llvm.load %[[BASE]]
// CHECK-NEXT: scf.while (%[[VAR:.*]] = %[[INIT]])
// CHECK-NEXT: %[[NEW_MASKED:.*]] = llvm.and %[[VAR]], %{{.*}}
// CHECK-NEXT: %[[RES_SHIFT:.*]] = llvm.shl %[[C1]], %{{.*}}
// CHECK-NEXT: %[[NEW:.*]] = llvm.or %[[NEW_MASKED]], %[[RES_SHIFT]]
// CHECK-NEXT: llvm.cmpxchg %[[BASE]], %[[VAR]], %[[NEW]]

// -----

func.func @i4_load_store(%arg: tensor<10xi4>, %i: index, %j: index)
    -> tensor<10xi4> {
  %v = tensor.extract %arg[%i] : tensor<10xi4>
  %r = tensor.insert %v into %arg[%j] : tensor<10xi4>
  return %r : tensor<10xi4>
}
// CHECK-LABEL: @i4_load_store
// CHECK-DAG: %[[C4:.*]] = arith.constant 4 : i8
// CHECK-DAG: %[[C15:.*]] = arith.constant 15 : i8
// CHECK-DAG: %[[C_NEG16:.*]] = arith.constant -16 : i8
// CHECK: llvm.getelementptr
// CHECK-SAME: -> !llvm.ptr, i8
// CHECK: %[[VALUE_I8:.*]] = arith.extui {{.*}} : i4 to i8
// CHECK: llvm.getelementptr
// CHECK-SAME: -> !llvm.ptr, i8
// CHECK: %[[CURRENT_I32:.*]] = llvm.load
// CHECK-SAME: !llvm.ptr -> i32
// CHECK: scf.while (%[[INIT:.*]] = %[[CURRENT_I32]])
// CHECK: %[[SHIFTED:.*]] = llvm.lshr %[[INIT]]
// CHECK: %[[CURRENT:.*]] = llvm.trunc %[[SHIFTED]]
// CHECK: %[[MASKED_CURRENT_LO:.*]] = arith.andi %[[CURRENT]], %[[C_NEG16]] : i8
// CHECK: %[[MASKED_VALUE_I8:.*]] = arith.andi %[[VALUE_I8]], %[[C15]] : i8
// CHECK: %[[NEW_LO:.*]] = arith.ori %[[MASKED_CURRENT_LO]], %[[MASKED_VALUE_I8]] : i8
// CHECK: %[[MASKED_CURRENT_HI:.*]] = arith.andi %[[CURRENT]], %[[C15]] : i8
// CHECK: %[[VALUE_HI:.*]] = arith.shli %[[VALUE_I8]], %[[C4]] : i8
// CHECK: %[[NEW_HI:.*]] = arith.ori %[[MASKED_CURRENT_HI]], %[[VALUE_HI]] : i8
// CHECK: %[[NEW_VALUE:.*]] = arith.select %{{.*}}, %[[NEW_LO]], %[[NEW_HI]] : i8
// CHECK: %[[NEW_VALUE_I32:.*]] = llvm.zext %[[NEW_VALUE]]
// CHECK: %[[MASKED_INIT:.*]] = llvm.and %[[INIT]]
// CHECK: %[[NEW_VALUE_SHIFTED:.*]] = llvm.shl %[[NEW_VALUE_I32]]
// CHECK: %[[NEW_INIT:.*]] = llvm.or %[[MASKED_INIT]], %[[NEW_VALUE_SHIFTED]]
// CHECK: llvm.cmpxchg %{{.*}}, %[[INIT]], %[[NEW_INIT]] seq_cst seq_cst
// CHECK: scf.condition

// -----

func.func @direct_atomic_rmw_overwrite(%in: tensor<8xi32>,
  %i: index) -> (tensor<8xi32>) {
  %c2 = arith.constant 2 : i32
  %ret = zkx.atomic_rmw %in[%i] : tensor<8xi32> {
    ^bb0(%current : i32):
      zkx.yield %c2 : i32
  }
  return %ret : tensor<8xi32>
}
// CHECK-PASCAL-LABEL: @direct_atomic_rmw_overwrite
// CHECK-PASCAL: %[[C2:.*]] = arith.constant 2
// CHECK-PASCAL: %[[ADDR:.*]] = llvm.getelementptr
// CHECK-PASCAL: llvm.store %[[C2]], %[[ADDR]] atomic unordered {alignment = 4 : i64}

// -----

func.func @direct_atomic_rmw_addi(%in: tensor<8xi32>,
  %i: index) -> (tensor<8xi32>) {
  %c2 = arith.constant 2 : i32
  %ret = zkx.atomic_rmw %in[%i] : tensor<8xi32> {
    ^bb0(%current : i32):
      %min = arith.addi %current, %c2 : i32
      zkx.yield %c2 : i32
  }
  return %ret : tensor<8xi32>
}
// CHECK-PASCAL-LABEL: @direct_atomic_rmw_addi
// CHECK-PASCAL: %[[C2:.*]] = arith.constant 2
// CHECK-PASCAL: %[[ADDR:.*]] = llvm.getelementptr
// CHECK-PASCAL: llvm.atomicrmw add %[[ADDR]], %[[C2]] seq_cst

// -----

func.func @direct_atomic_rmw_maxsi(%in: tensor<8xi32>,
  %i: index) -> (tensor<8xi32>) {
  %c2 = arith.constant 2 : i32
  %ret = zkx.atomic_rmw %in[%i] : tensor<8xi32> {
    ^bb0(%current : i32):
      %min = arith.maxsi %current, %c2 : i32
      zkx.yield %c2 : i32
  }
  return %ret : tensor<8xi32>
}
// CHECK-PASCAL-LABEL: @direct_atomic_rmw_maxsi
// CHECK-PASCAL: %[[C2:.*]] = arith.constant 2
// CHECK-PASCAL: %[[ADDR:.*]] = llvm.getelementptr
// CHECK-PASCAL: llvm.atomicrmw max %[[ADDR]], %[[C2]] seq_cst

// -----

func.func @direct_atomic_rmw_maxui(%in: tensor<8xi32>,
  %i: index) -> (tensor<8xi32>) {
  %c2 = arith.constant 2 : i32
  %ret = zkx.atomic_rmw %in[%i] : tensor<8xi32> {
    ^bb0(%current : i32):
      %min = arith.maxui %current, %c2 : i32
      zkx.yield %c2 : i32
  }
  return %ret : tensor<8xi32>
}
// CHECK-PASCAL-LABEL: @direct_atomic_rmw_maxui
// CHECK-PASCAL: %[[C2:.*]] = arith.constant 2
// CHECK-PASCAL: %[[ADDR:.*]] = llvm.getelementptr
// CHECK-PASCAL: llvm.atomicrmw umax %[[ADDR]], %[[C2]] seq_cst

// -----

func.func @direct_atomic_rmw_minsi(%in: tensor<8xi32>,
  %i: index) -> (tensor<8xi32>) {
  %c2 = arith.constant 2 : i32
  %ret = zkx.atomic_rmw %in[%i] : tensor<8xi32> {
    ^bb0(%current : i32):
      %min = arith.minsi %current, %c2 : i32
      zkx.yield %c2 : i32
  }
  return %ret : tensor<8xi32>
}
// CHECK-PASCAL-LABEL: @direct_atomic_rmw_minsi
// CHECK-PASCAL: %[[C2:.*]] = arith.constant 2
// CHECK-PASCAL: %[[ADDR:.*]] = llvm.getelementptr
// CHECK-PASCAL: llvm.atomicrmw min %[[ADDR]], %[[C2]] seq_cst

// -----

func.func @direct_atomic_rmw_minui(%in: tensor<8xi32>,
  %i: index) -> (tensor<8xi32>) {
  %c2 = arith.constant 2 : i32
  %ret = zkx.atomic_rmw %in[%i] : tensor<8xi32> {
    ^bb0(%current : i32):
      %min = arith.minui %current, %c2 : i32
      zkx.yield %c2 : i32
  }
  return %ret : tensor<8xi32>
}
// CHECK-PASCAL-LABEL: @direct_atomic_rmw_minui
// CHECK-PASCAL: %[[C2:.*]] = arith.constant 2
// CHECK-PASCAL: %[[ADDR:.*]] = llvm.getelementptr
// CHECK-PASCAL: llvm.atomicrmw umin %[[ADDR]], %[[C2]] seq_cst

// -----

func.func @unused_index_switch_results(%i: index) -> index {
  %ret, %ret2 = scf.index_switch %i -> tensor<8xi32>, tensor<3xf32>
  case 0 {
    %x, %y = "dummy.op1"() : () -> (tensor<8xi32>, tensor<3xf32>)
    scf.yield %x, %y : tensor<8xi32>, tensor<3xf32>
  }
  default {
    %x, %y = "dummy.op2"() : () -> (tensor<8xi32>, tensor<3xf32>)
    scf.yield %x, %y : tensor<8xi32>, tensor<3xf32>
  }
  return %i : index
}
// CHECK-LABEL: func.func @unused_index_switch_results
// CHECK-SAME:      (%[[I:.*]]: index)
// CHECK-NEXT:    scf.index_switch %[[I]]
// CHECK-NEXT:    case 0 {
// CHECK-NEXT:      "dummy.op1"
// CHECK-NEXT:      scf.yield
// CHECK-NEXT:    }
// CHECK-NEXT:    default {
// CHECK-NEXT:      "dummy.op2"
// CHECK-NEXT:    }
// CHECK-NEXT:    return %[[I]] : index

// -----

func.func @transfer_write(%arg0: tensor<43xf32> {zkx.slice_index = 1}) -> tensor<43xf32> {
  %c16 = arith.constant 16 : index
  %c22 = arith.constant 22 : index
  %cst = arith.constant dense<[1.0, 2.0]> : vector<2xf32>
  %out = vector.transfer_write %cst, %arg0[%c16] : vector<2xf32>, tensor<43xf32>
  %out2 = vector.transfer_write %cst, %out[%c22] : vector<2xf32>, tensor<43xf32>
  func.return %out2 : tensor<43xf32>
}
// CHECK-LABEL: @transfer_write
// CHECK:           %[[PTR1:.*]] = llvm.getelementptr inbounds %[[BUF:.*]][16]
// CHECK-NEXT:      llvm.store %[[CST:.*]], %[[PTR1]]
// CHECK-NEXT:      %[[PTR2:.*]] = llvm.getelementptr inbounds %[[BUF]][22]
// CHECK-NEXT:      llvm.store %[[CST]], %[[PTR2]]

// -----

func.func @transfer_read(%arg0: tensor<43xf32> {zkx.slice_index = 1}) -> vector<2xf32> {
  %c16 = arith.constant 16 : index
  %c0 = arith.constant 0.0 : f32
  %out = vector.transfer_read %arg0[%c16], %c0 : tensor<43xf32>, vector<2xf32>
  func.return %out : vector<2xf32>
}
// CHECK-LABEL: @transfer_read
// CHECK:           %[[PTR:.*]] = llvm.getelementptr inbounds %{{.*}}[16]
// CHECK-NEXT:      llvm.load %[[PTR]] : !llvm.ptr -> vector<2xf32>

// -----

func.func @transfer_write_i1(%arg0: tensor<43xi1> {zkx.slice_index = 1},
                              %v1: vector<2xi1>, %v2: vector<2xi1>) -> tensor<43xi1> {
  %c16 = arith.constant 16 : index
  %c22 = arith.constant 22 : index
  %out = vector.transfer_write %v1, %arg0[%c16] : vector<2xi1>, tensor<43xi1>
  %out2 = vector.transfer_write %v2, %out[%c22] : vector<2xi1>, tensor<43xi1>
  func.return %out2 : tensor<43xi1>
}
// CHECK-LABEL: @transfer_write_i1
// CHECK-SAME:      (%[[ARG0:.*]]: !llvm.ptr
// CHECK-SAME:       %[[V1:.*]]: vector<2xi1>, %[[V2:.*]]: vector<2xi1>)
// CHECK-DAG:       %[[PTR1:.*]] = llvm.getelementptr inbounds %[[BUF:.*]][16]
// CHECK-DAG:       %[[V1_EXT:.*]] = arith.extui %[[V1]]
// CHECK:           llvm.store %[[V1_EXT]], %[[PTR1]]
// CHECK-DAG:       %[[PTR2:.*]] = llvm.getelementptr inbounds %[[BUF]][22]
// CHECK-DAG:       %[[V2_EXT:.*]] = arith.extui %[[V2]]
// CHECK:           llvm.store %[[V2_EXT]], %[[PTR2]]

// -----

func.func @transfer_read_i1(%arg0: tensor<43xi1> {zkx.slice_index = 1}) -> vector<2xi1> {
  %c16 = arith.constant 16 : index
  %false = arith.constant false
  %out = vector.transfer_read %arg0[%c16], %false : tensor<43xi1>, vector<2xi1>
  func.return %out : vector<2xi1>
}
// CHECK-LABEL: @transfer_read_i1
// CHECK-DAG:       %[[C0:.*]] = arith.constant dense<0> : vector<2xi8>
// CHECK-DAG:       %[[PTR:.*]] = llvm.getelementptr inbounds %{{.*}}[16]
// CHECK:           %[[LOADED:.*]] = llvm.load %[[PTR]] : !llvm.ptr
// CHECK:           %[[CAST:.*]] = arith.cmpi ne, %[[LOADED]], %[[C0]]
// CHECK:           return %[[CAST]] : vector<2xi1>

// -----

func.func @int4_constant(%arg0: tensor<3xi4>, %arg1: index) -> i4 {
  %cst = arith.constant dense<[1, 2, 3]> : tensor<3xi4>
  %extracted = tensor.extract %arg0[%arg1] : tensor<3xi4>
  %extracted_0 = tensor.extract %cst[%arg1] : tensor<3xi4>
  %0 = arith.addi %extracted, %extracted_0 : i4
  return %0 : i4
}
// CHECK: llvm.mlir.global private constant
// CHECK-SAME: dense<[18, 48]>
// CHECK-LABEL: @int4_constant

// -----

func.func @for_op(%arg0: tensor<500xf32>) -> f32 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index

  %cst = arith.constant dense<[1.0, 2.0, 3.0, 4.0]> : vector<4xf32>
  %for:2 = scf.for %i = %c0 to %c2 step %c1
      iter_args(%cst_ = %cst, %arg_ = %arg0)
        -> (vector<4xf32>, tensor<500xf32>) {
    %nested_for:2 = scf.for %j = %c0 to %c2 step %c1
        iter_args(%cst__ = %cst_, %arg__ = %arg_)
          -> (vector<4xf32>, tensor<500xf32>) {
      %index = arith.addi %i, %j : index
      %tensor_elem = tensor.extract %arg__[%index] : tensor<500xf32>
      %vector_elem = vector.extract %cst__[%index] : f32 from vector<4xf32>
      %sum = arith.addf %tensor_elem, %vector_elem : f32
      %v_update = vector.insert %sum, %cst__[%index] : f32 into vector<4xf32>
      %t_update = tensor.insert %sum into %arg__[%index] : tensor<500xf32>
      scf.yield %v_update, %t_update : vector<4xf32>, tensor<500xf32>
    }
    scf.yield %nested_for#0, %nested_for#1 : vector<4xf32>, tensor<500xf32>
  }
  %result = tensor.extract %for#1[%c0] : tensor<500xf32>
  func.return %result : f32
}

// CHECK-LABEL: @for_op
// CHECK: scf.for {{.*}} -> (vector<4xf32>) {
// CHECK-NEXT: scf.for {{.*}} -> (vector<4xf32>) {

// -----

func.func @i4_constant(%arg0: tensor<3xi4>, %arg1: index) -> i4 {
  %cst = arith.constant dense<[1, 2, 3]> : tensor<3xi4>
  %extracted = tensor.extract %arg0[%arg1] : tensor<3xi4>
  %extracted_0 = tensor.extract %cst[%arg1] : tensor<3xi4>
  %0 = arith.addi %extracted, %extracted_0 : i4
  return %0 : i4
}
// CHECK: llvm.mlir.global private constant
// CHECK-SAME: dense<[18, 48]>
// CHECK-LABEL: @i4_constant

// -----

func.func @transfer_read_i4(%arg0: tensor<43xi4> {zkx.slice_index = 1}) -> vector<2xi4> {
  %c16 = arith.constant 16 : index
  %c0 = arith.constant 0 : i4
  %out = vector.transfer_read %arg0[%c16], %c0 : tensor<43xi4>, vector<2xi4>
  func.return %out : vector<2xi4>
}
// CHECK-LABEL: @transfer_read_i4
// CHECK: %[[PTR:.*]] = llvm.getelementptr inbounds %{{.*}}[8]
// CHECK-NEXT: %[[LOAD:.*]] = llvm.load %[[PTR]] : !llvm.ptr -> vector<2xi4>
// CHECK-NEXT: %[[SHUFFLE:.*]] = vector.shuffle %[[LOAD]], %[[LOAD]] [1, 0] : vector<2xi4>, vector<2xi4>
// CHECK-NEXT: return %[[SHUFFLE]] : vector<2xi4>


// -----

func.func @transfer_write_i4(%arg0: tensor<43xi4> {zkx.slice_index = 1},
                             %arg1: vector<2xi4>) -> tensor<43xi4> {
  %c10 = arith.constant 10 : index
  %out = vector.transfer_write %arg1, %arg0[%c10] : vector<2xi4>, tensor<43xi4>
  func.return %out : tensor<43xi4>
}
// CHECK-LABEL: @transfer_write_i4(
// CHECK-SAME: %[[ARG0:.*]]: !llvm.ptr
// CHECK-SAME: %[[ARG1:.*]]: vector<2xi4>) {
// CHECK-NEXT: %[[SHUFFLE:.*]] = vector.shuffle %[[ARG1]], %[[ARG1]] [1, 0] : vector<2xi4>, vector<2xi4>
// CHECK-NEXT: %[[PTR:.*]] = llvm.getelementptr inbounds %[[ARG0]][5]
// CHECK-NEXT: llvm.store %[[SHUFFLE]], %[[PTR]] : vector<2xi4>, !llvm.ptr
// CHECK-NEXT: return
