// RUN: mlir-opt %s -canonicalize -split-input-file | FileCheck %s

// CHECK-LABEL: func.func @pack_drop_identity_outer_dims_perm
// CHECK: %[[EMPTY:.+]] = tensor.empty() : tensor<8x32xf32>
// CHECK: %[[PACK:.+]] = linalg.pack %arg0 inner_dims_pos = [0] inner_tiles = [32] into %[[EMPTY]]
// CHECK-NOT: outer_dims_perm
// CHECK: return %[[PACK]] : tensor<8x32xf32>
func.func @pack_drop_identity_outer_dims_perm(%arg0: tensor<256xf32>) -> tensor<8x32xf32> {
  %empty = tensor.empty() : tensor<8x32xf32>
  %0 = linalg.pack %arg0 outer_dims_perm = [0] inner_dims_pos = [0] inner_tiles = [32] into %empty
    : tensor<256xf32> -> tensor<8x32xf32>
  return %0 : tensor<8x32xf32>
}

// -----

// CHECK-LABEL: func.func @pack_keep_non_identity_outer_dims_perm
// CHECK: linalg.pack %arg0 outer_dims_perm = [1, 0] inner_dims_pos = [1] inner_tiles = [32]
func.func @pack_keep_non_identity_outer_dims_perm(%arg0: tensor<5x256xf32>) -> tensor<8x5x32xf32> {
  %empty = tensor.empty() : tensor<8x5x32xf32>
  %0 = linalg.pack %arg0 outer_dims_perm = [1, 0] inner_dims_pos = [1] inner_tiles = [32] into %empty
    : tensor<5x256xf32> -> tensor<8x5x32xf32>
  return %0 : tensor<8x5x32xf32>
}

// -----

// CHECK-LABEL: func.func @unpack_drop_identity_outer_dims_perm
// CHECK: %[[EMPTY:.+]] = tensor.empty() : tensor<256xf32>
// CHECK: %[[UNPACK:.+]] = linalg.unpack %arg0 inner_dims_pos = [0] inner_tiles = [32] into %[[EMPTY]]
// CHECK-NOT: outer_dims_perm
// CHECK: return %[[UNPACK]] : tensor<256xf32>
func.func @unpack_drop_identity_outer_dims_perm(%arg0: tensor<8x32xf32>) -> tensor<256xf32> {
  %empty = tensor.empty() : tensor<256xf32>
  %0 = linalg.unpack %arg0 outer_dims_perm = [0] inner_dims_pos = [0] inner_tiles = [32] into %empty
    : tensor<8x32xf32> -> tensor<256xf32>
  return %0 : tensor<256xf32>
}

// -----

// CHECK-LABEL: func.func @unpack_keep_non_identity_outer_dims_perm
// CHECK: linalg.unpack %arg0 outer_dims_perm = [1, 0] inner_dims_pos = [1] inner_tiles = [32]
func.func @unpack_keep_non_identity_outer_dims_perm(%arg0: tensor<8x5x32xf32>) -> tensor<5x256xf32> {
  %empty = tensor.empty() : tensor<5x256xf32>
  %0 = linalg.unpack %arg0 outer_dims_perm = [1, 0] inner_dims_pos = [1] inner_tiles = [32] into %empty
    : tensor<8x5x32xf32> -> tensor<5x256xf32>
  return %0 : tensor<5x256xf32>
}
