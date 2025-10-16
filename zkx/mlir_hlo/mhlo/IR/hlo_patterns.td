include "zkx/mlir_hlo/mhlo/IR/hlo_ops.td"

def HasSameType : Constraint<CPred<"$0.getType() == $1.getType()">>;

def EliminateIdentityConvert : Pat<
  (MHLO_ConvertOp:$res $src),
  (replaceWithValue $src),
  [(HasSameType $res, $src)]>;
