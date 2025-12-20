# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
# Copyright 2022 The StableHLO Authors.
# Copyright 2025 The ZKX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# pylint: disable=wildcard-import,relative-beyond-top-level,g-import-not-at-top
from ._stablehlo_ops_gen import *
from ._stablehlo_ops_gen import _Dialect, ConstantOp
from .._mlir_libs._stablehlo import *
from ._ods_common import _cext as _ods_cext
from ._ods_common import (
    get_default_loc_context as _ods_get_default_loc_context,
)

_ods_ir = _ods_cext.ir


@_ods_cext.register_operation(_Dialect, replace=True)
class ConstantOpExt(ConstantOp):

    def __init__(self, result, value, *, loc=None, ip=None):
        _ods_context = _ods_get_default_loc_context(loc)
        attributes = {}
        attributes["value"] = (
            value
            if (
                isinstance(value, _ods_ir.Attribute)
                or not _ods_ir.AttrBuilder.contains("ElementsAttr")
            )
            else _ods_ir.AttrBuilder.get("ElementsAttr")(value, context=_ods_context)
        )
        op = _ods_ir.Operation.create(
            self.OPERATION_NAME, results=[result], attributes=attributes, loc=loc, ip=ip
        )
        super(ConstantOp, self).__init__(op)


def constant_ext(result, value, *, loc=None, ip=None) -> _ods_ir.Value:
    return ConstantOpExt(result=result, value=value, loc=loc, ip=ip).result
