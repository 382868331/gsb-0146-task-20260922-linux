"""patcov —— 代数数据类型模式的“覆盖见证”库。

公开接口：
    analyze(types, patterns, *, budget=DEFAULT_BUDGET, version=1,
            certificate=False)

- version=1（默认）：有限非递归类型，返回 :class:`AnalysisResult`；
- version=2：递归类型 + 整数闭区间 + or 绑定 + 守卫，返回
  :class:`patcov.v2.V2Result`，支持覆盖证书与独立检查器。
"""

from __future__ import annotations

from .analyze import (
    AnalysisResult,
    BranchResult,
    CoverageBudgetExhausted,
    CoverageError,
    DEFAULT_BUDGET,
    analyze,
)
from .certcheck import CertificateRejected, verify_certificate
from .model import (
    BOOL_NAME,
    MAX_CONSTRUCTORS,
    MAX_FIELD_ARITY,
    MAX_NESTING,
)
from .v2 import DEFAULT_BUDGET_V2, V2BranchResult, V2Result

__all__ = [
    "analyze",
    "AnalysisResult",
    "BranchResult",
    "CoverageError",
    "CoverageBudgetExhausted",
    "CertificateRejected",
    "verify_certificate",
    "V2Result",
    "V2BranchResult",
    "DEFAULT_BUDGET",
    "DEFAULT_BUDGET_V2",
    "MAX_CONSTRUCTORS",
    "MAX_NESTING",
    "MAX_FIELD_ARITY",
    "BOOL_NAME",
]
