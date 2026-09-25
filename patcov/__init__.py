"""patcov —— 有限代数数据类型模式的覆盖见证库。

公开接口：
    analyze(types, patterns, *, budget=DEFAULT_BUDGET) -> AnalysisResult
    compile_patterns(types, patterns, *, budget=DEFAULT_BUDGET) -> CompiledPatterns
    run(compiled, value) -> Match | None
"""

from __future__ import annotations

from .analyze import (
    AnalysisResult,
    BranchResult,
    CompiledPatterns,
    CoverageBudgetExhausted,
    CoverageError,
    DEFAULT_BUDGET,
    Match,
    analyze,
    compile_patterns,
    run,
)
from .model import BOOL_NAME, MAX_CONSTRUCTORS, MAX_FIELD_ARITY, MAX_NESTING

__all__ = [
    "analyze",
    "compile_patterns",
    "run",
    "Match",
    "CompiledPatterns",
    "AnalysisResult",
    "BranchResult",
    "CoverageError",
    "CoverageBudgetExhausted",
    "DEFAULT_BUDGET",
    "MAX_CONSTRUCTORS",
    "MAX_NESTING",
    "MAX_FIELD_ARITY",
    "BOOL_NAME",
]
