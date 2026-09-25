"""patcov —— 有限代数数据类型模式的覆盖见证库。

公开接口：
    analyze(types, patterns, *, budget=DEFAULT_BUDGET) -> AnalysisResult
    compile_matcher(types, patterns) -> Matcher
    matcher.match(value) -> MatchResult | None
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
from .match import MatchResult, Matcher, compile_matcher
from .model import BOOL_NAME, MAX_CONSTRUCTORS, MAX_FIELD_ARITY, MAX_NESTING

__all__ = [
    "analyze",
    "AnalysisResult",
    "BranchResult",
    "compile_matcher",
    "Matcher",
    "MatchResult",
    "CoverageError",
    "CoverageBudgetExhausted",
    "DEFAULT_BUDGET",
    "MAX_CONSTRUCTORS",
    "MAX_NESTING",
    "MAX_FIELD_ARITY",
    "BOOL_NAME",
]
