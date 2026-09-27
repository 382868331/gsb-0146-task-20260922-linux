"""公开接口 :func:`analyze`。

纯函数、无模块级状态：每次调用构造全新的求解结构，非法输入在校验阶段即抛出
:class:`CoverageError`，不会产生任何部分结果或外部变更。
"""

from __future__ import annotations

import sys
from dataclasses import dataclass

from .errors import CoverageError, CoverageBudgetExhausted
from .model import (
    BOOL_NAME,
    MAX_FIELD_ARITY,
    MAX_NESTING,
    TypeTable,
    build_type_table,
    normalize,
    table_by_id,
)
from .solver import BOT_ID, DecisionSolver

# 求解器按值树递归下降；位置序列长度最坏为 MAX_NESTING × MAX_FIELD_ARITY。
# 每个位置约需数个 Python 栈帧，据此设置解释器递归上限，保证“上限内”输入在
# Linux 原生 Python 3.14 上不触发 RecursionError（实测约 3 帧/位置，取 5 倍余量）。
_REQUIRED_LIMIT = 2000 + 5 * MAX_NESTING * MAX_FIELD_ARITY  # = 602_000
sys.setrecursionlimit(max(sys.getrecursionlimit(), _REQUIRED_LIMIT))

DEFAULT_BUDGET = 50_000_000
MIN_BUDGET = 1


@dataclass(frozen=True)
class BranchResult:
    """单个分支（按源顺序）的判定结果。

    :ivar index: 分支下标（从 0 开始）
    :ivar useful: 该分支是否存在未被前面分支覆盖的值
    :ivar witness: 有用时为字典序最小的见证值；冗余（不可达）时为 None
    :ivar redundant: 等价于 ``not useful``，冗余分支即永不匹配的分支
    :ivar count: 实际由该分支接收的值个数（精确整数）：匹配本分支、但不匹配
        任何前序分支的值的数量。同一值即使符合多个模式也只计入最先命中的
        分支，故各分支 count 互不重迭；``useful`` 等价于 ``count > 0``。
    """

    index: int
    useful: bool
    witness: object
    count: int

    @property
    def redundant(self) -> bool:
        return not self.useful


@dataclass(frozen=True)
class AnalysisResult:
    """整体分析结果。

    :ivar branches: 各分支判定（源顺序）
    :ivar uncovered: 整体遗漏值中字典序最小的一个；穷举时为 None
    :ivar exhaustive: 是否穷举（无遗漏）
    :ivar uncovered_count: 未被任何分支处理的值个数（精确整数）；
        ``exhaustive`` 等价于 ``uncovered_count == 0``
    :ivar total_values: 根类型的全部可能值个数（精确整数）
    :ivar budget_used: 实际消耗的搜索工作量单位
    :ivar budget: 本次预算

    恒等式（可相互核对）：
    ``sum(b.count for b in branches) + uncovered_count == total_values``
    """

    branches: tuple
    uncovered: object
    exhaustive: bool
    uncovered_count: int
    total_values: int
    budget_used: int
    budget: int

    @property
    def redundant_branches(self) -> tuple:
        """所有冗余分支的下标（源顺序）。"""
        return tuple(b.index for b in self.branches if b.redundant)


def _check_budget(budget) -> int:
    # 整数参数拒绝 bool（bool 是 int 子类）；不接受浮点
    if not isinstance(budget, int) or isinstance(budget, bool):
        raise CoverageError(
            f"budget: 必须是 int（工作量单位，{MIN_BUDGET}..2^63-1），"
            f"得到 {type(budget).__name__}"
        )
    if budget < MIN_BUDGET or budget > (1 << 63) - 1:
        raise CoverageError(f"budget: 必须落在 [{MIN_BUDGET}, 2^63-1]，得到 {budget}")
    return budget


def analyze(types, patterns, *, budget: int = DEFAULT_BUDGET) -> AnalysisResult:
    """代数数据类型模式的覆盖见证分析。

    :param types: 有序 dict，首项为被测根类型；形态见模块文档与 README。
    :param patterns: 分支模式列表（源顺序即匹配优先级顺序）；允许空列表，
        此时所有值均未覆盖。
    :param budget: 搜索工作量上限（正整数）。耗尽时抛
        :class:`CoverageBudgetExhausted`——它表示“预算内未能得出结论”，
        **不是**“已证明穷举”。
    :returns: :class:`AnalysisResult`
    :raises CoverageError: 输入违反契约（错误信息含可定位路径）
    :raises CoverageBudgetExhausted: 搜索预算耗尽
    """
    budget = _check_budget(budget)
    table: TypeTable = build_type_table(types)

    if not isinstance(patterns, (list, tuple)):
        raise CoverageError(
            f"patterns: 必须是 list/tuple（分支的源顺序列表），得到 {type(patterns).__name__}"
        )
    norm = []
    for i, pat in enumerate(patterns):
        norm.append(normalize(table, pat, table.root, f"patterns[{i}]"))

    by_id = table_by_id(table)
    solver = DecisionSolver(table, by_id, budget)
    root_sig = solver.root_sig(table.root)

    cov = BOT_ID
    branches = []
    for i, pat in enumerate(norm):
        witness_vid = solver.useful(cov, root_sig, solver.root_q(pat))
        useful = witness_vid is not None
        before = solver.count(cov, root_sig)
        # 冗余分支的加入是空操作（其值集已被覆盖）；统一加入不改变结果。
        cov = solver.add(cov, root_sig, solver.root_q(pat))
        after = solver.count(cov, root_sig)
        branches.append(
            BranchResult(index=i, useful=useful,
                         witness=solver.root_value(witness_vid) if useful else None,
                         count=after - before)
        )

    uncovered_vid = solver.smallest_uncovered(cov, root_sig)
    uncovered = solver.root_value(uncovered_vid) if uncovered_vid is not None else None
    total_values = solver.sig_size(root_sig)
    uncovered_count = total_values - solver.count(cov, root_sig)

    return AnalysisResult(
        branches=tuple(branches),
        uncovered=uncovered,
        exhaustive=(uncovered is None),
        uncovered_count=uncovered_count,
        total_values=total_values,
        budget_used=solver.used,
        budget=budget,
    )


__all__ = [
    "analyze",
    "AnalysisResult",
    "BranchResult",
    "CoverageError",
    "CoverageBudgetExhausted",
    "DEFAULT_BUDGET",
    "BOOL_NAME",
]
