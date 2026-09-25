"""公开接口 :func:`analyze` 与编译/执行入口 :func:`compile_patterns` / :func:`run`。

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
    validate_value,
)
from .matcher import MatchPlan
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
    :ivar bindings: 该分支确定的绑定表 ``{名字: 类型名}``（or 各备选绑定一致）
    """

    index: int
    useful: bool
    witness: object
    bindings: dict

    @property
    def redundant(self) -> bool:
        return not self.useful


@dataclass(frozen=True)
class AnalysisResult:
    """整体分析结果。

    :ivar branches: 各分支判定（源顺序）
    :ivar uncovered: 整体遗漏值中字典序最小的一个；穷举时为 None
    :ivar exhaustive: 是否穷举（无遗漏）
    :ivar budget_used: 实际消耗的搜索工作量单位
    :ivar budget: 本次预算
    """

    branches: tuple
    uncovered: object
    exhaustive: bool
    budget_used: int
    budget: int

    @property
    def redundant_branches(self) -> tuple:
        """所有冗余分支的下标（源顺序）。"""
        return tuple(b.index for b in self.branches if b.redundant)


@dataclass(frozen=True)
class Match:
    """一次成功匹配。

    :ivar branch: 首个命中分支下标（源顺序即优先级）
    :ivar bindings: 该分支的绑定字典 ``{名字: 捕获到的完整子树值}``
    """

    branch: int
    bindings: dict


@dataclass(frozen=True)
class CompiledPatterns:
    """编译结果：可重复用于多个输入值，不携带每次匹配的可变状态。

    :ivar table: 校验后的类型表
    :ivar nodes: 各分支规范化内部模式节点（源顺序）
    :ivar branch_bindings: 各分支绑定表 ``[{名字: 类型id}]``（or 各备选一致）
    :ivar budget/budget_used: 编译期（覆盖分析）预算口径
    """

    table: TypeTable
    nodes: tuple
    branch_bindings: tuple
    budget: int
    budget_used: int

    def run(self, value) -> Match | None:
        """对值执行匹配，返回首个命中分支与绑定；无命中返回 None。"""
        return run(self, value)

    def analyze(self, *, budget: int | None = None) -> AnalysisResult:
        """在同一已编译模式上重新做覆盖分析（默认沿用编译时预算）。"""
        norm = list(zip(self.nodes, self.branch_bindings))
        return _analyze_nodes(
            self.table, norm, self.budget if budget is None else budget
        )


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


def _normalize_patterns(table: TypeTable, patterns) -> list:
    if not isinstance(patterns, (list, tuple)):
        raise CoverageError(
            f"patterns: 必须是 list/tuple（分支的源顺序列表），得到 {type(patterns).__name__}"
        )
    norm = []
    for i, pat in enumerate(patterns):
        norm.append(normalize(table, pat, table.root, f"patterns[{i}]"))
    return norm


def _analyze_nodes(table: TypeTable, norm: list, budget: int) -> AnalysisResult:
    """对已规范化模式序列执行覆盖分析。绑定节点在求解器中按通配处理。"""
    by_id = table_by_id(table)
    solver = DecisionSolver(table, by_id, budget)
    root_sig = solver.root_sig(table.root)

    cov = BOT_ID
    branches = []
    for i, (pat, binds) in enumerate(norm):
        witness_vid = solver.useful(cov, root_sig, solver.root_q(pat))
        useful = witness_vid is not None
        binding_names = {name: by_id[tid].name for name, tid in binds.items()}
        branches.append(
            BranchResult(
                index=i,
                useful=useful,
                witness=solver.root_value(witness_vid) if useful else None,
                bindings=binding_names,
            )
        )
        # 冗余分支的加入是空操作（其值集已被覆盖）；统一加入不改变结果。
        cov = solver.add(cov, root_sig, solver.root_q(pat))

    uncovered_vid = solver.smallest_uncovered(cov, root_sig)
    uncovered = solver.root_value(uncovered_vid) if uncovered_vid is not None else None

    return AnalysisResult(
        branches=tuple(branches),
        uncovered=uncovered,
        exhaustive=(uncovered is None),
        budget_used=solver.used,
        budget=budget,
    )


def analyze(types, patterns, *, budget: int = DEFAULT_BUDGET) -> AnalysisResult:
    """代数数据类型模式的覆盖见证分析。

    :param types: 有序 dict，首项为被测根类型；形态见模块文档与 README。
    :param patterns: 分支模式列表（源顺序即匹配优先级顺序）；允许空列表，
        此时所有值均未覆盖。模式可含变量绑定 ``('var','x')`` 与
        as 绑定 ``('as', p, 'x')``；覆盖分析把它们当通配处理。
    :param budget: 搜索工作量上限（正整数）。耗尽时抛
        :class:`CoverageBudgetExhausted`——它表示“预算内未能得出结论”，
        **不是**“已证明穷举”。
    :returns: :class:`AnalysisResult`
    :raises CoverageError: 输入违反契约（错误信息含可定位路径；含绑定名在
        同一路径重复、or 各备选绑定名字/类型不一致）
    :raises CoverageBudgetExhausted: 搜索预算耗尽
    """
    budget = _check_budget(budget)
    table: TypeTable = build_type_table(types)
    norm = _normalize_patterns(table, patterns)
    return _analyze_nodes(table, norm, budget)


def compile_patterns(types, patterns, *, budget: int = DEFAULT_BUDGET) -> CompiledPatterns:
    """校验类型与模式（含绑定规则）并编译为可重复执行的匹配计划。

    编译期同时运行覆盖分析（绑定按通配），若预算耗尽则抛
    :class:`CoverageBudgetExhausted`——此时不会返回半成品。

    :param types: 有序 dict，首项为被测根类型。
    :param patterns: 分支模式列表（源顺序即匹配优先级）。
    :returns: :class:`CompiledPatterns`，可对多个值反复 ``run``。
    """
    budget = _check_budget(budget)
    table: TypeTable = build_type_table(types)
    norm = _normalize_patterns(table, patterns)
    # 跑一遍覆盖分析以保持与 analyze 完全一致的编译期拒绝/预算语义。
    analysis = _analyze_nodes(table, norm, budget)
    return CompiledPatterns(
        table=table,
        nodes=tuple(node for node, _ in norm),
        branch_bindings=tuple(dict(binds) for _, binds in norm),
        budget=budget,
        budget_used=analysis.budget_used,
    )


def run(compiled: CompiledPatterns, value) -> Match | None:
    """对已编译计划与一个合法类型值执行匹配。

    按源顺序逐分支尝试，返回**首个命中**分支下标与绑定字典：
    ``Match(branch=i, bindings={名字: 捕获到的完整子树值})``；
    无任何分支命中时显式返回 ``None``。

    :param compiled: :func:`compile_patterns` 的返回值。
    :param value: 根类型的合法值（形态同见证值）；非法值抛带路径的
        :class:`CoverageError`。
    """
    if not isinstance(compiled, CompiledPatterns):
        raise CoverageError(
            f"compiled: 必须是 compile_patterns 的返回值，得到 {type(compiled).__name__}"
        )
    table = compiled.table
    by_id = table_by_id(table)
    # 先校验值：非法输入不得进入匹配、也不得产生部分绑定。
    validate_value(by_id, table.root, value)
    plan = MatchPlan(by_id)
    for i, node in enumerate(compiled.nodes):
        binds = {}
        if plan.match(node, table.root, value, binds):
            # 规范化已保证：同路径不重名、or 各备选绑定集合一致；
            # 每个被声明绑定恰被赋值一次。
            return Match(branch=i, bindings=binds)
    return None


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
    "BOOL_NAME",
]
