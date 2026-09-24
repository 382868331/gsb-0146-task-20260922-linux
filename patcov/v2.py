"""v2 公开入口编排：递归类型 + 整数闭区间 + or 绑定 + 守卫。

与 v1 入口并存（``analyze(..., version=2)``）。职责：

- 规范化各分支（:class:`patcov.model.BranchDecl`）；
- 用 :class:`patcov.recsolver.V2Solver` 逐分支求有用性，再求整体遗漏；
- 预算耗尽给 ``status="unknown"`` 与*未决状态*（纯数据），绝不宣称穷尽；
- 成功时可选装配覆盖证书，交独立检查器 :mod:`patcov.certcheck` 复核
  （检查器不复用本搜索的任何缓存）。
"""

from __future__ import annotations

from dataclasses import dataclass

from .errors import CoverageError
from .guards import K_FALSE
from .model import (
    build_type_table,
    normalize_branch_v2,
    table_by_id,
)
from .recsolver import BudgetStop, V2Solver

DEFAULT_BUDGET_V2 = 20_000_000


@dataclass(frozen=True)
class V2BranchResult:
    """单个 v2 分支判定。

    :ivar useful: True/False 为确定判定；None 表示预算内未决
    :ivar witness: 有用时为题面序下最小见证；否则 None
    """

    index: int
    useful: object
    witness: object

    @property
    def redundant(self):
        return self.useful is False


@dataclass(frozen=True)
class V2Result:
    """v2 分析结果。

    :ivar status:
        - ``"exhaustive"``：已证明穷尽（uncovered is None）；
        - ``"incomplete"``：确定存在遗漏，uncovered 为题面序最小见证；
        - ``"unknown"``：预算耗尽，结论未知；pending 列未决有限状态。
    """

    status: str
    branches: tuple
    uncovered: object
    exhaustive: object
    certificate: object
    pending: tuple
    budget_used: int
    budget: int

    @property
    def redundant_branches(self):
        return tuple(b.index for b in self.branches if b.useful is False)

    @property
    def unknown_branches(self):
        return tuple(b.index for b in self.branches if b.useful is None)


def analyze_v2(types, patterns, *, budget: int, want_certificate: bool) -> V2Result:
    table = build_type_table(types, version=2)
    if not isinstance(patterns, (list, tuple)):
        raise CoverageError(
            f"patterns: 必须是 list/tuple（分支的源顺序列表），得到 {type(patterns).__name__}"
        )

    decls = [normalize_branch_v2(table, raw, i, f"patterns[{i}]")
             for i, raw in enumerate(patterns)]

    solver = V2Solver(table, table_by_id(table), budget)
    root_sig = (table.root,)

    branches: list = []
    prior_rows: frozenset = frozenset()
    try:
        for d in decls:
            tgt_rows = _drop_false(solver, solver.branch_rows(d, True))
            combined = frozenset(prior_rows) | tgt_rows
            r = solver.target(combined, root_sig)
            if r is None:
                branches.append(V2BranchResult(d.index, False, None))
            else:
                branches.append(V2BranchResult(d.index, True, r[2][0]))
            prior_rows = frozenset(prior_rows) | _drop_false(
                solver, solver.branch_rows(d, False))

        root_ans = solver.cover(prior_rows, root_sig, record=want_certificate)
    except BudgetStop as bs:
        pending = _render_pending(solver, bs)
        return V2Result(
            status="unknown",
            branches=tuple(_fill_unknown(branches, len(decls))),
            uncovered=None,
            exhaustive=None,
            certificate=None,
            pending=pending,
            budget_used=solver.used,
            budget=budget,
        )

    if root_ans is None:
        status, uncovered, exhaustive = "exhaustive", None, True
    else:
        status, uncovered, exhaustive = "incomplete", root_ans[2][0], False

    certificate = None
    if want_certificate:
        from .certificate import build_certificate
        certificate = build_certificate(solver, table, root_sig, prior_rows,
                                        status, uncovered, root_ans)

    return V2Result(
        status=status,
        branches=tuple(branches),
        uncovered=uncovered,
        exhaustive=exhaustive,
        certificate=certificate,
        pending=(),
        budget_used=solver.used,
        budget=budget,
    )


def _drop_false(solver: V2Solver, rows) -> frozenset:
    return frozenset(r for r in rows if solver.guard(r[2]) != K_FALSE)


def _fill_unknown(branches, total):
    out = list(branches)
    known = {b.index for b in out}
    for i in range(total):
        if i not in known:
            out.append(V2BranchResult(i, None, None))
    out.sort(key=lambda b: b.index)
    return out


def _render_pending(solver: V2Solver, bs: BudgetStop) -> tuple:
    out = []
    for is_tgt, rows, sig in sorted(bs.pending, key=lambda x: repr(x)):
        out.append({
            "question": "usefulness" if is_tgt else "coverage",
            "state": solver.render_state(rows, tuple(sig)),
        })
    return tuple(out)
