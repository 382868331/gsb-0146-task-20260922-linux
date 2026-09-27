#!/usr/bin/env python3
"""固定输入演示：真实计算覆盖见证，并实际触发一个拒绝边界与预算耗尽边界。

运行：python demo.py
不使用 sleep、不使用预录结果；所有输出均由 patcov.analyze 当场算出。
"""

from __future__ import annotations

import time

from patcov import (
    CoverageBudgetExhausted,
    CoverageError,
    analyze,
)


def render(value) -> str:
    """把见证值渲染为紧凑可读字符串。"""
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, tuple):
        return "(" + ", ".join(render(v) for v in value) + ")"
    return str(value)


def section(title: str) -> None:
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def main() -> None:
    overall_start = time.time()

    # --------------------------------------------------------------- 正常结果
    section("正常结果：有标签和（Shape）+ 积字段，按源顺序判定")
    types = {
        "Shape": {
            "kind": "sum",
            "variants": [
                ("Circle", [("r", "Bool")]),
                ("Rect", [("w", "Bool"), ("h", "Bool")]),
                ("Dot", []),
            ],
        },
    }
    patterns = [
        ("Circle", "True"),          # 0：只覆盖 (Circle, True)
        ("Rect", "True", "_"),       # 1：覆盖 w=True 的两个矩形
        ("or", "Dot", ("Circle", "False")),  # 2：or，备选重复语义求并
        ("Rect", "_", "False"),      # 3：检查剩余矩形
        "_",                         # 4：通配兜底
        "Dot",                       # 5：通配之后永不匹配（冗余）
    ]
    result = analyze(types, patterns)
    print("分支（源顺序 = 匹配优先级）：")
    for b in result.branches:
        if b.useful:
            print(f"  分支 {b.index}: 有用，首个命中 {b.count} 种输入，"
                  f"最小见证 = {render(b.witness)}")
        else:
            print(f"  分支 {b.index}: 冗余（永不匹配，命中 0 种输入）")
    print(f"冗余分支下标：{list(result.redundant_branches)}")
    print(f"全部可能值：{result.total_count}；各分支首命中之和：{result.covered_count}；"
          f"未被任何分支处理：{result.uncovered_count}")
    assert result.covered_count + result.uncovered_count == result.total_count
    if result.exhaustive:
        print("整体：已穷举，无遗漏值（遗漏数 = 0）。")
    else:
        print(f"整体：遗漏 {result.uncovered_count} 种，字典序最小遗漏值 = {render(result.uncovered)}")

    # ------------------------------------------------------- 巨大 Bool 积
    section("巨大 Bool 积：500 个 Bool 字段（值域 2^500，绝不展开笛卡尔积）")
    n = 500
    big_types = {
        "Big": {"kind": "product",
                "fields": [(f"b{i}", "Bool") for i in range(n)]},
    }
    t0 = time.time()
    big = analyze(big_types, [["False"] + ["_"] * (n - 1), ["True"] * n])
    dt = time.time() - t0
    head = render(big.uncovered[:4])
    print(f"  两个分支均有用；冗余分支：{list(big.redundant_branches)}")
    half = 1 << (n - 1)
    print(f"  精确计数（Python 大整数，非抽样估计）：分支0 命中 {big.branch_counts[0]}（=2^{n-1}：{big.branch_counts[0] == half}），")
    print(f"    分支1 命中 {big.branch_counts[1]}；遗漏 {big.uncovered_count}；全集 {big.total_count}（=2^{n}）")
    assert sum(big.branch_counts) + big.uncovered_count == big.total_count
    print(f"  字典序最小遗漏值前 4 位：{head.rstrip(')')}, ...)（共 {n} 位）")
    print(f"  实算耗时 {dt*1000:.1f} ms，工作量单位 {big.budget_used}（约 5×字段数，线性，非 2^{n}）")

    # ------------------------------------------------- 边界 1：非法输入被拒绝
    section("拒绝边界：构造器不属于该类型（错误可定位，不静默纠正）")
    bad_types = {"Color": {"kind": "enum", "constructors": ["Red", "Green", "Blue"]}}
    try:
        analyze(bad_types, ["Yelow"])  # 拼写错误的构造器
    except CoverageError as exc:
        print(f"  捕获 CoverageError：{exc}")
    else:
        raise AssertionError("非法构造器必须被拒绝")

    # 数值边界：budget 拒绝 bool 与 NaN
    for bad_budget in (True, float("nan")):
        try:
            analyze(bad_types, ["Red"], budget=bad_budget)
        except CoverageError as exc:
            print(f"  budget={bad_budget!r:>5} 被拒绝：{exc}")

    # ------------------------------------------ 边界 2：预算耗尽 ≠ 证明穷举
    section("预算边界：搜索预算耗尽（CoverageBudgetExhausted）与“已证明穷举”严格分开")
    try:
        # 预算 1：第一步即耗尽，此时结论未知，绝不能误报为“穷举”
        analyze(bad_types, ["Red"], budget=1)
    except CoverageBudgetExhausted as exc:
        print(f"  捕获 CoverageBudgetExhausted：{exc}")
    else:
        raise AssertionError("预算 1 必须耗尽并抛出异常")
    # 同一输入给足预算即可正常完成——对照证明上面不是“无解/穷举”
    ok = analyze(bad_types, ["Red"], budget=100_000)
    print(f"  同输入预算充足时：exhaustive={ok.exhaustive}，"
          f"最小遗漏值={render(ok.uncovered)}（确实未穷举，反证耗尽≠穷举）")

    print()
    print(f"演示总耗时 {time.time() - overall_start:.2f} 秒（要求约 8 秒内）。")


if __name__ == "__main__":
    main()
