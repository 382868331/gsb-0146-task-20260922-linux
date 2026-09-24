#!/usr/bin/env python3
"""固定输入演示：v2 递归类型 + 整数闭区间 + or 绑定 + 守卫 + 覆盖证书。

运行：python demo.py
不使用 sleep、不使用预录结果；所有结果均由 patcov.analyze 当场算出，
覆盖证书由独立检查器 patcov.verify_certificate 复核（不复用主搜索缓存）。

段落：
  1. v1 旧入口行为保留（快速对照）；
  2. v2 正常结果：递归列表上的深层遗漏 + 8 种互递归类型主规模；
  3. 三组新能力同用：整数绑定 + or + 守卫的穷尽/遗漏；
  4. 棘手边界：相邻整数区间留单点缺口、负数新序；
  5. 实际触发的拒绝：or 绑定名冲突；
  6. 预算耗尽：v2 给 unknown 与未决状态（不宣称穷尽），v1 仍抛异常；
  7. 覆盖证书：独立检查器复核，且篡改证书必被拒。
"""

from __future__ import annotations

import time

from patcov import (
    CertificateRejected,
    CoverageBudgetExhausted,
    CoverageError,
    analyze,
    verify_certificate,
)


def render(value) -> str:
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, tuple):
        return "(" + ", ".join(render(v) for v in value) + ")"
    return str(value)


def section(title: str) -> None:
    print()
    print("=" * 76)
    print(title)
    print("=" * 76)


def list_types():
    return {"List": {"kind": "sum", "variants": [
        ("Nil", []),
        ("Cons", [("h", "Bool"), ("t", "List")]),
    ]}}


def exact_length_pattern(n):
    p = "Nil"
    for _ in range(n):
        p = ("Cons", "_", p)
    return p


def all_false_value(n):
    v = "Nil"
    for _ in range(n):
        v = ("Cons", False, v)
    return v


def main() -> None:
    overall = time.time()

    # ------------------------------------------------- 1. v1 旧入口保留
    section("1. 兼容：version=1（默认）旧行为完全保留")
    old = {"Shape": {"kind": "sum", "variants": [
        ("Circle", [("r", "Bool")]), ("Dot", []), ("Line", [])]}}
    r1 = analyze(old, [("Circle", "True"), "_", "Dot"])
    print(f"  分支见证：{[render(b.witness) if b.useful else '冗余' for b in r1.branches]}")
    print(f"  穷尽：{r1.exhaustive}（AnalysisResult 形态、异常类型不变）")

    # ------------------------------------------------- 2. v2 递归列表
    section("2. v2 递归列表：固定深度展开无法伪装的‘只漏深层列表’")
    k = 6
    pats = ["Nil"] + [exact_length_pattern(n) for n in range(1, k)]
    r2 = analyze(list_types(), pats, version=2, certificate=True)
    print(f"  分支覆盖长度 0..{k - 1} 的列表（共 {len(pats)} 个分支）")
    print(f"  状态：{r2.status}；题面序最小遗漏（长度 {k}）= {render(r2.uncovered)}")
    summ = verify_certificate(list_types(), pats, r2)
    print(f"  独立检查器复核闭合关系：{summ['states_checked']} 个状态、"
          f"{summ['edges_checked']} 条转移，结论 {summ['conclusion']}")
    print(f"  深度参数族：把 k 改成任意输入值都给确定答案（无内建深度阈值）")

    # 主规模：8 种互递归类型 + 大整数端点
    types8 = {"R0": {"kind": "sum", "variants": [
        ("Stop0", []), ("Go0", [("n", "Big"), ("r", "R1")])]},
        "Big": {"kind": "integer", "range": [-10**9, 10**9]}}
    for i in range(1, 8):
        types8[f"R{i}"] = {"kind": "sum", "variants": [
            (f"Stop{i}", []),
            (f"Go{i}", [("n", "Big"), ("r", f"R{(i + 1) % 8}")])]}
    r8 = analyze(types8, ["Stop0", ("Go0", "_", "_")], version=2, certificate=True)
    print(f"  8 种互递归类型 + 端点 ±10^9：穷尽={r8.status == 'exhaustive'}，"
          f"工作量 {r8.budget_used}（不随值域大小增长）")

    # ------------------------------------------------- 3. 三组新能力同用
    section("3. 同一端到端例子：递归结构中的整数绑定 + or + 守卫")
    # Tree：叶子带整数标签；分支用 or 绑定同一变量 n，守卫只允许常数比较
    Tree = {"Tree": {"kind": "sum", "variants": [
        ("Leaf", [("tag", "Score")]),
        ("Join", [("l", "Tree"), ("r", "Tree")]),
    ], }, "Score": {"kind": "integer", "range": [-10**6, 10**6]}}
    branches = [
        ("Join", "_", "_"),  # 分支0：所有内部节点
        # 分支1：叶子且 (n <= 0 或 n 是单点 5)；or 两边绑定同名同类型 n
        {"pattern": ("Leaf", ("or", ("bind", "n", ("range", -10**6, 0)),
                             ("bind", "n", 5))),
         "when": ["or", ["<=", "n", 0], ["==", "n", 5]]},
        # 分支2：其余叶子（n > 0 且 n != 5），守卫用 and/!= 组合
        {"pattern": ("Leaf", ("bind", "n", "_")),
         "when": ["and", [">", "n", 0], ["!=", "n", 5]]},
    ]
    r3 = analyze(Tree, branches, version=2, certificate=True)
    print(f"  状态：{r3.status}；分支见证：")
    for b in r3.branches:
        print(f"    分支 {b.index}: {'有用，见证 ' + render(b.witness) if b.useful else '冗余'}")
    print(f"  整体最小遗漏：{'无（穷尽）' if r3.uncovered is None else render(r3.uncovered)}")
    print(f"  独立检查器：{verify_certificate(Tree, branches, r3)}")

    # ------------------------------------------------- 4. 棘手边界
    section("4. 棘手边界：相邻整数区间留单点缺口 + 负数新序")
    N = {"N": {"kind": "integer", "range": [-10, 10]}}
    gap_pats = [("range", -10, 4), ("range", 6, 10)]
    r4 = analyze(N, gap_pats, version=2)
    print(f"  区间 [-10,4] 与 [6,10]：唯一缺口见证 = {r4.uncovered}")
    r4b = analyze(N, [0, 1, -1], version=2)
    print(f"  已覆盖 {{-1,0,1}}：最小遗漏 = {r4b.uncovered}"
          "（新序 0,-1,1,-2,2：同绝对值负数先于正数）")
    r4c = analyze(N, [("bind", "x", "_"),
                      {"pattern": ("bind", "x", "_"), "when": ["<=", "x", 9]}],
                  version=2)
    print(f"  恒真分支后再写 x<=9：第二分支冗余={r4c.branches[1].redundant}，"
          f"最小遗漏={r4c.uncovered}（守卫分支也参与精确冗余判定）")

    # ------------------------------------------------- 5. 实际拒绝
    section("5. 拒绝边界：or 两边绑定名称不一致（可定位错误，不静默纠正）")
    try:
        analyze(N, [("or", ("bind", "x", 0), ("bind", "y", 1))], version=2)
    except CoverageError as exc:
        print(f"  捕获 CoverageError：{exc}")
    else:
        raise AssertionError("or 绑定冲突必须被拒绝")
    # 守卫引用未绑定变量
    try:
        analyze(N, [{"pattern": "_", "when": ["<=", "z", 0]}], version=2)
    except CoverageError as exc:
        print(f"  捕获 CoverageError：{exc}")

    # ------------------------------------------------- 6. 预算 unknown
    section("6. 预算边界：耗尽给 unknown 与未决状态，绝不宣称穷尽")
    # v1 语义不变：抛 CoverageBudgetExhausted
    try:
        analyze({"X": {"kind": "enum", "constructors": ["A", "B", "C"]}},
                ["A"], budget=1)
    except CoverageBudgetExhausted as exc:
        print(f"  v1：抛 CoverageBudgetExhausted（used={exc.used}）")
    # v2：status=unknown + pending
    ru = analyze(N, [("range", -10, 10)], version=2, budget=1)
    print(f"  v2：status={ru.status}，exhaustive={ru.exhaustive}，"
          f"未决问题 {len(ru.pending)} 个（现场可查、可加大预算重试）")
    print(f"  未决示例问题：{ru.pending[0]['question']}，"
          f"列={ru.pending[0]['state']['columns']}")
    ok = analyze(N, [("range", -10, 10)], version=2)
    print(f"  同输入预算充足：status={ok.status}（反证 unknown ≠ 穷尽）")

    # ------------------------------------------------- 7. 证书不可信任标志
    section("7. 覆盖证书：检查器独立核对，篡改结论/权值必被拒")
    import dataclasses
    base = analyze(list_types(), ["Nil", ("Cons", "True", "_")],
                   version=2, certificate=True)
    print(f"  原始结论 {base.status}，最小遗漏 {render(base.uncovered)}")
    tampered = dataclasses.replace(base, status="exhaustive",
                                   uncovered=None, exhaustive=True)
    try:
        verify_certificate(list_types(), ["Nil", ("Cons", "True", "_")], tampered)
    except CertificateRejected as exc:
        print(f"  篡改结论为 exhaustive 被拒：{exc}")
    cert = dict(base.certificate)
    answers = [dict(a) for a in cert["state_answers"]]
    for a in answers:
        if "weight" in a:
            a["weight"] += 999
    cert["state_answers"] = answers
    tampered2 = dataclasses.replace(base, certificate=cert)
    try:
        verify_certificate(list_types(), ["Nil", ("Cons", "True", "_")], tampered2)
    except CertificateRejected as exc:
        print(f"  篡改状态最优权值被拒：{exc}")

    print()
    print(f"演示总耗时 {time.time() - overall:.1f} 秒（要求 90 秒内；无 sleep）。")


if __name__ == "__main__":
    main()
