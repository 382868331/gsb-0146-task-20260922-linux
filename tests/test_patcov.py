"""必要测试。

期望值来自 tests/_reference.py 中*独立实现*的暴力参考：自行解析类型与模式、
显式枚举整个有限值域并独立匹配，不导入也不调用 patcov 的核心算法
（solver/analyze）。往返性质（见证确实未被前序覆盖且被本分支匹配）仅作补充。
"""

from __future__ import annotations

import math
import random
import unittest

from patcov import (
    AnalysisResult,
    CoverageBudgetExhausted,
    CoverageError,
    analyze,
)

from _reference import RefTypeTable, brute_force_expected, matches


def T(**kw):
    return kw


BOOL = {"kind": "bool"}  # 仅用于参考；被测库中 Bool 是内置名


BOOL_TYPES = {"Bool": {"kind": "bool"}}  # 内置 Bool 根的公开写法


# ----------------------------------------------------------------- 手核对小例

class TestHandChecked(unittest.TestCase):
    def test_enum_exhaustive(self):
        r = analyze(
            {"Answer": {"kind": "enum", "constructors": ["No", "Yes"]}},
            ["No", "Yes"],
        )
        self.assertTrue(r.exhaustive)
        self.assertIsNone(r.uncovered)
        self.assertEqual([(b.useful, b.witness) for b in r.branches],
                         [(True, "No"), (True, "Yes")])
        self.assertEqual(r.redundant_branches, ())

    def test_builtin_bool_root(self):
        # 直接以内置 Bool 为根：False < True
        r = analyze(BOOL_TYPES, ["True"])
        self.assertFalse(r.exhaustive)
        self.assertIs(r.uncovered, False)
        self.assertIs(r.branches[0].witness, True)

    def test_bool_first_wild_makes_rest_redundant(self):
        r = analyze(BOOL_TYPES, ["_", "False", "True"])
        self.assertTrue(r.exhaustive)
        self.assertTrue(r.branches[0].useful)
        self.assertTrue(all(b.redundant for b in r.branches[1:]))
        self.assertEqual(r.redundant_branches, (1, 2))

    def test_bool_partial_min_uncovered_false(self):
        r = analyze(BOOL_TYPES, ["True"])
        self.assertIs(r.uncovered, False)
        self.assertIs(r.branches[0].witness, True)

    def test_or_duplicates(self):
        # or 备选重复不影响结果；or 内部备选之间互不算冗余
        r = analyze(BOOL_TYPES, [("or", "True", "True", "False")])
        self.assertTrue(r.exhaustive)
        self.assertTrue(r.branches[0].useful)
        # 最小见证取字典序最小备选
        self.assertIs(r.branches[0].witness, False)

    def test_or_partial_then_wild(self):
        r = analyze(BOOL_TYPES, [("or", "True"), ("or", "False"), "True"])
        self.assertTrue(r.exhaustive)
        self.assertEqual(
            [(b.useful, b.witness) for b in r.branches],
            [(True, True), (True, False), (False, None)],
        )
        self.assertEqual(r.redundant_branches, (2,))

    def test_product_partial_coverage(self):
        types = {
            "Pair": {"kind": "product", "fields": [("a", "Bool"), ("b", "Bool")]},
        }
        r = analyze(types, [["True", "_"]])
        # 分支只匹配 a=True；最小见证（本分支匹配且未覆盖）= (True, False)
        self.assertEqual(r.branches[0].witness, (True, False))
        # 整体最小遗漏 = (False, False)
        self.assertEqual(r.uncovered, (False, False))
        r2 = analyze(types, [["True", "_"], ["_", "True"], ["False", "False"]])
        self.assertTrue(r2.exhaustive)
        self.assertEqual(
            [b.witness for b in r2.branches],
            [(True, False), (False, True), (False, False)],
        )
        self.assertEqual(r2.redundant_branches, ())

    def test_product_cross_field_not_collapsed(self):
        # (True,_) 与 (_,True) 不能覆盖 (False,False) —— 跨字段关联关键例
        types = {"P": {"kind": "product", "fields": [("a", "Bool"), ("b", "Bool")]}}
        r = analyze(types, [["True", "_"], ["_", "True"]])
        self.assertEqual(r.uncovered, (False, False))
        self.assertFalse(r.exhaustive)

    def test_never_matching_constructor(self):
        types = {"Color": {"kind": "enum", "constructors": ["Red", "Green", "Blue"]}}
        r = analyze(types, ["Red", "_", "Green"])
        self.assertTrue(r.exhaustive)
        self.assertEqual(
            [(b.useful, b.witness) for b in r.branches],
            [(True, "Red"), (True, "Green"), (False, None)],
        )
        self.assertEqual(r.redundant_branches, (2,))

    def test_unreachable_sum_variant_after_wild(self):
        types = {
            "Shape": {
                "kind": "sum",
                "variants": [("Circle", [("r", "Bool")]), ("Dot", []), ("Line", [])],
            },
        }
        r = analyze(types, [("Circle", "True"), "_", "Dot"])
        self.assertTrue(r.exhaustive)
        # 分支 0 只匹配 (Circle, True)，见证必须被本分支匹配
        self.assertEqual(r.branches[0].witness, ("Circle", True))
        # 通配的最小未覆盖见证是 (Circle, False)，Dot/Line 更靠后
        self.assertTrue(r.branches[1].useful)
        self.assertEqual(r.branches[1].witness, ("Circle", False))
        # Dot 在通配之后永不匹配
        self.assertTrue(r.branches[2].redundant)

    def test_sum_partial_variant_fields(self):
        types = {
            "Opt": {"kind": "sum", "variants": [
                ("None", []),
                ("Some", [("v", "Bool")]),
            ]},
        }
        r = analyze(types, [("Some", "True")])
        self.assertEqual(r.uncovered, "None")  # None 编号在前
        self.assertEqual(r.branches[0].witness, ("Some", True))  # 该分支唯一匹配值
        r2 = analyze(types, ["None", ("Some", "True")])
        self.assertEqual(r2.uncovered, ("Some", False))
        self.assertEqual(r2.branches[1].witness, ("Some", True))
        r3 = analyze(types, ["None", ("Some", "_")])
        self.assertTrue(r3.exhaustive)
        self.assertEqual(r3.branches[1].witness, ("Some", False))

    def test_lex_order_constructor_index_then_fields(self):
        types = {
            "E": {"kind": "enum", "constructors": ["A", "B", "C"]},
        }
        r = analyze(types, ["B", "C"])
        self.assertEqual(r.uncovered, "A")
        r = analyze(types, ["A", "C"])
        self.assertEqual(r.uncovered, "B")

    def test_nested_sum_product(self):
        # 根必须是首项：Outer 在前（字段类型 Inner 前向引用，库支持）
        types = {
            "Outer": {"kind": "product", "fields": [("i", "Inner"), ("k", "Bool")]},
            "Inner": {"kind": "sum", "variants": [("X", []), ("Y", [("b", "Bool")])]},
        }
        # 外层为积，见证形态：(Inner 值, Bool 值)
        r = analyze(types, [["X", "True"], [("Y", "_"), "_"]])
        # 分支 0 只匹配 (X, True)
        self.assertEqual(r.branches[0].witness, ("X", True))
        # 分支 1 最小匹配 = (("Y", False), False)
        self.assertEqual(r.branches[1].witness, (("Y", False), False))
        # 整体最小遗漏 = ("X", False)
        self.assertEqual(r.uncovered, ("X", False))
        self.assertFalse(r.exhaustive)


def _analyze_bool(pats):
    return analyze(BOOL_TYPES, pats)


# ----------------------------------------------------------------- 巨大 Bool 积

class TestHugeBoolProduct(unittest.TestCase):
    N = 500  # 值域 2^500，绝不允许展开

    def setUp(self):
        self.types = {
            "Big": {"kind": "product",
                    "fields": [(f"b{i}", "Bool") for i in range(self.N)]},
        }

    def test_single_all_true_branch(self):
        true_row = ["True"] * self.N
        r = analyze(self.types, [true_row])
        self.assertFalse(r.exhaustive)
        # 整体字典序最小遗漏 = 全 False
        self.assertEqual(r.uncovered, (False,) * self.N)
        # 该分支唯一匹配值 = 全 True（其见证即此）
        self.assertEqual(r.branches[0].witness, (True,) * self.N)

    def test_wild_covers_all_and_second_row_redundant(self):
        r = analyze(self.types, [["_"] * self.N, ["True"] * self.N])
        self.assertTrue(r.exhaustive)
        self.assertIsNone(r.uncovered)
        self.assertTrue(r.branches[0].useful)
        self.assertEqual(r.branches[0].witness, (False,) * self.N)
        self.assertTrue(r.branches[1].redundant)

    def test_singleton_row_min_uncovered(self):
        # 仅覆盖一个值（第 0 位 False、其余 True）：
        row = ["False"] + ["True"] * (self.N - 1)
        r = analyze(self.types, [row])
        # 分支见证即其唯一匹配值
        self.assertEqual(r.branches[0].witness, (False,) + (True,) * (self.N - 1))
        # 最小遗漏 = 全 False
        self.assertEqual(r.uncovered, (False,) * self.N)

    def test_first_field_false_then_wild_row(self):
        # 行 0 覆盖“第 0 位 False”的全部 2^(N-1) 个值：
        r = analyze(self.types, [["False"] + ["_"] * (self.N - 1)])
        self.assertEqual(r.branches[0].witness, (False,) * self.N)
        # 最小遗漏第 0 位为 True，其余取最小 False
        self.assertEqual(r.uncovered, (True,) + (False,) * (self.N - 1))

    def test_exact_counts_huge_product(self):
        # 精确计数（大整数符号运算，绝不逐值展开）：
        rows = [
            ["False"] + ["_"] * (self.N - 1),       # 0: 2^(N-1)
            ["True"] * self.N,                        # 1: 1
            ["True", "_"] + ["_"] * (self.N - 2),   # 2: 已被 0/1 占满 True 半空间外?
        ]
        r = analyze(self.types, rows)
        half = 1 << (self.N - 1)
        self.assertEqual(r.total_count, 1 << self.N)
        self.assertEqual(r.branch_counts, (half, 1, half - 1))
        self.assertEqual(r.covered_count, half + 1 + half - 1)
        self.assertEqual(r.covered_count, 1 << self.N)
        self.assertEqual(r.uncovered_count, 0)
        self.assertTrue(r.exhaustive)

    def test_exact_counts_partial(self):
        r = analyze(self.types, [["True"] * self.N])
        self.assertEqual(r.branch_counts, (1,))
        self.assertEqual(r.uncovered_count, (1 << self.N) - 1)
        self.assertEqual(r.total_count, 1 << self.N)


# ----------------------------------------------------------------- 精确计数核对

class TestExactCounts(unittest.TestCase):
    def test_sum_counts(self):
        types = {
            "Shape": {"kind": "sum", "variants": [
                ("Circle", [("r", "Bool")]),
                ("Rect", [("w", "Bool"), ("h", "Bool")]),
                ("Dot", [])]},
        }
        # 全集：Circle 2 + Rect 4 + Dot 1 = 7
        r = analyze(types, [("Circle", "True"), ("Rect", "True", "_"),
                            ("or", "Dot", ("Circle", "False")),
                            ("Rect", "_", "False")])
        self.assertEqual(r.total_count, 7)
        self.assertEqual(r.branch_counts, (1, 2, 2, 1))
        self.assertEqual(r.uncovered_count, 1)  # 仅剩 (Rect, False, True)
        self.assertEqual(r.uncovered, ("Rect", False, True))
        self.assertEqual(sum(r.branch_counts) + r.uncovered_count, 7)

    def test_redundant_branch_count_zero(self):
        types = {"Color": {"kind": "enum", "constructors": ["Red", "Green", "Blue"]}}
        r = analyze(types, ["_", "Red"])
        self.assertEqual(r.branch_counts, (3, 0))
        self.assertEqual(r.uncovered_count, 0)
        self.assertTrue(r.exhaustive)
        self.assertEqual(r.branches[1].count, 0)

    def test_no_patterns_counts_all_uncovered(self):
        types = {"Color": {"kind": "enum", "constructors": ["A", "B"]}}
        r = analyze(types, [])
        self.assertEqual(r.total_count, 2)
        self.assertEqual(r.branch_counts, ())
        self.assertEqual(r.covered_count, 0)
        self.assertEqual(r.uncovered_count, 2)

    def test_enum_nested_counts(self):
        types = {"Box": {"kind": "product", "fields": [
            ("a", "E"), ("b", "Bool")]},
            "E": {"kind": "enum", "constructors": ["X", "Y", "Z"]}}
        r = analyze(types, [["X", "True"], ["_", "False"]])
        # 全集 3*2=6；分支0 占 1；分支1 占 3（a 任取，b=False，含 (X,False)）
        self.assertEqual(r.total_count, 6)
        self.assertEqual(r.branch_counts, (1, 3))
        self.assertEqual(r.uncovered_count, 2)
        self.assertEqual(r.uncovered, ("Y", True))


# ----------------------------------------------------------------- 与独立暴力参考

class TestAgainstBruteForce(unittest.TestCase):
    """小类型上与独立枚举参考逐值核对，含随机 or 模式。"""

    SCENARIOS = [
        # (类型表, 随机模式池生成所用根类型深度)
        ({"Day": {"kind": "enum", "constructors": ["Mon", "Tue", "Wed"]}}, "Day", 60),
        ({"Pair": {"kind": "product",
                    "fields": [("a", "Bool"), ("b", "Bool")]}}, "Pair", 120),
        ({"Opt": {"kind": "sum", "variants": [
            ("None", []), ("Some", [("v", "Bool")])]}}, "Opt", 120),
        ({"Outer": {"kind": "product", "fields": [("i", "Inner"), ("k", "Bool")]},
          "Inner": {"kind": "sum", "variants": [("X", []), ("Y", [("b", "Bool")])]}},
         "Outer", 200),
        ({"Tri": {"kind": "product", "fields": [
            ("a", "Bool"), ("b", "Bool"), ("c", "Bool")]}}, "Tri", 200),
        ({"Box": {"kind": "sum", "variants": [
            ("P", [("x", "Bool"), ("y", "E")]),
            ("Q", [("z", "E")]),
            ("R", [])]},
          "E": {"kind": "enum", "constructors": ["A", "B"]}}, "Box", 200),
    ]

    def test_randomized(self):
        rng = random.Random(20260922)
        for types, root, trials in self.SCENARIOS:
            ref = RefTypeTable(types)
            values = ref.all_values(root)
            for t in range(trials):
                pats = [_random_pattern(rng, ref, root, depth=0)
                        for _ in range(rng.randint(1, 5))]
                # 30% 概率注入 or 包装（含重复备选）
                final = []
                for p in pats:
                    if rng.random() < 0.35:
                        k = rng.randint(1, 3)
                        alts = [p] + [_random_pattern(rng, ref, root, 0)
                                      for _ in range(k - 1)]
                        if rng.random() < 0.5:
                            alts.append(p)  # 重复备选
                        p = ("or", *alts)
                    final.append(p)
                got = analyze(types, final)
                exp = brute_force_expected(ref, root, final)
                self._compare(got, exp, types, final, values, ref, root)

    def _compare(self, got, exp, types, pats, values, ref, root):
        msg = f"\n类型={types}\n模式={pats}"
        self.assertEqual(got.uncovered, exp["uncovered"], msg)
        self.assertEqual(got.exhaustive, exp["uncovered"] is None, msg)
        self.assertEqual(len(got.branches), len(exp["branches"]), msg)
        # 精确计数核对：各分支首命中数、遗漏数、全集总数互相核对
        self.assertEqual(got.total_count, exp["total_count"], msg)
        self.assertEqual(got.uncovered_count, exp["uncovered_count"], msg)
        self.assertEqual(got.branch_counts, tuple(exp["branch_counts"]), msg)
        self.assertEqual(
            sum(got.branch_counts) + got.uncovered_count, got.total_count, msg
        )
        for b, eb in zip(got.branches, exp["branches"]):
            self.assertEqual(b.useful, eb["useful"], msg + f"\n分支 {b.index}")
            self.assertEqual(b.witness, eb["witness"], msg + f"\n分支 {b.index}")
            self.assertEqual(b.count, eb["count"], msg + f"\n分支 {b.index}")
            self.assertEqual(b.count > 0, b.useful, msg + f"\n分支 {b.index}")
            if eb["useful"]:
                # 补充往返：见证未被前序分支匹配、且被本分支匹配
                self.assertFalse(
                    any(matches(ref, root, q, eb["witness"]) for q in pats[:b.index]),
                    msg + f"\n分支 {b.index} 见证被前序覆盖",
                )
                self.assertTrue(
                    matches(ref, root, pats[b.index], eb["witness"]),
                    msg + f"\n分支 {b.index} 见证不被本分支匹配",
                )


def _random_pattern(rng, ref: RefTypeTable, tname: str, depth: int):
    """在参考类型上生成随机原始模式。"""
    t = ref.types[tname]
    kind = t["kind"]
    roll = rng.random()
    if depth >= 3 or roll < 0.35:
        return "_"
    if kind == "bool":
        return rng.choice(["True", "False"])
    if kind == "enum":
        return rng.choice(t["constructors"])
    if kind == "product":
        return [_random_pattern(rng, ref, ft, depth + 1) for _, ft in t["fields"]]
    # sum
    label, fields = rng.choice(t["variants"])
    if not fields:
        return label
    return (label, *[_random_pattern(rng, ref, ft, depth + 1) for _, ft in fields])


# ----------------------------------------------------------------- 输入错误语义

class TestErrorSemantics(unittest.TestCase):
    def test_types_not_dict(self):
        with self.assertRaisesRegex(CoverageError, r"types"):
            analyze([], ["_"])

    def test_empty_types(self):
        with self.assertRaisesRegex(CoverageError, r"为空"):
            analyze({}, ["_"])

    def test_bad_kind(self):
        with self.assertRaisesRegex(CoverageError, r"kind"):
            analyze({"X": {"kind": "weird"}}, ["_"])

    def test_redeclare_bool(self):
        with self.assertRaisesRegex(CoverageError, r"Bool"):
            analyze({"Bool": {"kind": "enum", "constructors": ["a"]}}, ["_"])

    def test_user_kind_bool(self):
        with self.assertRaisesRegex(CoverageError, r"bool"):
            analyze({"X": {"kind": "bool"}}, ["_"])

    def test_duplicate_constructor(self):
        with self.assertRaisesRegex(CoverageError, r"重复"):
            analyze({"X": {"kind": "enum", "constructors": ["a", "a"]}}, ["_"])

    def test_duplicate_field(self):
        with self.assertRaisesRegex(CoverageError, r"重复"):
            analyze({"X": {"kind": "product",
                            "fields": [["f", "Bool"], ["f", "Bool"]]}}, [["_", "_"]])

    def test_unknown_type_reference(self):
        with self.assertRaisesRegex(CoverageError, r"未知类型"):
            analyze({"X": {"kind": "product", "fields": [["f", "Nope"]]}}, [["_"]])

    def test_recursive_type_rejected(self):
        types = {"X": {"kind": "sum", "variants": [("C", [["x", "X"]]), ("D", [])]}}
        with self.assertRaisesRegex(CoverageError, r"递归"):
            analyze(types, ["D"])

    def test_nesting_limit(self):
        # L0 为枚举，L1..L13 为逐层 product，L13 字段嵌套深度 = 13 > 12
        types = {"L0": {"kind": "enum", "constructors": ["Z"]}}
        for i in range(1, 14):
            types[f"L{i}"] = {"kind": "product",
                              "fields": [["v", f"L{i-1}"]]}
        with self.assertRaisesRegex(CoverageError, r"嵌套深度"):
            analyze(types, ["_"])

    def test_constructor_count_limit(self):
        with self.assertRaisesRegex(CoverageError, r"128"):
            analyze({"X": {"kind": "enum",
                            "constructors": [f"c{i}" for i in range(129)]}}, ["_"])

    def test_field_arity_limit(self):
        from patcov import MAX_FIELD_ARITY
        over = MAX_FIELD_ARITY + 1
        with self.assertRaisesRegex(CoverageError, r"字段数"):
            analyze({"X": {"kind": "product",
                            "fields": [[f"f{i}", "Bool"] for i in range(over)]}}, ["_"])
        with self.assertRaisesRegex(CoverageError, r"字段数"):
            analyze({"X": {"kind": "sum",
                            "variants": [("C", [[f"f{i}", "Bool"]
                                               for i in range(over)])]}}, [("C",)])

    def test_pattern_bad_type(self):
        types = {"X": {"kind": "enum", "constructors": ["A"]}}
        with self.assertRaisesRegex(CoverageError, r"patterns\[0\]"):
            analyze(types, [123])

    def test_pattern_bool_rejected(self):
        types = {"X": {"kind": "enum", "constructors": ["A"]}}
        with self.assertRaisesRegex(CoverageError, r"patterns\[1\]"):
            analyze(types, ["A", True])

    def test_unknown_constructor_location(self):
        types = {"X": {"kind": "enum", "constructors": ["A"]}}
        with self.assertRaisesRegex(CoverageError, r"patterns\[0\].*B"):
            analyze(types, ["B"])

    def test_ctor_arity_mismatch(self):
        types = {"X": {"kind": "sum", "variants": [("C", [["f", "Bool"]])]}}
        with self.assertRaisesRegex(CoverageError, r"需要 1 个字段"):
            analyze(types, [("C",)])
        with self.assertRaisesRegex(CoverageError, r"需要 1 个字段"):
            analyze(types, [("C", "True", "False")])

    def test_ctor_on_product(self):
        types = {"P": {"kind": "product", "fields": [["f", "Bool"]]}}
        with self.assertRaisesRegex(CoverageError, r"product"):
            analyze(types, [("Nope", "True")])

    def test_product_pattern_arity(self):
        types = {"P": {"kind": "product", "fields": [["a", "Bool"], ["b", "Bool"]]}}
        with self.assertRaisesRegex(CoverageError, r"字段数"):
            analyze(types, [["True"]])

    def test_list_pattern_on_non_product(self):
        types = {"X": {"kind": "enum", "constructors": ["A"]}}
        with self.assertRaisesRegex(CoverageError, r"积模式"):
            analyze(types, [["_"]])

    def test_empty_tuple_pattern(self):
        types = {"X": {"kind": "enum", "constructors": ["A"]}}
        with self.assertRaisesRegex(CoverageError, r"空 tuple"):
            analyze(types, [()])

    def test_or_without_alternative(self):
        types = {"X": {"kind": "enum", "constructors": ["A"]}}
        with self.assertRaisesRegex(CoverageError, r"or"):
            analyze(types, [("or",)])

    def test_patterns_not_sequence(self):
        types = {"X": {"kind": "enum", "constructors": ["A"]}}
        with self.assertRaisesRegex(CoverageError, r"patterns"):
            analyze(types, {"A": 1})

    def test_unknown_key(self):
        with self.assertRaisesRegex(CoverageError, r"未知键"):
            analyze({"X": {"kind": "enum", "constructors": ["A"], "extra": 1}}, ["A"])

    # ------------------------------------------------- 数值参数：NaN/Inf/bool
    def test_budget_rejects_nan(self):
        types = {"X": {"kind": "enum", "constructors": ["A"]}}
        with self.assertRaises(CoverageError):
            analyze(types, ["A"], budget=float("nan"))

    def test_budget_rejects_infinity(self):
        types = {"X": {"kind": "enum", "constructors": ["A"]}}
        with self.assertRaises(CoverageError):
            analyze(types, ["A"], budget=math.inf)
        with self.assertRaises(CoverageError):
            analyze(types, ["A"], budget=-math.inf)

    def test_budget_rejects_bool(self):
        types = {"X": {"kind": "enum", "constructors": ["A"]}}
        with self.assertRaisesRegex(CoverageError, r"budget"):
            analyze(types, ["A"], budget=True)

    def test_budget_rejects_zero_and_negative(self):
        types = {"X": {"kind": "enum", "constructors": ["A"]}}
        with self.assertRaises(CoverageError):
            analyze(types, ["A"], budget=0)

    def test_budget_exhaustion_is_distinct_from_proof(self):
        # 预算耗尽是 CoverageBudgetExhausted，绝不能与“穷举=None”混淆
        types = {"X": {"kind": "enum", "constructors": ["A", "B", "C"]}}
        with self.assertRaises(CoverageBudgetExhausted) as cm:
            analyze(types, ["A"], budget=1)
        self.assertEqual(cm.exception.budget, 1)
        self.assertGreaterEqual(cm.exception.used, 1)

    def test_budget_sufficient_works(self):
        types = {"X": {"kind": "enum", "constructors": ["A", "B", "C"]}}
        r = analyze(types, ["A", "B", "C"], budget=10_000)
        self.assertTrue(r.exhaustive)
        self.assertLessEqual(r.budget_used, 10_000)


# ----------------------------------------------------------------- 无状态/失败原子性

class TestNoPartialState(unittest.TestCase):
    def test_repeated_calls_independent(self):
        types = {"X": {"kind": "enum", "constructors": ["A", "B"]}}
        r1 = analyze(types, ["A"])
        r2 = analyze(types, ["A"])
        self.assertEqual(r1.uncovered, r2.uncovered)
        self.assertEqual(r1.branches[0].witness, r2.branches[0].witness)

    def test_empty_patterns_uncovers_min(self):
        types = {"X": {"kind": "enum", "constructors": ["A", "B"]}}
        r = analyze(types, [])
        self.assertEqual(r.uncovered, "A")
        self.assertFalse(r.exhaustive)
        self.assertEqual(r.branches, ())

    def test_result_types(self):
        types = {"X": {"kind": "enum", "constructors": ["A", "B"]}}
        r = analyze(types, ["A"])
        self.assertIsInstance(r, AnalysisResult)
        self.assertEqual(r.budget, 50_000_000)


if __name__ == "__main__":
    unittest.main()
