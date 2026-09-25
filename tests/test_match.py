"""绑定（var/as）与编译/执行入口（compile_matcher / Matcher.match）的测试。

期望值来源：
- 小类型上由 tests/_reference.py 的*独立*递归绑定匹配（match_bindings /
  first_hit）逐值核对——它不导入 patcov 核心；
- “绑定视为通配”用参考里的独立 strip_public 生成纯模式对照 analyze。
"""

from __future__ import annotations

import random
import unittest

from patcov import (
    CoverageError,
    MatchResult,
    analyze,
    compile_matcher,
)

from _reference import RefTypeTable, brute_force_expected, first_hit, strip_public


BOOL_TYPES = {"Bool": {"kind": "bool"}}

SHAPE_TYPES = {
    "Shape": {
        "kind": "sum",
        "variants": [
            ("Circle", [("r", "Bool")]),
            ("Rect", [("w", "Bool"), ("h", "Bool")]),
            ("Dot", []),
        ],
    },
}

# 嵌套积与和：根为积，字段含和；和的变体字段又含积
NESTED_TYPES = {
    "Outer": {"kind": "product", "fields": [("i", "Inner"), ("k", "Bool")]},
    "Inner": {"kind": "sum", "variants": [("X", []), ("Y", [("p", "Pair")])]},
    "Pair": {"kind": "product", "fields": [("a", "Bool"), ("b", "Bool")]},
}


# ----------------------------------------------------------------- 手核对：绑定

class TestBindingHandChecked(unittest.TestCase):
    def test_var_binds_whole_root_value(self):
        m = compile_matcher(SHAPE_TYPES, [("var", "s")])
        hit = m.match(("Rect", True, False))
        self.assertEqual(hit.branch, 0)
        self.assertEqual(hit.bindings, {"s": ("Rect", True, False)})

    def test_var_inside_constructor_fields(self):
        m = compile_matcher(SHAPE_TYPES, [("Rect", ("var", "w"), ("var", "h"))])
        hit = m.match(("Rect", True, False))
        self.assertEqual(hit.bindings, {"w": True, "h": False})
        # 其他变体不命中
        self.assertIsNone(m.match(("Circle", True)))
        self.assertIsNone(m.match("Dot"))

    def test_as_captures_whole_subtree(self):
        # as 捕获完整子树：外层 as 绑定整个 Rect 值，内层 var 绑定各字段
        m = compile_matcher(
            SHAPE_TYPES,
            [("as", "whole", ("Rect", ("var", "w"), "True"))],
        )
        hit = m.match(("Rect", False, True))
        self.assertEqual(hit.branch, 0)
        self.assertEqual(
            hit.bindings, {"w": False, "whole": ("Rect", False, True)}
        )
        # 子模式不命中则 as 也不命中
        self.assertIsNone(m.match(("Rect", False, False)))

    def test_nested_product_and_sum_bindings(self):
        # 嵌套积与和：积字段里的和变体、和变体字段里的积，逐层绑定
        m = compile_matcher(
            NESTED_TYPES,
            [
                [("Y", ("as", "p", [("var", "a"), "True"])), ("var", "k")],
                ["_", ("var", "k")],
            ],
        )
        hit = m.match((("Y", (True, True)), False))
        self.assertEqual(hit.branch, 0)
        self.assertEqual(hit.bindings, {"a": True, "p": (True, True), "k": False})
        # 内层积第二字段不是 True → 落到分支 1
        hit2 = m.match((("Y", (True, False)), True))
        self.assertEqual(hit2.branch, 1)
        self.assertEqual(hit2.bindings, {"k": True})

    def test_or_same_name_in_both_alternatives(self):
        # 同名绑定出现在两个 or 备选：名字与类型一致，合法
        m = compile_matcher(
            NESTED_TYPES,
            [[("or", ("as", "v", "X"), ("as", "v", ("Y", "_"))), ("var", "k")]],
        )
        hit = m.match(("X", True))
        self.assertEqual(hit.bindings, {"v": "X", "k": True})
        hit2 = m.match((("Y", (False, False)), False))
        self.assertEqual(hit2.bindings, {"v": ("Y", (False, False)), "k": False})

    def test_or_var_alternatives_bind_value(self):
        # 两个备选都以 as 绑定整个 Opt 值：同名同类型，合法
        m = compile_matcher(
            {"Opt": {"kind": "sum", "variants": [
                ("None", []), ("Some", [("v", "Bool")])]}},
            [("or", ("as", "x", ("Some", "_")), ("as", "x", "None"))],
        )
        self.assertEqual(m.match(("Some", True)).bindings, {"x": ("Some", True)})
        self.assertEqual(m.match("None").bindings, {"x": "None"})

    def test_first_match_wins_source_order(self):
        # 两个分支都能命中时，源顺序在前者胜
        m = compile_matcher(BOOL_TYPES, [("var", "a"), ("var", "b")])
        hit = m.match(True)
        self.assertEqual(hit.branch, 0)
        self.assertEqual(hit.bindings, {"a": True})

    def test_no_match_returns_none(self):
        m = compile_matcher(BOOL_TYPES, ["True"])
        self.assertIsNone(m.match(False))

    def test_empty_patterns_no_match(self):
        m = compile_matcher(BOOL_TYPES, [])
        self.assertIsNone(m.match(True))
        self.assertEqual(m.branch_count, 0)

    def test_no_shared_capture_slots_across_branches(self):
        # 分支 0 先绑定 a 再失败；分支 1 的绑定字典不得混入 a
        m = compile_matcher(
            {"P": {"kind": "sum", "variants": [
                ("C", [("x", "Bool"), ("y", "Bool")])]}},
            [("C", ("var", "a"), "False"), ("C", ("var", "b"), "_")],
        )
        hit = m.match(("C", True, True))
        self.assertEqual(hit.branch, 1)
        self.assertEqual(hit.bindings, {"b": True})
        # 重复执行互不影响：返回值各自独立
        again = m.match(("C", True, True))
        self.assertEqual(again.bindings, {"b": True})
        hit.bindings["b"] = "MUTATED"
        self.assertEqual(m.match(("C", True, True)).bindings, {"b": True})

    def test_result_type(self):
        m = compile_matcher(BOOL_TYPES, ["_"])
        hit = m.match(False)
        self.assertIsInstance(hit, MatchResult)
        self.assertEqual(hit.branch, 0)
        self.assertEqual(hit.bindings, {})


# ----------------------------------------------------------------- 绑定规则拒绝

class TestBindingRuleErrors(unittest.TestCase):
    def test_duplicate_name_on_single_path(self):
        with self.assertRaisesRegex(CoverageError, r"重复绑定"):
            compile_matcher(
                {"Opt": {"kind": "sum", "variants": [
                    ("None", []), ("Some", [("v", "Bool")])]}},
                [("as", "x", ("Some", ("var", "x")))],
            )

    def test_duplicate_name_across_product_fields(self):
        with self.assertRaisesRegex(CoverageError, r"重复绑定"):
            compile_matcher(
                {"P": {"kind": "product",
                        "fields": [("a", "Bool"), ("b", "Bool")]}},
                [[("var", "x"), ("var", "x")]],
            )

    def test_or_alternatives_different_names_rejected(self):
        with self.assertRaisesRegex(CoverageError, r"完全相同"):
            compile_matcher(BOOL_TYPES, [("or", ("var", "x"), ("var", "y"))])

    def test_or_alternative_missing_binding_rejected(self):
        with self.assertRaisesRegex(CoverageError, r"完全相同"):
            compile_matcher(BOOL_TYPES, [("or", ("var", "x"), "_")])

    def test_or_same_name_different_types_rejected(self):
        # 同名 x 在一个备选里是 Bool、另一个里是 Color → 类型不同，拒绝
        types = {
            "S": {"kind": "sum", "variants": [
                ("C", [("f", "Bool")]), ("D", [("g", "Color")])]},
            "Color": {"kind": "enum", "constructors": ["R", "G", "B"]},
        }
        with self.assertRaisesRegex(CoverageError, r"完全相同"):
            compile_matcher(
                types, [("or", ("C", ("var", "x")), ("D", ("var", "x")))]
            )

    def test_or_binding_errors_also_rejected_by_analyze(self):
        # “分析阶段拒绝”：analyze 走同一规范化，同样报错
        with self.assertRaisesRegex(CoverageError, r"完全相同"):
            analyze(BOOL_TYPES, [("or", ("var", "x"), "True")])

    def test_binding_name_reserved(self):
        for bad in ("_", "or", "var", "as"):
            with self.assertRaisesRegex(CoverageError, r"保留字"):
                compile_matcher(BOOL_TYPES, [("var", bad)])

    def test_binding_name_not_string(self):
        with self.assertRaisesRegex(CoverageError, r"绑定名"):
            compile_matcher(BOOL_TYPES, [("var", 7)])

    def test_var_arity(self):
        with self.assertRaisesRegex(CoverageError, r"var"):
            compile_matcher(BOOL_TYPES, [("var",)])
        with self.assertRaisesRegex(CoverageError, r"var"):
            compile_matcher(BOOL_TYPES, [("var", "x", "extra")])

    def test_as_arity(self):
        with self.assertRaisesRegex(CoverageError, r"as"):
            compile_matcher(BOOL_TYPES, [("as", "x")])

    def test_constructor_named_var_now_reserved(self):
        with self.assertRaisesRegex(CoverageError, r"保留字"):
            compile_matcher(
                {"E": {"kind": "enum", "constructors": ["var"]}}, ["var"]
            )


# ----------------------------------------------------------------- 错误类型输入

class TestValueTypeErrors(unittest.TestCase):
    def test_bool_root_rejects_non_bool(self):
        m = compile_matcher(BOOL_TYPES, ["_"])
        for bad in (0, 1, "True", None, ()):
            with self.assertRaisesRegex(CoverageError, r"value"):
                m.match(bad)

    def test_enum_rejects_unknown_label_and_non_str(self):
        m = compile_matcher(
            {"Color": {"kind": "enum", "constructors": ["R", "G"]}}, ["_"]
        )
        with self.assertRaisesRegex(CoverageError, r"value"):
            m.match("B")
        with self.assertRaisesRegex(CoverageError, r"value"):
            m.match(True)

    def test_product_arity_and_field_path(self):
        m = compile_matcher(
            {"P": {"kind": "product",
                    "fields": [("a", "Bool"), ("b", "Bool")]}},
            ["_"],
        )
        with self.assertRaisesRegex(CoverageError, r"value"):
            m.match((True,))
        with self.assertRaisesRegex(CoverageError, r"value\[1\]"):
            m.match((True, "yes"))
        with self.assertRaisesRegex(CoverageError, r"value"):
            m.match([True, False])  # list 不是 tuple

    def test_sum_unknown_label_and_arity(self):
        m = compile_matcher(SHAPE_TYPES, ["_"])
        with self.assertRaisesRegex(CoverageError, r"value"):
            m.match(("Nope", True))
        with self.assertRaisesRegex(CoverageError, r"需要 1 个字段值"):
            m.match(("Circle", True, False))
        with self.assertRaisesRegex(CoverageError, r"value\[1\]"):
            m.match(("Circle", "not-bool"))
        with self.assertRaisesRegex(CoverageError, r"value"):
            m.match(42)

    def test_nested_error_path(self):
        m = compile_matcher(NESTED_TYPES, ["_"])
        with self.assertRaisesRegex(CoverageError, r"value\[0\]\[1\]\[0\]"):
            m.match((("Y", ("nope", True)), False))


# ----------------------------------------------------------------- 覆盖分析视绑定为通配

class TestAnalyzeTreatsBindingsAsWild(unittest.TestCase):
    def test_var_behaves_like_wild(self):
        r = analyze(BOOL_TYPES, [("var", "x")])
        self.assertTrue(r.exhaustive)
        self.assertIs(r.branches[0].witness, False)

    def test_as_behaves_like_subpattern(self):
        r = analyze(BOOL_TYPES, [("as", "x", "True")])
        self.assertFalse(r.exhaustive)
        self.assertIs(r.uncovered, False)
        self.assertIs(r.branches[0].witness, True)

    def test_exhaustive_with_redundant_branch(self):
        # 覆盖完整却有冗余分支：or 兜底后，后面的具体分支永不匹配
        types = {"Opt": {"kind": "sum", "variants": [
            ("None", []), ("Some", [("v", "Bool")])]}}
        pats = [
            "None",
            ("Some", ("var", "v")),
            ("or", "None", ("Some", "True")),  # 已被前两条完全覆盖
        ]
        r = analyze(types, pats)
        self.assertTrue(r.exhaustive)
        self.assertIsNone(r.uncovered)
        self.assertEqual(r.redundant_branches, (2,))
        self.assertEqual([b.useful for b in r.branches], [True, True, False])

    def test_pure_pattern_results_unchanged(self):
        # 同一批随机绑定模式：analyze 结果必须与独立 strip 后的纯模式一致
        rng = random.Random(20260925)
        scenarios = [
            ({"Pair": {"kind": "product",
                        "fields": [("a", "Bool"), ("b", "Bool")]}}, "Pair", 60),
            ({"Opt": {"kind": "sum", "variants": [
                ("None", []), ("Some", [("v", "Bool")])]}}, "Opt", 60),
            (NESTED_TYPES, "Outer", 80),
        ]
        for types, root, trials in scenarios:
            ref = RefTypeTable(types)
            for _ in range(trials):
                pats = [_gen_pattern(rng, ref, root, 0, _Namer())
                        for _ in range(rng.randint(1, 4))]
                got = analyze(types, pats)
                exp = analyze(types, [strip_public(p) for p in pats])
                self.assertEqual(got.uncovered, exp.uncovered, f"\n模式={pats}")
                self.assertEqual(got.exhaustive, exp.exhaustive)
                self.assertEqual(
                    [(b.useful, b.witness) for b in got.branches],
                    [(b.useful, b.witness) for b in exp.branches],
                    f"\n模式={pats}",
                )


# ----------------------------------------------------------------- 与独立参考逐值核对

class TestMatchAgainstReference(unittest.TestCase):
    SCENARIOS = [
        ({"Day": {"kind": "enum", "constructors": ["Mon", "Tue", "Wed"]}}, "Day", 40),
        ({"Pair": {"kind": "product",
                    "fields": [("a", "Bool"), ("b", "Bool")]}}, "Pair", 80),
        ({"Opt": {"kind": "sum", "variants": [
            ("None", []), ("Some", [("v", "Bool")])]}}, "Opt", 80),
        (NESTED_TYPES, "Outer", 150),
        ({"Box": {"kind": "sum", "variants": [
            ("P", [("x", "Bool"), ("y", "E")]),
            ("Q", [("z", "E")]),
            ("R", [])]},
          "E": {"kind": "enum", "constructors": ["A", "B"]}}, "Box", 150),
    ]

    def test_randomized(self):
        rng = random.Random(914)
        for types, root, trials in self.SCENARIOS:
            ref = RefTypeTable(types)
            values = ref.all_values(root)
            for _ in range(trials):
                pats = [_gen_pattern(rng, ref, root, 0, _Namer())
                        for _ in range(rng.randint(1, 5))]
                matcher = compile_matcher(types, pats)
                for v in values:
                    got = matcher.match(v)
                    exp = first_hit(ref, root, pats, v)
                    msg = f"\n类型={types}\n模式={pats}\n值={v}"
                    if exp is None:
                        self.assertIsNone(got, msg)
                    else:
                        self.assertIsNotNone(got, msg)
                        self.assertEqual(got.branch, exp[0], msg)
                        self.assertEqual(got.bindings, exp[1], msg)
                # 同批模式的 analyze 与暴力参考（strip 后）一致
                got_a = analyze(types, pats)
                exp_a = brute_force_expected(
                    ref, root, [strip_public(p) for p in pats]
                )
                self.assertEqual(got_a.uncovered, exp_a["uncovered"])
                self.assertEqual(
                    [b.useful for b in got_a.branches],
                    [b["useful"] for b in exp_a["branches"]],
                )


# ----------------------------------------------------------------- 随机合法绑定模式

class _Namer:
    """单调新鲜绑定名：天然保证单路径不重名。"""

    def __init__(self):
        self.n = 0

    def __call__(self) -> str:
        self.n += 1
        return f"v{self.n}"


def _gen_pattern(rng, ref: RefTypeTable, tname: str, depth: int, namer,
                 allow_bind: bool = True):
    """生成保证合法的随机绑定模式（or 各备选绑定集合按构造一致）。"""
    t = ref.types[tname]
    kind = t["kind"]
    roll = rng.random()
    if allow_bind and roll < 0.12:
        return ("var", namer())
    if allow_bind and roll < 0.22:
        return ("as", namer(), _gen_pattern(rng, ref, tname, depth + 1, namer))
    if depth >= 3 or roll < 0.4:
        return "_"
    # or：备选各自独立生成（不含绑定），再以同一 as 名包装 → 绑定必然一致
    if rng.random() < 0.25:
        k = rng.randint(1, 3)
        alts = [
            _gen_pattern(rng, ref, tname, depth + 1, namer, allow_bind=False)
            for _ in range(k)
        ]
        if allow_bind and rng.random() < 0.5:
            name = namer()
            alts = [("as", name, a) for a in alts]
        return ("or", *alts)
    if kind == "bool":
        return rng.choice(["True", "False"])
    if kind == "enum":
        return rng.choice(t["constructors"])
    if kind == "product":
        return [_gen_pattern(rng, ref, ft, depth + 1, namer, allow_bind)
                for _, ft in t["fields"]]
    # sum
    label, fields = rng.choice(t["variants"])
    if not fields:
        return label
    return (label, *[_gen_pattern(rng, ref, ft, depth + 1, namer, allow_bind)
                     for _, ft in fields])


if __name__ == "__main__":
    unittest.main()
