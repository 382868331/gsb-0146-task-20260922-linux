"""变量绑定 / as 绑定 / 编译执行入口的必要测试。

- 覆盖语义：绑定按通配处理——含绑定模式的 analyze 结果与“剥掉绑定”的纯模式
  完全一致，并继续与独立暴力参考（env=None 时绑定即通配）逐值核对。
- 执行语义：compile_patterns + run 在小类型整个值域上逐值与独立递归参照
  tests/_reference.ref_run 核对分支下标与每个绑定值。
- 规则：同路径绑定名不重复；or 各备选绑定名字与类型必须完全一致。
"""

from __future__ import annotations

import random
import unittest

from patcov import (
    CoverageError,
    Match,
    analyze,
    compile_patterns,
    run,
)

from _reference import RefTypeTable, brute_force_expected, ref_run


# ----------------------------------------------------------------- 手核小例

class TestHandCheckedBindings(unittest.TestCase):
    SHAPE = {
        "Shape": {"kind": "sum", "variants": [
            ("Circle", [("r", "Bool")]),
            ("Rect", [("w", "Bool"), ("h", "Bool")]),
            ("Dot", [])]},
    }

    def test_var_binds_field_and_subtree(self):
        types = {
            "P": {"kind": "product", "fields": [("a", "Bool"), ("b", "Shape")]},
            "Shape": self.SHAPE["Shape"],
        }
        cp = compile_patterns(types, [
            ["True", ("as", ("Circle", ("var", "rad")), "c")],
            ["_", "Dot"],
        ])
        m = run(cp, (True, ("Circle", False)))
        self.assertEqual(m, Match(0, {"rad": False, "c": ("Circle", False)}))
        # as 捕获的是完整子树（整个带标签变体），不止字段
        m2 = run(cp, (True, ("Circle", True)))
        self.assertEqual(m2.bindings["c"], ("Circle", True))
        self.assertIs(m2.bindings["rad"], True)
        # 第二分支
        m3 = run(cp, (False, "Dot"))
        self.assertEqual(m3, Match(1, {}))
        # 无命中显式 None
        self.assertIsNone(run(cp, (False, ("Rect", True, True))))

    def test_nested_product_and_sum(self):
        types = {
            "Outer": {"kind": "product", "fields": [("i", "Inner"), ("k", "Bool")]},
            "Inner": {"kind": "sum", "variants": [
                ("X", []), ("Y", [("b", "Bool")])]},
        }
        pats = [
            ["X", ("var", "k")],
            [("as", ("Y", ("var", "b")), "y"), "_"],
        ]
        cp = compile_patterns(types, pats)
        self.assertEqual(run(cp, ("X", True)), Match(0, {"k": True}))
        self.assertEqual(run(cp, (("Y", False), False)),
                         Match(1, {"b": False, "y": ("Y", False)}))
        # 绑定类型表
        r = analyze(types, pats)
        self.assertEqual(r.branches[0].bindings, {"k": "Bool"})
        self.assertEqual(r.branches[1].bindings, {"b": "Bool", "y": "Inner"})

    def test_or_alternatives_bind_same_names(self):
        # 同名绑定出现在两个 or 备选：类型相同 -> 合法；命中备选决定绑定值
        cp = compile_patterns(self.SHAPE, [
            ("or", ("Circle", ("var", "x")), ("Rect", ("var", "x"), "_")),
            "Dot",
        ])
        self.assertEqual(run(cp, ("Circle", True)), Match(0, {"x": True}))
        self.assertEqual(run(cp, ("Rect", False, True)), Match(0, {"x": False}))
        self.assertEqual(run(cp, "Dot"), Match(1, {}))
        r = analyze(self.SHAPE, [
            ("or", ("Circle", ("var", "x")), ("Rect", ("var", "x"), "_")),
            "Dot",
        ])
        self.assertTrue(r.exhaustive)  # Circle/全部 + Rect 字段通配 + Dot = 全集
        self.assertEqual(r.branches[0].bindings, {"x": "Bool"})

    def test_or_bindings_independent_slots(self):
        # 第一备选失败时其半成品绑定不得泄漏到第二备选
        types = {"E": {"kind": "enum", "constructors": ["A", "B"]}}
        cp = compile_patterns(types, [
            ("or", ("as", "A", "z"), ("as", "B", "z")),
        ])
        m = run(cp, "B")
        self.assertEqual(m.bindings, {"z": "B"})

    def test_exhaustive_but_redundant_branch(self):
        # 覆盖完整却含冗余分支：前两支已穷举，第三支永不命中
        types = {"Color": {"kind": "enum", "constructors": ["R", "G", "B"]}}
        pats = [("as", "R", "x"), "_", "G"]
        r = analyze(types, pats)
        self.assertTrue(r.exhaustive)
        self.assertIsNone(r.uncovered)
        self.assertTrue(r.branches[0].useful)
        self.assertTrue(r.branches[1].useful)
        self.assertTrue(r.branches[2].redundant)
        cp = compile_patterns(types, pats)
        self.assertEqual(run(cp, "R"), Match(0, {"x": "R"}))
        self.assertEqual(run(cp, "G"), Match(1, {}))   # 落到通配，而非冗余的分支 2
        self.assertEqual(run(cp, "B"), Match(1, {}))
        self.assertIsNotNone(run(cp, "G"))  # 有命中


# ----------------------------------------------------------------- 规则拒绝

class TestBindingRules(unittest.TestCase):
    SHAPE = {
        "Shape": {"kind": "sum", "variants": [
            ("Circle", [("r", "Bool")]),
            ("Rect", [("w", "Bool"), ("h", "Bool")]),
            ("Dot", [])]},
    }
    PAIR = {"P": {"kind": "product", "fields": [("a", "Bool"), ("b", "Bool")]}}

    def _reject(self, types, pats, needle=""):
        with self.assertRaisesRegex(CoverageError, needle):
            compile_patterns(types, pats)

    def test_duplicate_name_same_path_var_var(self):
        self._reject(self.PAIR, [[("var", "x"), ("var", "x")]], "同一条路径上重复")

    def test_duplicate_name_as_and_inner(self):
        self._reject(self.SHAPE,
                     [("as", ("Circle", ("var", "x")), "x")], "同一条路径上重复")

    def test_duplicate_after_or_in_sibling(self):
        # or 在每个路径上都绑定 x，后续兄弟位置再绑 x 必拒
        self._reject(self.PAIR,
                     [[("or", ("var", "x"), ("var", "x")), ("var", "x")]],
                     "同一条路径上重复")

    def test_same_name_in_two_or_branches_missing_in_one(self):
        self._reject(self.SHAPE,
                     [("or", ("Circle", ("var", "x")), "Dot")],
                     "or 的各备选必须绑定完全相同")

    def test_same_name_in_two_or_branches_extra_in_one(self):
        self._reject(self.SHAPE,
                     [("or", ("Circle", ("var", "x")),
                       ("Rect", ("var", "x"), ("var", "y")))],
                     "or 的各备选必须绑定完全相同")

    def test_or_name_type_mismatch(self):
        # x 在备选 1 是 Bool，备选 2 是整个 Shape
        self._reject(self.SHAPE,
                     [("or", ("Circle", ("var", "x")), ("var", "x"))],
                     "类型不一致")

    def test_bind_name_underscore_rejected(self):
        self._reject(self.PAIR, [[("var", "_"), "_"]], "绑定名")

    def test_bind_name_non_string(self):
        self._reject(self.PAIR, [[("var", 7), "_"]], "绑定名")

    def test_var_arity(self):
        self._reject(self.PAIR, [[("var",), "_"]])
        self._reject(self.PAIR, [[("var", "x", "y"), "_"]])

    def test_as_arity(self):
        self._reject(self.PAIR, [[("as", "_"), "_"]])

    def test_reserved_constructor_name(self):
        for bad in ("or", "var", "as"):
            with self.assertRaises(CoverageError):
                compile_patterns(
                    {"E": {"kind": "enum", "constructors": [bad]}}, ["_"])


# ----------------------------------------------------------------- 值校验

class TestValueValidation(unittest.TestCase):
    def setUp(self):
        self.types = {
            "P": {"kind": "product", "fields": [("a", "Bool"), ("b", "Shape")]},
            "Shape": {"kind": "sum", "variants": [
                ("Circle", [("r", "Bool")]), ("Dot", [])]},
        }
        self.cp = compile_patterns(self.types, [["_", "_"]])

    def test_wrong_scalar_type(self):
        with self.assertRaisesRegex(CoverageError, r"value\[0\].*Bool"):
            run(self.cp, (1, "Dot"))
        with self.assertRaisesRegex(CoverageError, r"value.*bool"):
            run(compile_patterns({"Bool": {"kind": "bool"}}, ["_"]), 0)

    def test_wrong_outer_shape(self):
        with self.assertRaisesRegex(CoverageError, r"value.*tuple"):
            run(self.cp, ["True", "Dot"])  # list 不是 tuple

    def test_bad_arity(self):
        with self.assertRaisesRegex(CoverageError, r"2 个字段"):
            run(self.cp, (True,))

    def test_unknown_label(self):
        with self.assertRaisesRegex(CoverageError, r"Square"):
            run(self.cp, (True, ("Square", True)))

    def test_nonvariant_as_tuple(self):
        with self.assertRaisesRegex(CoverageError, r"零元"):
            run(self.cp, (True, ("Dot",)))

    def test_label_with_missing_fields(self):
        with self.assertRaisesRegex(CoverageError, r"1 个字段值"):
            run(self.cp, (True, ("Circle",)))

    def test_nested_bad_field_type(self):
        with self.assertRaisesRegex(CoverageError, r"value\[1\]\[1\]"):
            run(self.cp, (True, ("Circle", "yes")))

    def test_compiled_argument_type(self):
        with self.assertRaisesRegex(CoverageError, r"compile_patterns"):
            run([["_", "_"]], (True, "Dot"))


# ----------------------------------------------------------------- 覆盖等价

def strip_bindings(p):
    """独立地把绑定语法剥成纯模式：var -> '_'，as -> 内层。"""
    if isinstance(p, list):
        return [strip_bindings(x) for x in p]
    if isinstance(p, tuple):
        tag = p[0]
        if tag == "var":
            return "_"
        if tag == "as":
            return strip_bindings(p[1])
        return tuple(strip_bindings(x) for x in p)
    return p


# ----------------------------------------------------------------- 随机互校

SCENARIOS = [
    ({"Day": {"kind": "enum", "constructors": ["Mon", "Tue", "Wed"]}}, "Day"),
    ({"Pair": {"kind": "product",
               "fields": [("a", "Bool"), ("b", "Bool")]}}, "Pair"),
    ({"Opt": {"kind": "sum", "variants": [
        ("None", []), ("Some", [("v", "Bool")])]}}, "Opt"),
    ({"Outer": {"kind": "product", "fields": [("i", "Inner"), ("k", "Bool")]},
      "Inner": {"kind": "sum", "variants": [("X", []), ("Y", [("b", "Bool")])]}},
     "Outer"),
    ({"Box": {"kind": "sum", "variants": [
        ("P", [("x", "Bool"), ("y", "E")]),
        ("Q", [("z", "E")]),
        ("R", [])]},
      "E": {"kind": "enum", "constructors": ["A", "B"]}}, "Box"),
]


def _alloc():
    n = [0]

    def fresh():
        n[0] += 1
        return f"v{n[0] - 1}"
    return fresh


def _gen_pat(rng, ref, tname, depth, alloc):
    """生成带绑定的合法模式，返回 (raw_pattern, binds: {名字: 类型名})。

    绑定名在*单条分支*内全局唯一（或备选通过扰动共享同一组名字），
    因此生成结果一定通过“同路径不重名 / or 备选一致”校验。
    """
    t = ref.types[tname]
    kind = t["kind"]

    def leaf_literal():
        if kind == "bool":
            return rng.choice(["True", "False"]), {}
        if kind == "enum":
            return rng.choice(t["constructors"]), {}
        if kind == "product":
            return ["_" for _ in t["fields"]], {}
        zero = [lab for lab, fs in t["variants"] if not fs]
        if zero:
            return rng.choice(zero), {}
        return "_", {}

    if depth >= 3:
        return leaf_literal()
    roll = rng.random()

    if roll < 0.20:
        return "_", {}
    if roll < 0.38:
        return leaf_literal()
    if roll < 0.55 and kind == "product":
        subs = [_gen_pat(rng, ref, ft, depth + 1, alloc) for _, ft in t["fields"]]
        return [p for p, _ in subs], _merge(subs)
    if roll < 0.72 and kind == "sum":
        label, fields = rng.choice(t["variants"])
        subs = [_gen_pat(rng, ref, ft, depth + 1, alloc) for _, ft in fields]
        if not fields:
            return label, {}
        return (label, *[p for p, _ in subs]), _merge(subs)
    if roll < 0.84:
        name = alloc()
        return ("var", name), {name: tname}
    if roll < 0.94:
        # as：内层同类型再生成一次（不增加 depth，概率自然衰减，仍受节点上限保护）
        name = alloc()
        inner, ib = _gen_pat(rng, ref, tname, depth + 1, alloc)
        binds = dict(ib)
        if name in binds:  # 理论上不会（全局唯一），防御性处理
            return inner, ib
        binds[name] = tname
        return ("as", inner, name), binds
    # or：先生成备选 1，其余备选只扰动字面量/通配，绑定节点原样保留
    alt1, b1 = _gen_pat(rng, ref, tname, depth, alloc)
    alts = [alt1]
    for _ in range(rng.randint(1, 2)):
        alts.append(_perturb(rng, ref, tname, alt1))
    if rng.random() < 0.3:
        alts.append(alt1)  # 重复备选
    return ("or", *alts), dict(b1)


def _merge(subs):
    out = {}
    for _, b in subs:
        out.update(b)
    return out


def _perturb(rng, ref, tname, pat):
    """保持绑定节点位置/名字不变，只扰动字面量与通配，保证 or 备选绑定一致。"""
    t = ref.types[tname]
    kind = t["kind"]
    if isinstance(pat, list):
        return [_perturb(rng, ref, ft, x)
                for (_, ft), x in zip(t["fields"], pat)]
    if isinstance(pat, str):
        if pat == "_":
            if kind == "bool":
                return rng.choice(["_", "True", "False"])
            if kind == "enum":
                return rng.choice(["_", *t["constructors"]])
            if kind == "product":
                return ["_" for _ in t["fields"]]
            zero = ["_"] + [lab for lab, fs in t["variants"] if not fs]
            return rng.choice(zero)
        # 字面量：可能变通配
        if rng.random() < 0.5:
            return "_"
        if kind == "bool":
            return "False" if pat == "True" else "True"
        if kind == "enum":
            return rng.choice([c for c in t["constructors"] if c != pat] or [pat])
        return pat
    tag = pat[0]
    if tag == "var":
        return pat
    if tag == "as":
        return ("as", _perturb(rng, ref, tname, pat[1]), pat[2])
    if tag == "or":
        return ("or", *[_perturb(rng, ref, tname, a) for a in pat[1:]])
    # 构造器
    fields = dict(t["variants"])[tag]
    return (tag, *[_perturb(rng, ref, ft, x)
                   for (_, ft), x in zip(fields, pat[1:])])


class TestRandomizedBindings(unittest.TestCase):
    TRIALS = 150

    def test_randomized(self):
        rng = random.Random(20260925)
        for types, root in SCENARIOS:
            ref = RefTypeTable(types)
            values = ref.all_values(root)
            for _ in range(self.TRIALS):
                alloc = _alloc()
                pats = []
                binds_per_branch = []
                for _ in range(rng.randint(1, 4)):
                    p, b = _gen_pat(rng, ref, root, 0, alloc)
                    pats.append(p)
                    binds_per_branch.append(b)

                # 1) 编译成功（生成器保证绑定合法）
                cp = compile_patterns(types, pats)

                # 2) 覆盖：与剥掉绑定的纯模式完全一致
                got = analyze(types, pats)
                stripped = [strip_bindings(p) for p in pats]
                plain = analyze(types, stripped)
                msg = f"\n类型={types}\n模式={pats}\n剥后={stripped}"
                self.assertEqual(
                    [(b.useful, b.witness) for b in got.branches],
                    [(b.useful, b.witness) for b in plain.branches], msg)
                self.assertEqual(got.uncovered, plain.uncovered, msg)
                self.assertEqual(got.exhaustive, plain.exhaustive, msg)

                # 3) 覆盖：与独立暴力参考一致（参考 env=None 时绑定即通配）
                exp = brute_force_expected(ref, root, pats)
                self.assertEqual(got.uncovered, exp["uncovered"], msg)
                for b, eb in zip(got.branches, exp["branches"]):
                    self.assertEqual(b.useful, eb["useful"], msg)
                    self.assertEqual(b.witness, eb["witness"], msg)
                    self.assertEqual(b.bindings, binds_per_branch[b.index], msg)

                # 4) 执行：值域内每个值逐值核对独立递归参照
                for v in values:
                    m = run(cp, v)
                    em = ref_run(ref, root, pats, v)
                    if em is None:
                        self.assertIsNone(m, msg + f"\n值={v!r} 不应命中")
                    else:
                        ei, eenv = em
                        self.assertIsNotNone(m, msg + f"\n值={v!r} 应命中")
                        self.assertEqual(m.branch, ei, msg + f"\n值={v!r}")
                        self.assertEqual(m.bindings, eenv, msg + f"\n值={v!r}")
                        # 绑定键集合恰为该分支声明的绑定，且每个值类型位置正确
                        self.assertEqual(set(m.bindings), set(binds_per_branch[ei]),
                                         msg + f"\n值={v!r}")
                        for name, tv in binds_per_branch[ei].items():
                            # 独立复核：该绑定值确实位于声明类型位置
                            self.assertIn(name, m.bindings)


if __name__ == "__main__":
    unittest.main()
