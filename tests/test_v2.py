"""v2 测试：递归类型、整数闭区间、or 绑定、守卫、覆盖证书。

组织：
- 小有限域（含小整数 range）与 tests/_reference_v2.py 的*独立暴力参考*
  逐值核对（参考不导入 patcov）；
- 递归类型用结构性质 + 深度参数化类型族 + 覆盖证书独立检查器核对；
- 边界：空递归类型、or 绑定冲突、相邻整数区间、负数排序、预算耗尽 unknown、
  证书篡改必被拒（检查器不信任求解器的布尔标志）。
"""

from __future__ import annotations

import random
import unittest

from patcov import (
    CertificateRejected,
    CoverageBudgetExhausted,
    CoverageError,
    analyze,
    verify_certificate,
)

from _reference_v2 import RefV2Table, brute_force_v2_indexed
from _v2_helpers import (
    all_false_list,
    all_false_list_pattern,
    exact_length_list,
    list_types,
)


# ============================================================ 手工核对

class TestIntegerBasic(unittest.TestCase):
    T = {"N": {"kind": "integer", "range": [0, 10]}}

    def test_single_literal(self):
        r = analyze(self.T, [5], version=2)
        self.assertEqual(r.status, "incomplete")
        self.assertEqual(r.uncovered, 0)
        self.assertTrue(r.branches[0].useful)
        self.assertEqual(r.branches[0].witness, 5)

    def test_witness_order_abs_then_sign(self):
        # 0,-1,1,-2,2,... 的新序：只覆盖 0 与 1，最小遗漏应为 -1
        types = {"Z": {"kind": "integer", "range": [-5, 5]}}
        r = analyze(types, [0, 1], version=2)
        self.assertEqual(r.uncovered, -1)
        # 覆盖 -1 后，最小遗漏为 1？不，1 已覆盖 -> 2? 先 -2：
        r2 = analyze(types, [0, 1, -1], version=2)
        self.assertEqual(r2.uncovered, -2)
        r3 = analyze(types, [0, 1, -1, -2, 2], version=2)
        self.assertEqual(r3.uncovered, -3)

    def test_adjacent_intervals_exhaustive(self):
        pats = [("range", 0, 5), ("range", 6, 10)]
        r = analyze(self.T, pats, version=2)
        self.assertEqual(r.status, "exhaustive")
        self.assertEqual([b.witness for b in r.branches], [0, 6])

    def test_adjacent_intervals_gap_one(self):
        # [0,4] 与 [6,10]：唯一整数缺口 5
        pats = [("range", 0, 4), ("range", 6, 10)]
        r = analyze(self.T, pats, version=2)
        self.assertEqual(r.status, "incomplete")
        self.assertEqual(r.uncovered, 5)

    def test_large_endpoint_no_enumeration(self):
        types = {"Big": {"kind": "integer", "range": [-10**9, 10**9]}}
        r = analyze(types, [("range", 0, 10**9)], version=2)
        self.assertEqual(r.uncovered, -1)
        r2 = analyze(types, [("bind", "x", "_"),
                             {"pattern": ("bind", "x", "_"),
                              "when": ["<=", "x", -1]}], version=2)
        # 第一分支绑定全集恒真 => 已穷尽，第二分支冗余
        self.assertEqual(r2.status, "exhaustive")
        self.assertTrue(r2.branches[1].redundant)


class TestGuards(unittest.TestCase):
    T = {"N": {"kind": "integer", "range": [-10, 10]}}

    def test_guard_comparison_split(self):
        pats = [
            {"pattern": ("bind", "x", "_"), "when": ["<=", "x", 0]},
            {"pattern": ("bind", "x", "_"), "when": [">", "x", 0]},
        ]
        r = analyze(self.T, pats, version=2)
        self.assertEqual(r.status, "exhaustive")
        self.assertEqual([b.witness for b in r.branches], [0, 1])

    def test_guard_boolean_combo_and_neq(self):
        # x != 0 且 x in [-2,2] => {-2,-1,1,2}；新序最小匹配 -1，遗漏 0
        pats = [
            {"pattern": ("bind", "x", "_"),
             "when": ["and", ["!=", "x", 0], ["and", [">=", "x", -2], ["<=", "x", 2]]]},
        ]
        r = analyze(self.T, pats, version=2)
        self.assertEqual(r.uncovered, 0)
        self.assertEqual(r.branches[0].witness, -1)

    def test_guard_false_branch_is_redundant(self):
        pats = [
            {"pattern": ("bind", "x", "_"), "when": "false"},
        ]
        r = analyze(self.T, pats, version=2)
        self.assertTrue(r.branches[0].redundant)
        self.assertEqual(r.uncovered, 0)

    def test_guard_unbound_var_rejected(self):
        with self.assertRaisesRegex(CoverageError, "未由本分支"):
            analyze(self.T, [{"pattern": "_", "when": ["<=", "y", 0]}], version=2)

    def test_guard_non_integer_bind_rejected(self):
        types = {"Bool": {"kind": "bool"}}
        # 守卫只能比较整数绑定；Bool 上的绑定不可用于守卫
        with self.assertRaises(CoverageError):
            analyze(types, [{"pattern": ("bind", "b", "_"),
                            "when": ["==", "b", 1]}], version=2)

    def test_guard_constant_must_be_int(self):
        with self.assertRaisesRegex(CoverageError, "int"):
            analyze(self.T, [{"pattern": ("bind", "x", "_"),
                             "when": ["<=", "x", True]}], version=2)

    def test_guard_variable_comparison_rejected(self):
        # 右侧必须是常数
        with self.assertRaises(CoverageError):
            analyze(self.T, [{"pattern": ("bind", "x", "_"),
                             "when": ["<=", "x", "y"]}], version=2)

    def test_guard_subset_is_redundant(self):
        types = {"N": {"kind": "integer", "range": [0, 10]}}
        pats = [
            {"pattern": ("bind", "x", "_"), "when": ["<=", "x", 5]},
            {"pattern": ("bind", "x", "_"), "when": ["<=", "x", 3]},
        ]
        r = analyze(types, pats, version=2)
        self.assertTrue(r.branches[0].useful)
        self.assertTrue(r.branches[1].redundant)
        self.assertEqual(r.uncovered, 6)

    def test_nested_constructor_bind_guard(self):
        types = {"S": {"kind": "sum", "variants": [
            ("C", [("n", "N")]), ("D", [])]},
            "N": {"kind": "integer", "range": [0, 10]}}
        pats = [{"pattern": ("C", ("bind", "x", "_")),
                 "when": ["<=", "x", 2]}]
        r = analyze(types, pats, version=2, certificate=True)
        # C 在声明序先于 D；本分支最小见证 (C,0)，整体最小遗漏 (C,3)
        self.assertEqual(r.branches[0].witness, ("C", 0))
        self.assertEqual(r.uncovered, ("C", 3))
        verify_certificate(types, pats, r)


class TestOrBinds(unittest.TestCase):
    T = {"N": {"kind": "integer", "range": [-10, 10]}}

    def test_bind_name_conflict_rejected(self):
        with self.assertRaisesRegex(CoverageError, "绑定名称不一致"):
            analyze(self.T, [("or", ("bind", "x", 0), ("bind", "y", 1))], version=2)

    def test_bind_type_conflict_rejected(self):
        # 同一 or 两边：C 字段是整数、D 字段是 Bool，绑定 x 类型冲突
        types = {"S": {"kind": "sum", "variants": [
            ("C", [("n", "N")]), ("D", [("b", "Bool")])]},
            "N": {"kind": "integer", "range": [0, 5]}}
        with self.assertRaisesRegex(CoverageError, "类型不一致"):
            analyze(types, [("or", ("C", ("bind", "x", "_")),
                            ("D", ("bind", "x", "_")))], version=2)

    def test_or_alternatives_invisible_to_each_other(self):
        # 与 v1 同语义：or 内备选互不判冗余；整体取并
        r = analyze(self.T, [("or", 1, 1, 2)], version=2)
        self.assertEqual(r.status, "incomplete")
        self.assertEqual(r.uncovered, 0)
        self.assertEqual(r.branches[0].witness, 1)

    def test_or_with_guard_consistent_binds(self):
        pats = [
            {"pattern": ("or", ("bind", "x", ("range", -10, 0)),
                         ("bind", "x", ("range", 1, 10))),
             "when": ["!=", "x", 5]},
        ]
        r = analyze(self.T, pats, version=2)
        # 覆盖除 5 外全部（按新序）；最小见证 -10? 新序最小被匹配值: 0
        # range1=[-10,0] 含 0 => 最小匹配 0；遗漏最小 = 5
        self.assertEqual(r.uncovered, 5)
        self.assertEqual(r.branches[0].witness, 0)


# ============================================================ 递归类型


class TestRecursiveList(unittest.TestCase):
    def test_nil_only_misses_deep(self):
        r = analyze(list_types(), ["Nil"], version=2)
        self.assertEqual(r.status, "incomplete")
        self.assertEqual(r.uncovered, ("Cons", False, "Nil"))

    def test_exact_lengths_smallest_uncovered_next_length(self):
        # 覆盖长度 0..k-1 => 最小遗漏 = 长度 k（全 False）
        for k in (1, 2, 4, 7):
            pats = ["Nil"] + [exact_length_list(n) for n in range(1, k)]
            r = analyze(list_types(), pats, version=2, certificate=True)
            self.assertEqual(r.uncovered, all_false_list(k),
                             msg=f"k={k}")
            # 长度小于 k 的值被覆盖：抽查长度 k-1 全 True 不是遗漏
            summ = verify_certificate(list_types(), pats, r)
            self.assertEqual(summ["conclusion"], "incomplete")

    def test_constructor_recursion_exhaustive(self):
        r = analyze(list_types(), ["Nil", ("Cons", "_", "_")], version=2)
        self.assertEqual(r.status, "exhaustive")

    def test_deep_singleton_then_wild(self):
        # 深层单值分支 + 通配：单值分支有用，通配兜底后穷尽
        deep_val = all_false_list(50)       # 见证值（bool）
        deep_pat = all_false_list_pattern(50)  # 模式（"False" 字符串）
        r = analyze(list_types(), [deep_pat, "_"], version=2)
        self.assertEqual(r.status, "exhaustive")
        self.assertTrue(r.branches[0].useful)
        self.assertEqual(r.branches[0].witness, deep_val)
        self.assertTrue(r.branches[1].useful)


class TestWitnessDepthTypeFamily(unittest.TestCase):
    """遗漏见证深度由*输入参数* k 控制的类型族；验证器不预设最大深度。

    对每个 k：分支恰好覆盖长度 0..k-1 的 Bool 列表（共 k 个分支），
    最小遗漏必为长度 k 的全 False 列表（构造器节点数 k+1）。
    全程无任何深度阈值；并对每个 k 独立核对覆盖证书。
    """

    def test_depth_parameterized_family(self):
        # 快速回归只取小 k；大 k（无深度阈值）在 test_v2_extended 中覆盖
        for k in (0, 1, 3, 6):
            if k == 0:
                pats = []
            else:
                pats = ["Nil"] + [exact_length_list(n) for n in range(1, k)]
            r = analyze(list_types(), pats, version=2, certificate=True)
            expected = all_false_list(k)
            self.assertEqual(r.status, "incomplete")
            self.assertEqual(r.uncovered, expected)
            summ = verify_certificate(list_types(), pats, r)
            self.assertEqual(summ["uncovered"], expected)
            self.assertGreater(summ["states_checked"], 0)


class TestBinaryTree(unittest.TestCase):
    TREE = {"Tree": {"kind": "sum", "variants": [
        ("Leaf", []),
        ("Node", [("l", "Tree"), ("v", "Bool"), ("r", "Tree")]),
    ]}}

    def test_partial(self):
        r = analyze(self.TREE, ["Leaf", ("Node", "Leaf", "_", "Leaf")], version=2)
        self.assertEqual(r.status, "incomplete")
        # 最小遗漏：Node(Leaf,False,Node(Leaf,False,Leaf))（结构最小）
        self.assertEqual(
            r.uncovered,
            ("Node", "Leaf", False, ("Node", "Leaf", False, "Leaf")),
        )

    def test_exhaustive(self):
        r = analyze(self.TREE, ["Leaf", ("Node", "_", "_", "_")], version=2)
        self.assertEqual(r.status, "exhaustive")

    def test_deeply_nested_witness(self):
        # 只覆盖 Leaf 与高度恰为 1 的 Node；最小遗漏高度为 2
        r = analyze(self.TREE, ["Leaf", ("Node", "Leaf", "_", "Leaf")], version=2)
        cert_pats = ["Leaf", ("Node", "Leaf", "_", "Leaf")]
        r2 = analyze(self.TREE, cert_pats, version=2, certificate=True)
        verify_certificate(self.TREE, cert_pats, r2)
        self.assertEqual(r.uncovered, r2.uncovered)


class TestEmptyRecursiveTypes(unittest.TestCase):
    def test_directly_empty(self):
        types = {"E": {"kind": "sum", "variants": [("C", [("e", "E")])]}}
        r = analyze(types, [("C", "_")], version=2, certificate=True)
        # 值域为空：任何分支都冗余，整体穷尽（真空）
        self.assertEqual(r.status, "exhaustive")
        self.assertTrue(all(b.redundant for b in r.branches))
        verify_certificate(types, [("C", "_")], r)

    def test_empty_no_branches_also_exhaustive(self):
        types = {"E": {"kind": "sum", "variants": [("C", [("e", "E")])]}}
        r = analyze(types, [], version=2, certificate=True)
        self.assertEqual(r.status, "exhaustive")
        verify_certificate(types, [], r)

    def test_mutual_empty(self):
        types = {
            "A": {"kind": "sum", "variants": [("K", [("b", "B")])]},
            "B": {"kind": "sum", "variants": [("L", [("a", "A")])]},
        }
        r = analyze(types, [("K", ("L", "_"))], version=2)
        self.assertEqual(r.status, "exhaustive")

    def test_constructor_with_empty_field_is_unconstructable_other_lives(self):
        types = {
            "A": {"kind": "sum", "variants": [
                ("Bad", [("e", "E")]), ("Z", [])]},
            "E": {"kind": "sum", "variants": [("C", [("e", "E")])]},
        }
        # Bad 不可构造；只覆盖 Z 即穷尽
        r = analyze(types, ["Z"], version=2)
        self.assertEqual(r.status, "exhaustive")
        # 尝试匹配 Bad 的分支冗余（其字段 E 无值可匹配）；Z 仍遗漏
        r2 = analyze(types, [("Bad", ("C", "_"))], version=2)
        self.assertTrue(r2.branches[0].redundant)
        self.assertEqual(r2.status, "incomplete")
        self.assertEqual(r2.uncovered, "Z")
        # Bad 与 Z 都覆盖后穷尽
        r3 = analyze(types, [("Bad", ("C", "_")), "Z"], version=2)
        self.assertEqual(r3.status, "exhaustive")
        self.assertTrue(r3.branches[0].redundant)

    def test_nonconstructive_listed(self):
        types = {
            "A": {"kind": "sum", "variants": [("Bad", [("e", "E")]), ("Z", [])]},
            "E": {"kind": "sum", "variants": [("C", [("e", "E")])]},
        }
        r = analyze(types, ["Z"], version=2, certificate=True)
        self.assertEqual(set(r.certificate["nonconstructive"]), {"E"})
        verify_certificate(types, ["Z"], r)


# ============================================================ 与暴力参考

class TestAgainstBruteForceV2(unittest.TestCase):
    """小有限域上与独立暴力参考逐值一致（含整数/绑定/守卫/or）。"""

    SCENARIOS = [
        ({"Day": {"kind": "enum", "constructors": ["Mon", "Tue", "Wed"]}}, "Day"),
        ({"Pair": {"kind": "product",
                   "fields": [("a", "Bool"), ("b", "Bool")]}}, "Pair"),
        ({"N3": {"kind": "integer", "range": [-2, 2]}}, "N3"),
        ({"N7": {"kind": "integer", "range": [0, 6]}}, "N7"),
        ({"Box": {"kind": "sum", "variants": [
            ("P", [("x", "N2")]), ("Q", []), ("R", [("a", "Bool"), ("y", "N2")])]},
          "N2": {"kind": "integer", "range": [-1, 1]}}, "Box"),
        ({"Mix": {"kind": "product", "fields": [
            ("e", "E"), ("n", "N3")]},
          "E": {"kind": "enum", "constructors": ["A", "B"]},
          "N3": {"kind": "integer", "range": [-2, 2]}}, "Mix"),
    ]

    def test_randomized(self):
        rng = random.Random(20260924)
        trials = 0
        for types, root in self.SCENARIOS:
            ref = RefV2Table(types)
            for _ in range(30):
                pats = [_random_pat(rng, ref, root, 0)
                        for _ in range(rng.randint(1, 4))]
                got = analyze(types, pats, version=2, certificate=False)
                exp = brute_force_v2_indexed(ref, root, pats)
                self._compare(got, exp, types, pats, root, ref)
                trials += 1
        self.assertGreater(trials, 100)

    def _compare(self, got, exp, types, pats, root, ref):
        msg = f"\n类型={types}\n模式={pats}"
        self.assertEqual(got.status,
                         "exhaustive" if exp["exhaustive"] else "incomplete", msg)
        self.assertEqual(got.uncovered, exp["uncovered"], msg)
        self.assertEqual(len(got.branches), len(exp["branches"]), msg)
        for b, eb in zip(got.branches, exp["branches"]):
            self.assertEqual(b.useful, eb["useful"], msg + f"\n分支 {b.index}")
            self.assertEqual(b.witness, eb["witness"], msg + f"\n分支 {b.index}")


def _random_pat(rng, ref: RefV2Table, tname, depth):
    t = ref.types[tname]
    kind = t["kind"]
    roll = rng.random()
    if depth >= 2 or roll < 0.3:
        return "_"
    if kind == "bool":
        return rng.choice(["True", "False"])
    if kind == "enum":
        return rng.choice([c[0] for c in t["ctors"]])
    if kind == "integer":
        lo, hi = t["range"]
        choice = rng.random()
        if choice < 0.4:
            return rng.randint(lo, hi)
        if choice < 0.7:
            a = rng.randint(lo, hi)
            b = rng.randint(a, hi)
            return ("range", a, b)
        if depth == 0:
            # 带守卫的绑定只在顶层分支合法
            return {"pattern": ("bind", "x", "_"),
                    "when": _random_guard(rng, lo, hi)}
        return rng.randint(lo, hi)
    if kind == "product":
        return [_random_pat(rng, ref, ft, depth + 1) for _fn, ft in t["fields"]]
    # sum
    lab, fields = rng.choice(t["ctors"])
    if not fields:
        return lab
    if rng.random() < 0.25:
        # or 包装（绑定一致：直接 or 两个同形构造器，绑定少见）
        a = (lab, *[_random_pat(rng, ref, ft, depth + 1) for _fn, ft in fields])
        b = (lab, *[_random_pat(rng, ref, ft, depth + 1) for _fn, ft in fields])
        return ("or", a, b)
    return (lab, *[_random_pat(rng, ref, ft, depth + 1) for _fn, ft in fields])


def _random_guard(rng, lo, hi):
    c = rng.randint(lo, hi)
    op = rng.choice(["<=", ">", "==", "!="])
    g = [op, "x", c]
    if rng.random() < 0.4:
        g = ["and", g, ["in", "x", [lo, rng.randint(lo, hi)]]]
    if rng.random() < 0.3:
        g = ["or", g, ["==", "x", rng.randint(lo, hi)]]
    if rng.random() < 0.2:
        g = ["not", g]
    return g


# ============================================================ 覆盖证书

class TestCertificate(unittest.TestCase):
    def test_finite_enum_standalone_control_with_certificate(self):
        # 有限枚举只作独立对照：v2 也能处理，且证书经独立检查器核对
        types = {"Day": {"kind": "enum",
                         "constructors": ["Mon", "Tue", "Wed"]}}
        pats = ["Mon", "Wed"]
        r = analyze(types, pats, version=2, certificate=True)
        self.assertEqual(r.status, "incomplete")
        self.assertEqual(r.uncovered, "Tue")
        summ = verify_certificate(types, pats, r)
        self.assertEqual(summ["uncovered"], "Tue")

    def _pairs(self):
        types = {
            "List": {"kind": "sum", "variants": [
                ("Nil", []), ("Cons", [("h", "Bool"), ("t", "List")])]},
        }
        pats = ["Nil", ("Cons", "True", "_")]
        return types, pats

    def test_cert_roundtrip(self):
        types, pats = self._pairs()
        r = analyze(types, pats, version=2, certificate=True)
        summ = verify_certificate(types, pats, r)
        self.assertEqual(summ["conclusion"], "incomplete")
        self.assertGreaterEqual(summ["states_checked"], 3)
        self.assertGreaterEqual(summ["edges_checked"], 3)

    def test_cert_submitted_by_default_can_opt_out(self):
        types, pats = self._pairs()
        # 题面要求：覆盖成功默认提交有限状态闭合关系
        r = analyze(types, pats, version=2)
        self.assertIsNotNone(r.certificate)
        self.assertEqual(r.certificate["format"], "patcov-coverage-cert/v2")
        # 显式关闭以节省内存
        r2 = analyze(types, pats, version=2, certificate=False)
        self.assertIsNone(r2.certificate)
        # 显式要求等价于默认
        r3 = analyze(types, pats, version=2, certificate=True)
        self.assertIsNotNone(r3.certificate)

    def test_cert_tamper_conclusion_rejected(self):
        types, pats = self._pairs()
        r = analyze(types, pats, version=2, certificate=True)
        bad = _replace(r, status="exhaustive", uncovered=None, exhaustive=True)
        with self.assertRaises(CertificateRejected):
            verify_certificate(types, pats, bad)

    def test_cert_tamper_witness_rejected(self):
        types, pats = self._pairs()
        r = analyze(types, pats, version=2, certificate=True)
        bad = _replace(r, uncovered="Nil")
        with self.assertRaises(CertificateRejected):
            verify_certificate(types, pats, bad)

    def test_cert_tamper_state_weight_rejected(self):
        types, pats = self._pairs()
        r = analyze(types, pats, version=2, certificate=True)
        cert = dict(r.certificate)
        answers = [dict(a) for a in cert["state_answers"]]
        for a in answers:
            if "weight" in a:
                a["weight"] = a["weight"] + 1000
        cert["state_answers"] = answers
        bad = _replace(r, certificate=cert)
        with self.assertRaises(CertificateRejected):
            verify_certificate(types, pats, bad)

    def test_cert_tamper_edges_rejected(self):
        types, pats = self._pairs()
        r = analyze(types, pats, version=2, certificate=True)
        cert = dict(r.certificate)
        answers = [dict(a) for a in cert["state_answers"]]
        # 删掉根的一条后继边 => 闭合关系/段划分对不上
        root = next(a for a in answers if a["state"] == cert["root_state"])
        if len(root["edges"]) > 1:
            root["edges"] = root["edges"][:1]
        cert["state_answers"] = answers
        bad = _replace(r, certificate=cert)
        with self.assertRaises(CertificateRejected):
            verify_certificate(types, pats, bad)

    def test_cert_guard_integer_case(self):
        types = {"N": {"kind": "integer", "range": [-10, 10]}}
        pats = [{"pattern": ("bind", "x", "_"), "when": ["<", "x", 3]}]
        r = analyze(types, pats, version=2, certificate=True)
        summ = verify_certificate(types, pats, r)
        self.assertEqual(summ["uncovered"], 3)

    def test_root_witness_key_matches_value_key_with_product(self):
        # product 是匿名唯一构造器，前序键不得插入构造器编号；
        # 证书根状态见证键必须逐元素等于 model.value_key(见证值)
        from patcov.model import build_type_table, table_by_id, value_key
        types = {"S": {"kind": "sum", "variants": [
            ("P", [("q", "Q")]), ("Z", [])]},
            "Q": {"kind": "product",
                  "fields": [("a", "Bool"), ("b", "Bool")]}}
        pats = [("P", ["True", "_"])]
        r = analyze(types, pats, version=2, certificate=True)
        table = build_type_table(types, version=2)
        by = table_by_id(table)
        root_ans = next(a for a in r.certificate["state_answers"]
                        if a["state"] == r.certificate["root_state"])
        self.assertEqual(root_ans["outcome"], "witness")
        self.assertEqual(tuple(root_ans["witness_key"]),
                         value_key(by, table.root, r.uncovered))
        verify_certificate(types, pats, r)


import dataclasses


def _replace(result, **kw):
    return dataclasses.replace(result, **kw)


# ============================================================ 预算 / unknown

class TestBudgetUnknown(unittest.TestCase):
    T = {"N": {"kind": "integer", "range": [0, 100]}}

    def test_budget_exhaustion_is_unknown_not_exhaustive(self):
        with self.assertRaises(CoverageBudgetExhausted):
            # v1 行为不变：预算耗尽抛异常
            analyze({"X": {"kind": "enum", "constructors": ["A", "B", "C"]}},
                    ["A"], budget=1)
        # v2：返回 unknown + pending，绝不宣称穷尽
        r = analyze(self.T, [5], version=2, budget=1)
        self.assertEqual(r.status, "unknown")
        self.assertIsNone(r.exhaustive)
        self.assertIsNotNone(r.pending)
        self.assertTrue(any(p["question"] in ("coverage", "usefulness")
                            for p in r.pending))

    def test_v1_still_rejects_v2_syntax(self):
        # 兼容性：默认 version=1 仍拒绝整数/递归/绑定/守卫与 certificate=True
        with self.assertRaises(CoverageError):
            analyze({"N": {"kind": "integer", "range": [0, 5]}}, [1])
        rec = {"L": {"kind": "sum", "variants": [
            ("Nil", []), ("Cons", [("h", "Bool"), ("t", "L")])]}}
        with self.assertRaisesRegex(CoverageError, "递归"):
            analyze(rec, ["Nil"])
        with self.assertRaisesRegex(CoverageError, "certificate"):
            analyze({"X": {"kind": "enum", "constructors": ["A"]}}, ["A"],
                    certificate=True)

    def test_unknown_retry_with_more_budget_resolves(self):
        # 同一输入小预算 unknown，足预算给确定结论（unknown 不是通用答案）
        ru = analyze(self.T, [("range", 0, 100)], version=2, budget=1)
        self.assertEqual(ru.status, "unknown")
        ok = analyze(self.T, [("range", 0, 100)], version=2)
        self.assertEqual(ok.status, "exhaustive")

    def test_unknown_result_has_no_checkable_certificate(self):
        ru = analyze(self.T, [5], version=2, budget=1)
        self.assertIsNone(ru.certificate)
        with self.assertRaises(CertificateRejected):
            verify_certificate(self.T, [5], ru)

    def test_same_input_deterministic(self):
        types = list_types()
        pats = ["Nil", ("Cons", "True", "_")]
        r1 = analyze(types, pats, version=2)
        r2 = analyze(types, pats, version=2)
        self.assertEqual(r1.status, r2.status)
        self.assertEqual(r1.uncovered, r2.uncovered)
        self.assertEqual([b.witness for b in r1.branches],
                         [b.witness for b in r2.branches])



if __name__ == "__main__":
    unittest.main()
