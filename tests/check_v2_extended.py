"""v2 **扩展验证**（较慢，与快速回归分开运行）。

运行：
    python -m unittest tests.test_v2_extended -v

覆盖：
- 小有限域上的大批量随机核对（约 720 例，独立暴力参考，不调用被测核心
  计算期望值）；
- 遗漏见证深度由输入参数 k 控制的类型族（k 到 50，验证器不预设最大深度）；
- 非退化主规模：8 种互递归类型、80 个分支、整数端点 ±10^9，工作量不随
  值域大小增长；
- 深层递归列表/二叉树 + 覆盖证书的独立闭合/最小性核对。
"""

from __future__ import annotations

import unittest

from patcov import analyze, verify_certificate

from _reference_v2 import RefV2Table, brute_force_v2_indexed
from _v2_helpers import (
    TREE_TYPES,
    all_false_list,
    exact_length_list,
    list_types,
)
from test_v2 import TestAgainstBruteForceV2, _random_pat


class TestLargeRandomized(unittest.TestCase):
    """小有限域大批量随机：每例与独立暴力参考逐值一致。"""

    def test_many_randomized(self):
        rng = random.Random(9242026)
        trials = 0
        for types, root in TestAgainstBruteForceV2.SCENARIOS:
            ref = RefV2Table(types)
            for _ in range(120):
                pats = [_random_pat(rng, ref, root, 0)
                        for _ in range(rng.randint(1, 4))]
                got = analyze(types, pats, version=2, certificate=False)
                exp = brute_force_v2_indexed(ref, root, pats)
                msg = f"\n类型={types}\n模式={pats}"
                self.assertEqual(
                    got.status,
                    "exhaustive" if exp["exhaustive"] else "incomplete", msg)
                self.assertEqual(got.uncovered, exp["uncovered"], msg)
                for b, eb in zip(got.branches, exp["branches"]):
                    self.assertEqual(b.useful, eb["useful"], msg)
                    self.assertEqual(b.witness, eb["witness"], msg)
                trials += 1
        self.assertGreater(trials, 700)


class TestDeepWitnessFamily(unittest.TestCase):
    """见证深度随输入参数 k 增大；任何深度都给确定答案且证书可独立核对。"""

    def test_k_to_fifty(self):
        for k in (8, 12, 20, 35, 50):
            pats = ["Nil"] + [exact_length_list(n) for n in range(1, k)]
            r = analyze(list_types(), pats, version=2, certificate=True)
            expected = all_false_list(k)
            self.assertEqual(r.status, "incomplete")
            self.assertEqual(r.uncovered, expected)
            summ = verify_certificate(list_types(), pats, r)
            self.assertEqual(summ["uncovered"], expected)
            # 状态数应随 k 线性（有限状态，而非按深度指数/阈值截断）
            self.assertLess(summ["states_checked"], 6 * k + 10)


class TestScale(unittest.TestCase):
    """非退化主规模：≥8 种递归类型、80 分支、端点 ±10^9。"""

    def _build(self):
        # 根类型 R0 首项；8 种互引递归类型 + 1 个大整数类型
        types = {
            "R0": {"kind": "sum", "variants": [
                ("Stop0", []), ("Go0", [("n", "Big"), ("r", "R1")])]},
            "Big": {"kind": "integer", "range": [-10**9, 10**9]},
        }
        for i in range(1, 8):
            nxt = f"R{(i + 1) % 8}"
            types[f"R{i}"] = {"kind": "sum", "variants": [
                (f"Stop{i}", []), (f"Go{i}", [("n", "Big"), ("r", nxt)])]}
        return types

    def test_eighty_branches_large_endpoints(self):
        types = self._build()
        # 40 个相邻整数区间（覆盖 [-10^9, 10^9-1]，留下单点 10^9）
        pats = []
        lo = -10**9
        step = (2 * 10**9) // 40
        for k in range(40):
            a = lo + k * step
            b = -10**9 + (k + 1) * step - 1
            pats.append(("Go0", ("range", a, b), "_"))
        pats.append("Stop0")

        def pat_for(ridx, d):
            if d == 0:
                return f"Stop{ridx}"
            return (f"Go{ridx}", "_", pat_for((ridx + 1) % 8, d - 1))

        for d in range(39):
            pats.append(("Go0", "_", pat_for(1, d)))
        self.assertEqual(len(pats), 80)
        r = analyze(types, pats, version=2, certificate=True)
        self.assertIn(r.status, ("exhaustive", "incomplete"))
        summ = verify_certificate(types, pats, r)
        self.assertEqual(summ["conclusion"], r.status)
        # 工作量只与区间数/分支数相关，不随 10^9 值域增长
        self.assertLess(r.budget_used, 5_000_000)

    def test_eight_recursive_types_exhaustive_pattern(self):
        """8 种互递归类型上的标准穷尽：每个 Stop + 每个 Go 通配。"""
        types = self._build()
        # 对根 R0：Stop0 与 Go0(_,_) 即穷尽（其字段 Big/R1 由通配覆盖）
        r = analyze(types, ["Stop0", ("Go0", "_", "_")],
                    version=2, certificate=True)
        self.assertEqual(r.status, "exhaustive")
        verify_certificate(types, ["Stop0", ("Go0", "_", "_")], r)

    def test_large_endpoint_gap_witness(self):
        types = {"Big": {"kind": "integer", "range": [-10**9, 10**9]}}
        # 只覆盖单点 10^9：最小遗漏 0（新序），绝不枚举 2·10^9 个值
        r = analyze(types, [10**9], version=2, certificate=True)
        self.assertEqual(r.uncovered, 0)
        self.assertLess(r.budget_used, 100)
        verify_certificate(types, [10**9], r)


class TestDeepTreeCertificate(unittest.TestCase):
    def test_partial_height_gap(self):
        # 覆盖 Leaf 与恰高度 1 的 Node；最小遗漏高度为 2，证书独立核对
        pats = ["Leaf", ("Node", "Leaf", "_", "Leaf")]
        r = analyze(TREE_TYPES, pats, version=2, certificate=True)
        self.assertEqual(
            r.uncovered,
            ("Node", "Leaf", False, ("Node", "Leaf", False, "Leaf")),
        )
        verify_certificate(TREE_TYPES, pats, r)


import random  # noqa: E402  (置于末以保持文件结构清晰)


if __name__ == "__main__":
    unittest.main()
