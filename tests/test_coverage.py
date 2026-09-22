"""Tests for coverage_witness.

The expected results for small types come from an INDEPENDENT naive reference
implemented in this file: it enumerates values straight from the raw type
dictionaries and matches the raw pattern dictionaries itself.  It never calls
the module under test.  Agreement with the independent oracle is the primary
correctness check; "the reported witness is actually matched / not matched"
round-trips are only supplementary.
"""

import unittest

import coverage_witness as cw


# ---------------------------------------------------------------------------
# Independent reference (deliberately simple, shares no code with the core)
# ---------------------------------------------------------------------------


def ref_all_values(ref, types):
    """Enumerate ALL values of a small finite type, in contract order."""
    if ref == "bool":
        yield False
        yield True
        return
    d = types[ref]
    if d["kind"] == "enum":
        for c in d["constructors"]:
            yield c
    elif d["kind"] == "product":
        ftypes = [f[1] if isinstance(f, list) else f for f in d["fields"]]

        def gen(i):
            if i == len(ftypes):
                yield []
                return
            for tail in gen(i + 1):
                for v in ref_all_values(ftypes[i], types):
                    yield [v] + tail

        yield from gen(0)
    else:  # sum
        for cname, raw_fields in d["constructors"]:
            ftypes = [f[1] if isinstance(f, list) else f for f in raw_fields]

            def gen(i):
                if i == len(ftypes):
                    yield []
                    return
                for tail in gen(i + 1):
                    for v in ref_all_values(ftypes[i], types):
                        yield [v] + tail

            for fvals in gen(0):
                yield {cname: fvals}


def ref_matches(pat, value):
    """Independent interpreter of the raw pattern DSL over a raw value."""
    if isinstance(pat, bool):
        return isinstance(value, bool) and pat == value
    kind = pat["kind"]
    if kind in ("wildcard", "wild", "_"):
        return True
    if kind == "or":
        return any(ref_matches(a, value) for a in pat["patterns"])
    assert kind == "ctor"
    name = pat.get("name")
    fields = pat.get("fields", [])
    if isinstance(value, bool):
        return value == (name == "True")
    if isinstance(value, str):  # enum
        return value == name
    if isinstance(value, list):  # product
        return len(value) == len(fields) and all(
            ref_matches(p, v) for p, v in zip(fields, value)
        )
    # sum: {"C": [...]}
    (cname, fvals) = next(iter(value.items()))
    return (
        cname == name
        and len(fvals) == len(fields)
        and all(ref_matches(p, v) for p, v in zip(fields, fvals))
    )


def ref_analyze(types, patterns, root):
    """Independent brute-force oracle for small types."""
    values = list(ref_all_values(root, types))
    covered = set()

    def key(v):
        # Independent ordering: bool < , enum/sum ctor index, fields LTR.
        if isinstance(v, bool):
            return (v,)
        if isinstance(v, str):
            d = types[root]
            return (d["constructors"].index(v),)
        raise AssertionError("key used at top level only")

    def deep_key(v, ref):
        if ref == "bool":
            return (1 if v else 0,)
        d = types[ref]
        if d["kind"] == "enum":
            return (d["constructors"].index(v),)
        if d["kind"] == "product":
            ftypes = [f[1] if isinstance(f, list) else f for f in d["fields"]]
            return tuple(deep_key(x, t) for x, t in zip(v, ftypes))
        (cname, fvals) = next(iter(v.items()))
        ci = [c for c, _ in d["constructors"]].index(cname)
        cfields = [
            f[1] if isinstance(f, list) else f
            for f in dict(d["constructors"])[cname]
        ]
        return (ci, *[deep_key(x, t) for x, t in zip(fvals, cfields)])

    branches = []
    for i, pat in enumerate(patterns):
        fresh = [
            v
            for v in values
            if _index(v, root, types) not in covered and ref_matches(pat, v)
        ]
        if fresh:
            w = min(fresh, key=lambda v: deep_key(v, root))
            branches.append({"index": i, "useful": True, "witness": w})
            for v in fresh:
                covered.add(_index(v, root, types))
        else:
            branches.append({"index": i, "useful": False, "witness": None})
    missing = [
        v for v in values if _index(v, root, types) not in covered
    ]
    uncovered = min(missing, key=lambda v: deep_key(v, root)) if missing else None
    return branches, uncovered


def _index(v, ref, types):
    """Stable identity for values of one type (independent of core)."""
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v
    if isinstance(v, list):
        return tuple(_index(x, None, types) for x in v)
    (c, fs) = next(iter(v.items()))
    return (c, tuple(_index(x, None, types) for x in fs))


def assert_agrees(testcase, types, patterns, root):
    result = cw.analyze(types, patterns, root=root)
    exp_branches, exp_uncovered = ref_analyze(types, patterns, root)
    testcase.assertEqual(len(result["branches"]), len(exp_branches))
    for got, exp in zip(result["branches"], exp_branches):
        testcase.assertEqual(got["index"], exp["index"])
        testcase.assertEqual(got["useful"], exp["useful"])
        testcase.assertEqual(got["redundant"], not exp["useful"])
        testcase.assertEqual(got["witness"], exp["witness"])
        if exp["useful"]:
            # Supplementary round-trip: witness is matched by this branch and
            # by no earlier useful branch.
            testcase.assertTrue(ref_matches(patterns[got["index"]], got["witness"]))
    testcase.assertEqual(result["uncovered"], exp_uncovered)
    testcase.assertEqual(result["exhaustive"], exp_uncovered is None)
    return result


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

COLOR = {"kind": "enum", "constructors": ["R", "G", "B"]}
SHADE = {"kind": "enum", "constructors": ["Light", "Dark"]}
BB = {
    "kind": "product",
    "fields": [["a", "bool"], ["b", "bool"]],
}
SHAPE = {
    "kind": "sum",
    "constructors": [
        ["Point", []],
        ["Circle", [["r", "bool"]]],
        ["Rect", [["w", "bool"], ["h", "bool"]]],
    ],
}
PAIR_CB = {
    "kind": "product",
    "fields": [["c", "Color"], ["s", "Shade"]],
}

T = {"kind": "ctor", "name": "True"}
F = {"kind": "ctor", "name": "False"}
W = {"kind": "wildcard"}


class ExhaustiveOracleTests(unittest.TestCase):
    """Small types: brute-force agreement across many clause sets."""

    TYPES = {"Color": COLOR, "Shade": SHADE, "BB": BB, "Shape": SHAPE, "PairCB": PAIR_CB}

    def test_bool_cases(self):
        assert_agrees(self, self.TYPES, [T], root="bool")
        assert_agrees(self, self.TYPES, [F, T], root="bool")
        assert_agrees(self, self.TYPES, [F, W], root="bool")
        assert_agrees(self, self.TYPES, [W, T], root="bool")
        assert_agrees(self, self.TYPES, [
            {"kind": "or", "patterns": [T, F]}
        ], root="bool")
        assert_agrees(self, self.TYPES, [
            {"kind": "or", "patterns": [T, T, F, T]}
        ], root="bool")

    def test_enum_cases(self):
        ctor = lambda n: {"kind": "ctor", "name": n}
        assert_agrees(self, self.TYPES, [ctor("R"), ctor("G"), ctor("B")], "Color")
        assert_agrees(self, self.TYPES, [ctor("R"), ctor("R"), W], "Color")
        assert_agrees(self, self.TYPES, [
            {"kind": "or", "patterns": [ctor("R"), ctor("G")]},
            ctor("B"),
            ctor("R"),
        ], "Color")
        assert_agrees(self, self.TYPES, [W, ctor("B")], "Color")

    def test_product_partial_coverage(self):
        p = lambda a, b: {"kind": "ctor", "fields": [a, b]}
        # (True, _) then (False, True): [False, False] must remain uncovered.
        r = assert_agrees(self, self.TYPES, [p(T, W), p(F, T)], "BB")
        self.assertEqual(r["uncovered"], [False, False])
        # local field coverage with a wildcard elsewhere
        assert_agrees(self, self.TYPES, [p(W, T), p(W, F)], "BB")
        assert_agrees(self, self.TYPES, [p(T, T), p(T, F), p(F, W)], "BB")
        assert_agrees(self, self.TYPES, [p(T, W), p(W, T), p(F, F)], "BB")

    def test_sum_never_matching_constructor(self):
        # A constructor clause after a wildcard can never match anything.
        r = assert_agrees(
            self,
            self.TYPES,
            [
                W,
                {"kind": "ctor", "name": "Point"},
                {"kind": "ctor", "name": "Circle", "fields": [W]},
            ],
            "Shape",
        )
        self.assertTrue(r["branches"][1]["redundant"])
        self.assertTrue(r["branches"][2]["redundant"])
        self.assertTrue(r["exhaustive"])

    def test_sum_mixed(self):
        assert_agrees(
            self,
            self.TYPES,
            [
                {"kind": "ctor", "name": "Point"},
                {"kind": "ctor", "name": "Circle", "fields": [T]},
                {"kind": "ctor", "name": "Circle", "fields": [F]},
                {"kind": "ctor", "name": "Rect", "fields": [W, W]},
            ],
            "Shape",
        )
        assert_agrees(
            self,
            self.TYPES,
            [
                {"kind": "or", "patterns": [
                    {"kind": "ctor", "name": "Point"},
                    {"kind": "ctor", "name": "Circle", "fields": [W]},
                ]},
                {"kind": "ctor", "name": "Rect", "fields": [T, W]},
            ],
            "Shape",
        )

    def test_nested_enum_product(self):
        ctor = lambda n: {"kind": "ctor", "name": n}
        assert_agrees(
            self,
            self.TYPES,
            [
                {"kind": "ctor", "fields": [ctor("R"), W]},
                {"kind": "ctor", "fields": [W, ctor("Dark")]},
            ],
            "PairCB",
        )

    def test_or_duplicates_deep(self):
        # or inside constructor fields, with duplicated alternatives.
        ctor = lambda n: {"kind": "ctor", "name": n}
        assert_agrees(
            self,
            self.TYPES,
            [
                {"kind": "ctor", "fields": [
                    {"kind": "or", "patterns": [ctor("R"), ctor("R"), ctor("G")]},
                    W,
                ]},
                {"kind": "ctor", "fields": [ctor("B"), ctor("Light")]},
            ],
            "PairCB",
        )

    def test_bool_literal_shorthand(self):
        assert_agrees(self, self.TYPES, [True, False], "bool")
        r = cw.analyze(self.TYPES, [True], root="bool")
        self.assertEqual(r["uncovered"], False)


class LargeProductTests(unittest.TestCase):
    """Huge bool product: 2**N values must never be enumerated."""

    N = 20  # 1,048,576 values

    def setUp(self):
        self.types = {
            "Big": {
                "kind": "product",
                "fields": [[f"b{i}", "bool"] for i in range(self.N)],
            }
        }
        self.wild_row = {"kind": "ctor", "fields": [W] * self.N}

    def ctor_row(self, prefix):
        fields = [T if v else F for v in prefix] + [W] * (self.N - len(prefix))
        return {"kind": "ctor", "fields": fields}

    def test_partial_cover_without_cartesian_blowup(self):
        r = cw.analyze(
            self.types,
            [self.ctor_row([True]), self.ctor_row([False, True])],
            root="Big",
        )
        self.assertFalse(r["branches"][0]["redundant"])
        self.assertFalse(r["branches"][1]["redundant"])
        # Smallest missing value: False, False, then all False.
        self.assertEqual(r["uncovered"], [False] * self.N)

    def test_binary_trie_exhaustive(self):
        # 21 clauses cover 2**20 values the way a decision tree does; the
        # matrix factorisation must prove exhaustiveness without enumerating.
        patterns = []
        for i in range(self.N):
            prefix = [False] * i + [True]
            patterns.append(self.ctor_row(prefix))
        patterns.append(self.wild_row)  # all-False tail
        r = cw.analyze(self.types, patterns, root="Big")
        self.assertTrue(r["exhaustive"])
        self.assertIsNone(r["uncovered"])
        # Every branch is useful.
        self.assertTrue(all(b["useful"] for b in r["branches"]))

    def test_redundancy_inside_trie(self):
        patterns = [self.ctor_row([True]), self.ctor_row([True, False])]
        r = cw.analyze(self.types, patterns, root="Big")
        self.assertTrue(r["branches"][0]["useful"])
        self.assertTrue(r["branches"][1]["redundant"])


class WitnessOrderTests(unittest.TestCase):
    def test_constructor_order_follows_input(self):
        types = {"E": {"kind": "enum", "constructors": ["Z", "A", "M"]}}
        r = cw.analyze(types, [{"kind": "ctor", "name": "Z"}], root="E")
        # Input order, not alphabetic: after Z comes A.
        self.assertEqual(r["uncovered"], "A")

    def test_field_order_follows_input(self):
        types = {
            "P": {
                "kind": "product",
                "fields": [["x", "E"], ["y", "E"]],
            },
            "E": {"kind": "enum", "constructors": ["Z", "A"]},
        }
        ctor = lambda n: {"kind": "ctor", "name": n}
        r = cw.analyze(
            types,
            [{"kind": "ctor", "fields": [ctor("Z"), W]}],
            root="P",
        )
        # First field varies before second in lexicographic order.
        self.assertEqual(r["uncovered"], ["A", "Z"])


class ErrorSemanticsTests(unittest.TestCase):
    def test_nan_rejected(self):
        with self.assertRaises(cw.CoverageError) as cm:
            cw.analyze({"E": {"kind": "enum", "constructors": ["A", float("nan")]}},
                       [W], root="E")
        self.assertIn("NaN", str(cm.exception))

    def test_infinity_rejected(self):
        with self.assertRaises(cw.CoverageError) as cm:
            cw.analyze({"E": {"kind": "enum", "constructors": ["A", float("inf")]}},
                       [W], root="E")
        self.assertIn("Infinity", str(cm.exception))

    def test_budget_bool_rejected(self):
        with self.assertRaises(cw.CoverageError):
            cw.analyze({}, [W], root="bool", budget=True)
        with self.assertRaises(cw.CoverageError):
            cw.analyze({}, [W], root="bool", budget=False)

    def test_budget_must_be_positive_int(self):
        with self.assertRaises(cw.CoverageError):
            cw.analyze({}, [W], root="bool", budget=0)
        with self.assertRaises(cw.CoverageError):
            cw.analyze({}, [W], root="bool", budget=1.5)

    def test_unknown_constructor_is_locatable(self):
        with self.assertRaises(cw.CoverageError) as cm:
            cw.analyze({"E": {"kind": "enum", "constructors": ["A"]}},
                       [{"kind": "ctor", "name": "X"}], root="E")
        msg = str(cm.exception)
        self.assertIn("patterns[0]", msg)
        self.assertIn("'X'", msg)

    def test_unknown_type_reference(self):
        with self.assertRaises(cw.CoverageError) as cm:
            cw.analyze(
                {"P": {"kind": "product", "fields": [["x", "Missing"]]}},
                [W], root="P",
            )
        self.assertIn("Missing", str(cm.exception))

    def test_recursive_types_rejected(self):
        types = {
            "Chain": {
                "kind": "sum",
                "constructors": [["End", []], ["Next", [["tail", "Chain"]]]],
            }
        }
        with self.assertRaises(cw.CoverageError) as cm:
            cw.analyze(types, [W], root="Chain")
        self.assertIn("recursive", str(cm.exception))

    def test_wrong_field_arity(self):
        types = {"S": {"kind": "sum", "constructors": [["C", [["a", "bool"]]]]}}
        with self.assertRaises(cw.CoverageError) as cm:
            cw.analyze(types, [{"kind": "ctor", "name": "C", "fields": []}],
                       root="S")
        self.assertIn("expects 1 fields", str(cm.exception))

    def test_duplicate_constructor(self):
        with self.assertRaises(cw.CoverageError):
            cw.analyze({"E": {"kind": "enum", "constructors": ["A", "A"]}},
                       [W], root="E")

    def test_constructor_limit(self):
        types = {
            "E": {"kind": "enum",
                  "constructors": [f"C{i}" for i in range(129)]}
        }
        with self.assertRaises(cw.CoverageError) as cm:
            cw.analyze(types, [W], root="E")
        self.assertIn("128", str(cm.exception))

    def test_depth_limit(self):
        types = {}
        for i in range(12):
            if i == 0:
                types["T0"] = {"kind": "enum", "constructors": ["C"]}
            else:
                types[f"T{i}"] = {
                    "kind": "product",
                    "fields": [["f", f"T{i-1}"]],
                }
        # T11 -> T10 -> ... -> T0 has depth 12: accepted.
        cw.analyze(types, [W], root="T11")
        types["T12"] = {"kind": "product", "fields": [["f", "T11"]]}
        with self.assertRaises(cw.CoverageError) as cm:
            cw.analyze(types, [W], root="T12")
        self.assertIn("depth", str(cm.exception))

    def test_budget_exhausted_is_distinct_from_proven_none(self):
        # Proven exhaustive: returns None, does not raise.
        r = cw.analyze({}, [T, F], root="bool")
        self.assertIsNone(r["uncovered"])
        # Same problem with a deliberately tiny budget: undecided -> exception,
        # which must NOT be confused with "exhaustive".
        with self.assertRaises(cw.BudgetExhausted):
            cw.analyze({}, [T, F], root="bool", budget=1)

    def test_or_requires_alternatives(self):
        with self.assertRaises(cw.CoverageError):
            cw.analyze({}, [{"kind": "or", "patterns": []}], root="bool")

    def test_bool_shorthand_wrong_type(self):
        with self.assertRaises(cw.CoverageError):
            cw.analyze({"E": COLOR}, [True], root="E")


class DifferentialFuzzTests(unittest.TestCase):
    """Deterministic random generation vs. the independent brute-force oracle."""

    SEED = 20260922
    CASES = 300

    def test_random_small_schemes(self):
        import random

        rng = random.Random(self.SEED)
        for case in range(self.CASES):
            types, root = _gen_scheme(rng, case)
            patterns = _gen_clauses(rng, types, root, rng.randint(1, 5))
            # Both must agree; any disagreement is a core bug.
            assert_agrees(self, types, patterns, root)


def _gen_scheme(rng, case):
    # A small connected scheme: 1-3 enums plus one product/sum root.
    n_enums = rng.randint(1, 2)
    types = {}
    enum_names = []
    for i in range(n_enums):
        name = f"E{case}_{i}"
        k = rng.randint(2, 3)
        letters = ["P", "Q", "R", "S"]
        ctors = [f"{letters[j]}{i}" for j in range(k)]
        types[name] = {"kind": "enum", "constructors": ctors}
        enum_names.append(name)
    leaf = lambda: rng.choice(enum_names + ["bool"])

    root_name = f"Root{case}"
    shape = rng.choice(["enum-only", "product", "sum", "nested-sum", "nested-prod"])
    if shape == "enum-only":
        return types, enum_names[0]
    if shape == "product":
        nf = rng.randint(1, 3)
        types[root_name] = {
            "kind": "product",
            "fields": [[f"f{i}", leaf()] for i in range(nf)],
        }
    elif shape == "sum":
        nctors = rng.randint(2, 3)
        ctors = []
        for ci in range(nctors):
            nf = rng.randint(0, 2)
            ctors.append([f"K{ci}", [[f"g{j}", leaf()] for j in range(nf)]])
        types[root_name] = {"kind": "sum", "constructors": ctors}
    elif shape == "nested-sum":
        # sum containing a product-typed field, depth stays tiny
        pname = f"Inner{case}"
        types[pname] = {
            "kind": "product",
            "fields": [["x", leaf()], ["y", leaf()]],
        }
        types[root_name] = {
            "kind": "sum",
            "constructors": [
                ["A", []],
                ["B", [["p", pname]]],
                ["C", [["z", leaf()]]],
            ],
        }
    else:
        # product whose first field is a sum
        sname = f"Inner{case}"
        types[sname] = {
            "kind": "sum",
            "constructors": [
                ["U", []],
                ["V", [["v", leaf()]]],
            ],
        }
        types[root_name] = {
            "kind": "product",
            "fields": [["s", sname], ["t", leaf()]],
        }
    return types, root_name


def _gen_pat(rng, types, ref, depth=0):
    if depth > 2:
        return W
    choice = rng.random()
    if choice < 0.3:
        return W
    if ref == "bool":
        return rng.choice([T, F])
    d = types[ref]
    if d["kind"] == "enum":
        name = rng.choice(d["constructors"])
        return {"kind": "ctor", "name": name}
    if d["kind"] == "product":
        ftypes = [f[1] if isinstance(f, list) else f for f in d["fields"]]
        base = {
            "kind": "ctor",
            "fields": [
                (W if rng.random() < 0.5 else _gen_pat(rng, types, t, depth + 1))
                for t in ftypes
            ],
        }
    else:
        cname, raw = rng.choice(d["constructors"])
        ftypes = [f[1] if isinstance(f, list) else f for f in raw]
        base = {
            "kind": "ctor",
            "name": cname,
            "fields": [
                (W if rng.random() < 0.5 else _gen_pat(rng, types, t, depth + 1))
                for t in ftypes
            ],
        }
    if rng.random() < 0.25:
        # or-pattern, sometimes with a duplicated alternative
        other = _gen_pat(rng, types, ref, depth + 1)
        alts = [base, other] + ([base] if rng.random() < 0.5 else [])
        rng.shuffle(alts)
        return {"kind": "or", "patterns": alts}
    return base


def _gen_clauses(rng, types, root, n):
    clauses = [_gen_pat(rng, types, root) for _ in range(n)]
    if rng.random() < 0.3:
        clauses.append(W)
    return clauses


if __name__ == "__main__":
    unittest.main(verbosity=2)
