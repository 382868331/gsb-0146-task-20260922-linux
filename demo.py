"""Fixed-input demonstration for coverage_witness.

Run with ``python demo.py``.  Everything here is computed live (no sleeps, no
pre-recorded answers): one normal coverage analysis on a small labeled-sum
type, a large bool-product analysis that never enumerates its 2**N values,
and two genuinely triggered error boundaries (an illegal input and a search
budget running out -- the latter distinct from a proven exhaustive match).
"""

import time

import coverage_witness as cw

W = {"kind": "wildcard"}


def heading(title):
    print("=" * 68)
    print(title)
    print("=" * 68)


def show(result):
    for b in result["branches"]:
        state = "redundant (covers nothing new)" if b["redundant"] else "useful"
        print(f"  branch {b['index']}: {state}")
        if b["useful"]:
            print(f"      smallest witness newly covered: {b['witness']}")
    print(f"  exhaustive : {result['exhaustive']}")
    print(f"  uncovered  : {result['uncovered']}  "
          f"(lexicographically smallest missed value)")


def main():
    # ------------------------------------------------------------------
    # 1. Normal result on a finite labeled sum with an enum payload.
    # ------------------------------------------------------------------
    heading("1. Normal analysis  (type Box = Empty | Full(Answer))")
    types = {
        "Answer": {"kind": "enum", "constructors": ["Yes", "No", "Maybe"]},
        "Box": {
            "kind": "sum",
            "constructors": [
                ["Empty", []],
                ["Full", [["a", "Answer"]]],
            ],
        },
    }
    ctor = lambda n: {"kind": "ctor", "name": n}
    patterns = [
        {"kind": "ctor", "name": "Full", "fields": [ctor("Yes")]},      # 0
        {"kind": "ctor", "name": "Full", "fields": [ctor("No")]},       # 1
        {"kind": "or", "patterns": [ctor("Empty"), ctor("Empty")]},     # 2 dup or
        {"kind": "ctor", "name": "Full", "fields": [ctor("Yes")]},      # 3 never
    ]
    t0 = time.perf_counter()
    result = cw.analyze(types, patterns, root="Box")
    print(f"  (computed in {time.perf_counter() - t0:.4f}s)")
    show(result)
    assert result["branches"][2]["useful"], "Empty must still be useful"
    assert result["branches"][3]["redundant"], "duplicate Full(Yes) is redundant"
    assert result["uncovered"] == {"Full": ["Maybe"]}

    # ------------------------------------------------------------------
    # 2. Huge bool product: 2**40 values, factorised, never enumerated.
    # ------------------------------------------------------------------
    heading("2. Large bool product  (Big = bool x ... x bool, N = 40 => 2**40)")
    N = 40
    big_types = {
        "Big": {"kind": "product",
                "fields": [[f"b{i}", "bool"] for i in range(N)]}
    }
    T, F = {"kind": "ctor", "name": "True"}, {"kind": "ctor", "name": "False"}

    def row(prefix):
        return {"kind": "ctor",
                "fields": [T if v else F for v in prefix] + [W] * (N - len(prefix))}

    trie = [row([False] * i + [True]) for i in range(N)]
    trie.append({"kind": "ctor", "fields": [F] * N})  # all-False leaf
    t0 = time.perf_counter()
    big = cw.analyze(big_types, trie, root="Big")
    dt = time.perf_counter() - t0
    print(f"  {N + 1} clauses proved exhaustive over 2**{N} = "
          f"{2 ** N} values in {dt:.4f}s (no Cartesian expansion)")
    print(f"  exhaustive: {big['exhaustive']}, all branches useful: "
          f"{all(b['useful'] for b in big['branches'])}")
    assert big["exhaustive"] and all(b["useful"] for b in big["branches"])

    # ------------------------------------------------------------------
    # 3a. Triggered rejection boundary: non-finite input (NaN), located.
    # ------------------------------------------------------------------
    heading("3a. Rejected illegal input  (NaN constructor name)")
    bad_types = {"E": {"kind": "enum",
                       "constructors": ["Ok", float("nan")]}}
    try:
        cw.analyze(bad_types, [W], root="E")
    except cw.CoverageError as exc:  # really raised, caught and reported
        print(f"  CoverageError caught (input rejected, nothing corrected):")
        print(f"    {exc}")
    else:
        raise AssertionError("NaN input must be rejected")

    # ------------------------------------------------------------------
    # 3b. Triggered search boundary: budget exhausted != proven exhaustive.
    # ------------------------------------------------------------------
    heading("3b. Search budget exhausted  (distinct from a proven exhaustive match)")
    t0 = time.perf_counter()
    try:
        cw.analyze(types, patterns, root="Box", budget=1)
    except cw.BudgetExhausted as exc:
        print(f"  BudgetExhausted caught after {time.perf_counter() - t0:.4f}s:")
        print(f"    {exc}")
    else:
        raise AssertionError("budget=1 must be exhausted on this clause set")
    proven = cw.analyze(types, patterns, root="Box")
    print(f"  With the default budget the SAME clauses finish and "
          f"exhaustive={proven['exhaustive']}:")
    print("  'undecided' (exception) is never reported as 'exhaustive'.")

    print("=" * 68)
    print("demo finished.")


if __name__ == "__main__":
    main()
