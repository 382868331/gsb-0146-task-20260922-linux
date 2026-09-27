"""独立暴力参考实现 —— 不导入、不调用 patcov 的任何核心模块。

用途仅为在*小*类型上为测试提供期望值：
- 自行解析类型声明（独立代码路径）；
- 显式枚举整个有限值域；
- 独立实现模式匹配、字典序与逐分支有用性。

被测核心（patcov.solver / patcov.analyze）从未在此被导入。
"""

from __future__ import annotations


class RefTypeTable:
    def __init__(self, types: dict):
        # 独立校验最小子集（测试输入保证合法）
        self.raw = types
        self.types: dict[str, dict] = {}
        for name, spec in types.items():
            if name == "Bool":
                continue
            kind = spec["kind"]
            if kind == "enum":
                self.types[name] = {"kind": "enum",
                                    "constructors": list(spec["constructors"])}
            elif kind == "product":
                self.types[name] = {"kind": "product",
                                    "fields": [tuple(f) for f in spec["fields"]]}
            else:
                self.types[name] = {"kind": "sum",
                                    "variants": [(lab, [tuple(f) for f in fs])
                                                  for lab, fs in spec["variants"]]}
        self.types["Bool"] = {"kind": "bool"}

    # ----------------------------------------------------------- 值枚举
    def all_values(self, tname: str):
        out = []
        self._gen(tname, out)
        return sorted(out, key=lambda v: self.key(tname, v))

    def _gen(self, tname: str, out: list):
        t = self.types[tname]
        kind = t["kind"]
        if kind == "bool":
            out.extend([False, True])
            return
        if kind == "enum":
            out.extend(t["constructors"])
            return
        if kind == "product":
            per_field = []
            for _, ft in t["fields"]:
                vals = []
                self._gen(ft, vals)
                per_field.append(vals)
            for combo in _cartesian(per_field):
                out.append(tuple(combo))
            return
        # sum
        for label, fields in t["variants"]:
            if not fields:
                out.append(label)
                continue
            per_field = []
            for _, ft in fields:
                vals = []
                self._gen(ft, vals)
                per_field.append(vals)
            for combo in _cartesian(per_field):
                out.append((label, *combo))

    # ----------------------------------------------------------- 字典序
    def key(self, tname: str, value):
        t = self.types[tname]
        kind = t["kind"]
        if kind == "bool":
            return (1 if value else 0,)
        if kind == "product":
            k = ()
            for (_, ft), v in zip(t["fields"], value):
                k += self.key(ft, v)
            return k
        if isinstance(value, tuple):
            label, fvals = value[0], value[1:]
        else:
            label, fvals = value, ()
        labels = (t["constructors"] if kind == "enum"
                  else [lab for lab, _ in t["variants"]])
        idx = labels.index(label)
        k = (idx,)
        if kind == "sum":
            fields = dict(t["variants"])[label]
            for (_, ft), v in zip(fields, fvals):
                k += self.key(ft, v)
        return k

    # ----------------------------------------------------------- 匹配
    def matches(self, tname: str, pat, value) -> bool:
        return _match(self.types, tname, pat, value)


def _cartesian(lists):
    """小列表笛卡尔积（仅参考枚举用）。"""
    if not lists:
        yield ()
        return
    for head in lists[0]:
        for tail in _cartesian(lists[1:]):
            yield (head, *tail)


def _match(types, tname, pat, value) -> bool:
    if pat == "_":
        return True
    t = types[tname]
    kind = t["kind"]

    if isinstance(pat, list):
        # product
        if kind != "product" or not isinstance(value, tuple):
            return False
        if len(pat) != len(value):
            return False
        return all(_match(types, ft, p, v)
                   for (_, ft), p, v in zip(t["fields"], pat, value))

    if isinstance(pat, str):
        if kind == "bool":
            return value is (pat == "True")
        # enum 或零元 sum 变体
        return value == pat

    # tuple
    tag = pat[0]
    if tag == "or":
        return any(_match(types, tname, alt, value) for alt in pat[1:])
    # 构造器模式
    if kind != "sum" or not isinstance(value, tuple):
        return False
    if value[0] != tag:
        return False
    fields = dict(t["variants"])[tag]
    args = pat[1:]
    if len(args) != len(fields):
        return False
    return all(_match(types, ft, p, v)
               for (_, ft), p, v in zip(fields, args, value[1:]))


def matches(ref: RefTypeTable, tname: str, pat, value) -> bool:
    return ref.matches(tname, pat, value)


def brute_force_expected(ref: RefTypeTable, root: str, patterns) -> dict:
    """逐分支暴力计算期望值（独立逐值计数）。

    返回 {total_count, uncovered_count, uncovered, branches:[{useful,witness,count}]}。
    """
    universe = set(ref.all_values(root))
    covered: set = set()
    branches = []
    for i, pat in enumerate(patterns):
        matched = {v for v in universe if ref.matches(root, pat, v)}
        fresh = matched - covered
        if fresh:
            witness = min(fresh, key=lambda v: ref.key(root, v))
            branches.append({"useful": True, "witness": witness, "count": len(fresh)})
        else:
            branches.append({"useful": False, "witness": None, "count": 0})
        covered |= matched
    leftover = universe - covered
    uncovered = min(leftover, key=lambda v: ref.key(root, v)) if leftover else None
    return {
        "uncovered": uncovered,
        "branches": branches,
        "total_count": len(universe),
        "uncovered_count": len(leftover),
        "branch_counts": [b["count"] for b in branches],
    }
