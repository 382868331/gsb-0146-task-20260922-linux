"""v2 独立暴力参考 —— 不导入、不调用 patcov 的任何模块。

仅在*小而有限*（非递归，整数类型 range 很小）的类型上为测试提供期望值：

- 自行解析类型声明（独立代码路径，只认测试生成的合法子集）；
- 显式枚举整个有限值域；
- 独立实现模式匹配（含整数字面量/区间、绑定、or）、守卫求值、
  逐分支有用性与题面见证次序（构造器节点数优先，再前序：构造器声明序、
  整数新序 0,-1,1,-2,2,...）。

递归类型不在此枚举（值域无限）；递归情形由结构性质、深度参数化类型族与
覆盖证书独立检查器覆盖。
"""

from __future__ import annotations


# ------------------------------------------------------------------ 类型表

class RefV2Table:
    def __init__(self, types: dict):
        self.types: dict[str, dict] = {}
        for name, spec in types.items():
            if name == "Bool":
                continue
            kind = spec["kind"]
            if kind == "enum":
                self.types[name] = {"kind": "enum",
                                    "ctors": [(c, ()) for c in spec["constructors"]]}
            elif kind == "product":
                self.types[name] = {"kind": "product",
                                    "fields": [tuple(f) for f in spec["fields"]]}
            elif kind == "integer":
                self.types[name] = {"kind": "integer",
                                    "range": tuple(spec["range"])}
            else:
                self.types[name] = {
                    "kind": "sum",
                    "ctors": [(lab, tuple((fn, ft) for fn, ft in fs))
                              for lab, fs in spec["variants"]],
                }
        self.types["Bool"] = {"kind": "bool", "ctors": [("False", ()), ("True", ())]}

    # ---------------------------------------------------------------- 枚举
    def all_values(self, tname: str):
        out = []
        self._gen(tname, out, set())
        return out

    def _gen(self, tname, out, stack):
        if tname in stack:
            raise ValueError("参考枚举不支持递归类型")
        t = self.types[tname]
        kind = t["kind"]
        if kind == "bool":
            out.extend([False, True])
        elif kind == "integer":
            lo, hi = t["range"]
            out.extend(range(lo, hi + 1))
        elif kind == "product":
            per = []
            for _fn, ft in t["fields"]:
                col = []
                self._gen(ft, col, stack)
                per.append(col)
            for combo in _cartesian(per):
                out.append(tuple(combo))
        else:
            for lab, fields in t["ctors"]:
                if not fields:
                    out.append(lab)
                    continue
                per = []
                for _fn, ft in fields:
                    col = []
                    self._gen(ft, col, stack)
                    per.append(col)
                for combo in _cartesian(per):
                    out.append((lab, *combo))

    # ---------------------------------------------------------------- 权/键
    def weight(self, tname, v) -> int:
        t = self.types[tname]
        kind = t["kind"]
        if kind in ("bool",):
            return 1
        if kind == "integer":
            return 0
        if kind == "product":
            return sum(self.weight(ft, v[j])
                       for j, (_fn, ft) in enumerate(t["fields"]))
        label = v[0] if isinstance(v, tuple) else v
        fields = dict(t["ctors"])[label]
        w = 1
        fvals = () if isinstance(v, str) else v[1:]
        for (_fn, ft), fv in zip(fields, fvals):
            w += self.weight(ft, fv)
        return w

    def prekey(self, tname, v):
        """前序键：构造器声明序编号 + 字段；整数 (abs, 负先)。"""
        t = self.types[tname]
        kind = t["kind"]
        if kind == "bool":
            return (0 if v is False else 1,)
        if kind == "integer":
            return (abs(v), 0 if v < 0 else 1)
        if kind == "product":
            k = ()
            for (_fn, ft), fv in zip(t["fields"], v):
                k += self.prekey(ft, fv)
            return k
        label = v[0] if isinstance(v, tuple) else v
        labels = [c[0] for c in t["ctors"]]
        idx = labels.index(label)
        k = (idx,)
        fields = dict(t["ctors"])[label]
        fvals = () if isinstance(v, str) else v[1:]
        for (_fn, ft), fv in zip(fields, fvals):
            k += self.prekey(ft, fv)
        return k

    def sort_key(self, tname, v):
        return (self.weight(tname, v), self.prekey(tname, v))


def _cartesian(lists):
    if not lists:
        yield ()
        return
    for head in lists[0]:
        for tail in _cartesian(lists[1:]):
            yield (head, *tail)


# ------------------------------------------------------------------ 守卫

def _eval_guard(g, env, branch):
    if g == "true":
        return True
    if g == "false":
        return False
    op = g[0]
    if op == "and":
        return all(_eval_guard(x, env, branch) for x in g[1:])
    if op == "or":
        return any(_eval_guard(x, env, branch) for x in g[1:])
    if op == "not":
        return not _eval_guard(g[1], env, branch)
    name = g[1]
    z = env[name]
    if op == "in":
        return g[2][0] <= z <= g[2][1]
    c = g[2]
    return {"<": z < c, "<=": z <= c, ">": z > c, ">=": z >= c,
            "==": z == c, "!=": z != c}[op]


# ------------------------------------------------------------------ 匹配

def match(ref: RefV2Table, tname, pat, value, env):
    t = ref.types[tname]
    kind = t["kind"]
    if pat == "_":
        return True
    if isinstance(pat, dict):
        spec, guard = pat["pattern"], pat.get("when", "true")
        save = dict(env)
        if _match_node(ref, tname, spec, value, env) and \
                _eval_guard(guard, env, None):
            return True
        env.clear()
        env.update(save)
        return False
    return _match_node(ref, tname, pat, value, env)


def _match_node(ref, tname, pat, value, env):
    t = ref.types[tname]
    kind = t["kind"]
    if pat == "_":
        return True
    if isinstance(pat, int) and not isinstance(pat, bool):
        return kind == "integer" and value == pat
    if isinstance(pat, list):
        if kind != "product" or not isinstance(value, tuple):
            return False
        return len(pat) == len(value) and all(
            _match_node(ref, ft, pat[j], value[j], env)
            for j, (_fn, ft) in enumerate(t["fields"]))
    if isinstance(pat, str):
        if kind == "bool":
            return value is (pat == "True")
        return value == pat
    tag = pat[0]
    if tag == "or":
        # 各备选互不可见；逐条在保存的环境上尝试，成功才保留绑定
        for alt in pat[1:]:
            save = dict(env)
            if _match_node(ref, tname, alt, value, env):
                return True
            env.clear()
            env.update(save)
        return False
    if tag == "bind":
        name, inner = pat[1], pat[2]
        if _match_node(ref, tname, inner, value, env):
            env[name] = value
            return True
        return False
    if tag == "range":
        return kind == "integer" and pat[1] <= value <= pat[2]
    # 构造器
    if kind not in ("sum", "bool"):
        return False
    labels = [c[0] for c in t["ctors"]]
    if tag not in labels:
        return False
    idx = labels.index(tag)
    if kind == "bool":
        return value is (idx == 1)
    fields = t["ctors"][idx][1]
    label = value[0] if isinstance(value, tuple) else value
    if label != tag:
        return False
    fvals = () if isinstance(value, str) else value[1:]
    return len(fvals) == len(fields) and all(
        _match_node(ref, ft, pat[j + 1], fvals[j], env)
        for j, (_fn, ft) in enumerate(fields))


def brute_force_v2_indexed(ref: RefV2Table, root: str, patterns) -> dict:
    universe = ref.all_values(root)
    covered: set = set()
    branches = []
    for i, pat in enumerate(patterns):
        fresh = []
        for v in universe:
            if v in covered:
                continue
            env = {}
            if match(ref, root, pat, v, env):
                fresh.append(v)
        if fresh:
            witness = min(fresh, key=lambda v: ref.sort_key(root, v))
            branches.append({"useful": True, "witness": witness})
            for v in universe:
                if match(ref, root, pat, v, {}):
                    covered.add(v)
        else:
            branches.append({"useful": False, "witness": None})
    leftover = [v for v in universe if v not in covered]
    uncovered = min(leftover, key=lambda v: ref.sort_key(root, v)) if leftover else None
    return {"uncovered": uncovered,
            "exhaustive": uncovered is None,
            "branches": branches}


def _any_prev(*_args, **_kw):
    return False
