"""覆盖证书的**独立**检查器（v2）。

检查器只拿到两样东西：

1. 用户原始输入（类型声明、分支模式）；
2. 求解器产出的证书（纯数据，挂在 :class:`patcov.v2.V2Result` 上）。

它不信任证书中的任何结论标志，而是用*自己的*经典 Maranget 矩阵引擎（普通
tuple 矩阵、独立 memo；不导入 :mod:`patcov.recsolver`，不读其任何缓存/闭合
记录）从原始输入重新计算结论、分支有用性、最小见证、全部有限状态与后继，
然后：

- 逐状态核对证书与独立引擎状态签名一致、后继标签/构造器/整数段一致，且
  整数段恰好无重叠无缝隙地划分类型闭区间、段代表为新序最小整数；
- 用局部最优（Bellman）方程核对每个见证状态的（构造器节点数，前序键）；
- 用独立匹配器核对遗漏见证不被任何分支匹配、有用性见证被本分支匹配且不
  被前序分支匹配。

任何不符抛 :class:`CertificateRejected`。
"""

from __future__ import annotations

from .guards import (
    K_FALSE,
    ast_atom_intervals,
    ast_subst,
    cut_segments,
    int_key,
    iv_member,
    render_ast,
    segment_min_new_order,
)
from .model import (
    build_type_table,
    normalize_branch_v2,
    table_by_id,
    value_key,
)


class CertificateRejected(ValueError):
    """证书与独立复核结果不一致。"""


# 行终结符：模式格耗尽、列也耗尽时的存活行标记（与主引擎同义，独立定义）
END = ("END",)


def _freeze(obj):
    if isinstance(obj, dict):
        return tuple(sorted((k, _freeze(v)) for k, v in obj.items()))
    if isinstance(obj, (list, tuple)):
        return tuple(_freeze(x) for x in obj)
    return obj


# ============================================================ 独立矩阵引擎

class IndepMatrixEngine:
    """经典 Maranget 模式矩阵（tuple 表示），独立 memo。

    列序列 sig：tid 的 tuple。
    行 row：(pats, branch, guard, is_target)；pats 与 sig 等长：
      'W' | ('C', idx, args) | ('T', 0, args) | ('IE', intervals, names)
    """

    def __init__(self, table):
        self.table = table
        self.by = table_by_id(table)
        mv = _MinValues(table, self.by)
        self.min_value = mv.min_value
        self.min_weight = mv.min_weight
        self.cover_memo: dict = {}
        self.target_memo: dict = {}
        # canon-key -> (outcome, weight|None, key|None, edges)
        self.cover_states: dict = {}

    # ---------------------------------------------------------- 模式准备
    def prepare(self, node, tid):
        tag = node[0]
        decl = self.by[tid]
        if tag == "W":
            return "W"
        if tag == "T":
            fts = [ft for _, ft in decl.ctors[0].fields]
            return ("T", 0,
                    tuple(self.prepare(a, fts[j]) for j, a in enumerate(node[1])))
        if tag == "C":
            idx = node[1][1]
            ctor = decl.ctors[idx]
            fts = [ft for _, ft in ctor.fields]
            return ("C", idx,
                    tuple(self.prepare(a, fts[j]) for j, a in enumerate(node[2])))
        if tag == "I":
            return ("IE", node[1], ())
        if tag == "B":
            if decl.kind == "integer":
                names, cur = [], node
                while cur[0] == "B":
                    names.append(cur[1])
                    cur = cur[2]
                if cur[0] == "W":
                    return ("IE", (decl.int_range,), tuple(names))
                return ("IE", cur[1], tuple(names))
            return self.prepare(node[2], tid)
        if tag == "END":
            return END
        raise AssertionError(tag)

    def root_rows(self, branch_decls):
        out = set()
        for d in branch_decls:
            for rn in d.rows:
                out.add(((self.prepare(rn, self.table.root), END),
                         d.index, d.guard, False))
        return frozenset(out)

    def _ckey(self, rows, sig):
        return (tuple(sig), frozenset(rows))

    @staticmethod
    def _dedup(rows):
        return frozenset(rows)

    def constructive_ctors(self, tid):
        decl = self.by[tid]
        return tuple(j for j, c in enumerate(decl.ctors)
                     if all(ft in self.table.productive for _, ft in c.fields))

    def _child_sig(self, tid, j, rest):
        ctor = self.by[tid].ctors[j]
        return tuple(ft for _, ft in ctor.fields) + rest

    @staticmethod
    def _ctor_prefix(decl, j):
        # product 为匿名唯一构造器，前序键不占编号；其余拼声明序编号
        return () if decl.kind == "product" else (j,)

    def _specialize_ctor(self, rows, tid, j):
        decl = self.by[tid]
        arity = len(decl.ctors[j].fields)
        out = []
        for pats, b, g, tgt in rows:
            head, tail = pats[0], pats[1:]
            if head == "W":
                child = ("W",) * arity
            elif head[0] == "T":
                if decl.kind != "product" or j != 0:
                    continue
                child = head[2]
            elif head[0] == "C":
                if head[1] != j:
                    continue
                child = head[2]
            else:
                continue
            out.append((child + tail, b, g, tgt))
        return self._dedup(out)

    def _int_info(self, rows):
        iv_sets, hvk = [], set()
        for pats, b, _g, _t in rows:
            head = pats[0]
            if head[0] == "IE":
                iv_sets.append(head[1])
                for name in head[2]:
                    hvk.add((b, name))
        for pats, b, g, _t in rows:
            if g[0] != "K":
                for key in hvk:
                    iv_sets.extend(ast_atom_intervals(g, key))
        return iv_sets, hvk

    def _specialize_int(self, rows, z, hvk):
        out = []
        for pats, b, g, tgt in rows:
            head, tail = pats[0], pats[1:]
            if head == "W":
                out.append((pats, b, g, tgt))
            elif head[0] == "IE":
                if not iv_member(head[1], z):
                    continue
                ng = g
                for name in head[2]:
                    if (b, name) in hvk:
                        ng = ast_subst(ng, (b, name), z)
                if ng == K_FALSE:
                    continue
                out.append((("W",) + tail, b, ng, tgt))
        return self._dedup(out)

    def _min_chain(self, sig):
        w, k, vals = 0, (), []
        for t in sig:
            w += self.min_weight[t]
            hv = self.min_value[t]
            k += value_key(self.by, t, hv)
            vals.append(hv)
        return w, k, tuple(vals)

    @staticmethod
    def _pop_wild(rows):
        return frozenset((r[0][1:], r[1], r[2], r[3]) for r in rows)

    # ---------------------------------------------------------- 覆盖
    def cover(self, rows, sig):
        """None=已覆盖；否则 (weight, key, value_tuple)（值按当前列序）。"""
        key = self._ckey(rows, sig)
        if key in self.cover_memo:
            return self.cover_memo[key]
        ans, edges = self._cover_body(rows, sig)
        self.cover_memo[key] = ans
        self.cover_states[key] = (
            "covered" if ans is None else "witness",
            None if ans is None else ans[0],
            None if ans is None else list(ans[1]),
            edges,
            tuple(sig), tuple(rows),
        )
        return ans

    def _cover_body(self, rows, sig):
        edges = []
        if not sig:
            # 列空间 = {()}：无存活行 => () 未覆盖；有存活行（守卫已常量化为真）=> 覆盖
            edges.append({"via": "unit_witness" if not rows else "unit_covered"})
            return ((0, (), ()) if not rows else None), edges
        tid, rest = sig[0], sig[1:]
        decl = self.by[tid]
        if tid not in self.table.productive:
            edges.append({"via": "empty_domain"})
            return None, edges
        if not rows:
            w, k, vals = self._min_chain(sig)
            edges.append({"via": "rows_empty"})
            return (w, k, vals), edges
        if all(r[0][0] == "W" for r in rows):
            r = self.cover(self._pop_wild(rows), rest)
            edges.append({"via": "wild_pop"})
            if r is None:
                return None, edges
            hv = self.min_value[tid]
            return (self.min_weight[tid] + r[0],
                    value_key(self.by, tid, hv) + r[1],
                    (hv,) + r[2]), edges
        if decl.kind == "integer":
            return self._cover_int(rows, sig, edges)
        best = None
        for j in self.constructive_ctors(tid):
            spec = self._specialize_ctor(rows, tid, j)
            r = self.cover(spec, self._child_sig(tid, j, rest))
            edges.append({"via": "constructor", "constructor_index": j})
            if r is None:
                continue
            arity = len(decl.ctors[j].fields)
            v0 = _assemble(decl, j, r[2][:arity])
            cand = ((1 if decl.kind in ("bool", "enum", "sum") else 0) + r[0],
                    self._ctor_prefix(decl, j) + r[1],
                    (v0,) + r[2][arity:])
            if best is None or (cand[0], cand[1]) < (best[0], best[1]):
                best = cand
        return best, edges

    def _int_segments(self, rows, tid):
        decl = self.by[tid]
        lo_typ, hi_typ = decl.int_range
        iv_sets, hvk = self._int_info(rows)
        out = []
        for slo, shi, _zrep in cut_segments(iv_sets):
            lo = lo_typ if slo is None else max(slo, lo_typ)
            hi = hi_typ if shi is None else min(shi, hi_typ)
            if lo <= hi:
                z = segment_min_new_order(lo, hi)
                if lo_typ <= z <= hi_typ:
                    out.append((lo, hi, z))
        return out, hvk

    def _cover_int(self, rows, sig, edges):
        tid, rest = sig[0], sig[1:]
        best = None
        segs, hvk = self._int_segments(rows, tid)
        for lo, hi, z in segs:
            spec = self._specialize_int(rows, z, hvk)
            r = self.cover(self._pop_wild(spec), rest)
            edges.append({"via": "integer_segment",
                          "segment": [lo, hi], "representative": z})
            if r is not None:
                cand = (r[0], int_key(z) + r[1], (z,) + r[2])
                if best is None or (cand[0], cand[1]) < (best[0], best[1]):
                    best = cand
        return best, edges

    # ---------------------------------------------------------- 有用性
    def target(self, rows, sig):
        if not any(r[3] for r in rows):
            return None
        key = ("t",) + self._ckey(rows, sig)
        if key in self.target_memo:
            return self.target_memo[key]
        ans = self._target_body(rows, sig)
        self.target_memo[key] = ans
        return ans

    def _target_body(self, rows, sig):
        if not sig:
            if any(not r[3] for r in rows):
                return None
            return (0, (), ())
        tid, rest = sig[0], sig[1:]
        decl = self.by[tid]
        if tid not in self.table.productive:
            return None
        if all(r[0][0] == "W" for r in rows):
            r = self.target(self._pop_wild(rows), rest)
            if r is None:
                return None
            hv = self.min_value[tid]
            return (self.min_weight[tid] + r[0],
                    value_key(self.by, tid, hv) + r[1],
                    (hv,) + r[2])
        if decl.kind == "integer":
            return self._target_int(rows, sig)
        best = None
        for j in self.constructive_ctors(tid):
            spec = self._specialize_ctor(rows, tid, j)
            if not any(r[3] for r in spec):
                continue
            r = self.target(spec, self._child_sig(tid, j, rest))
            if r is None:
                continue
            arity = len(decl.ctors[j].fields)
            v0 = _assemble(decl, j, r[2][:arity])
            cand = ((1 if decl.kind in ("bool", "enum", "sum") else 0) + r[0],
                    self._ctor_prefix(decl, j) + r[1],
                    (v0,) + r[2][arity:])
            if best is None or (cand[0], cand[1]) < (best[0], best[1]):
                best = cand
        return best

    def _target_int(self, rows, sig):
        tid, rest = sig[0], sig[1:]
        segs, hvk = self._int_segments(rows, tid)
        best = None
        for lo, hi, z in segs:
            spec = self._specialize_int(rows, z, hvk)
            if not any(r[3] for r in spec):
                continue
            r = self.target(self._pop_wild(spec), rest)
            if r is not None:
                cand = (r[0], int_key(z) + r[1], (z,) + r[2])
                if best is None or (cand[0], cand[1]) < (best[0], best[1]):
                    best = cand
        return best


def _assemble(decl, idx, fv):
    if decl.kind == "bool":
        return idx == 1
    name = decl.ctors[idx].name
    if decl.kind == "product":
        return tuple(fv)
    if not fv:
        return name
    return (name, *fv)


class _MinValues:
    """每个可构造类型的（最小构造器节点数，最小值），最小不动点求解。"""

    def __init__(self, table, by):
        # 阶段 1：最小构造器节点数不动点
        weight: dict = {}
        changed = True
        while changed:
            changed = False
            for tid in table.productive:
                decl = by[tid]
                if decl.kind == "integer":
                    if tid not in weight:
                        weight[tid] = 0
                        changed = True
                    continue
                best = None
                for c in decl.ctors:
                    if not all(ft in table.productive for _, ft in c.fields):
                        continue
                    cost = 1 if decl.kind in ("bool", "enum", "sum") else 0
                    ok = True
                    for _, ft in c.fields:
                        if ft not in weight:
                            ok = False
                            break
                        cost += weight[ft]
                    if ok and (best is None or cost < best):
                        best = cost
                if best is not None and tid not in weight:
                    weight[tid] = best
                    changed = True
        # 阶段 2：等权构造器按声明序取第一个；整数取新序最小
        choice: dict = {}
        for tid in table.productive:
            decl = by[tid]
            if decl.kind == "integer":
                lo, hi = decl.int_range
                choice[tid] = 0 if lo <= 0 <= hi else (hi if hi < 0 else lo)
                continue
            for j, c in enumerate(decl.ctors):
                if not all(ft in table.productive for _, ft in c.fields):
                    continue
                cost = 1 if decl.kind in ("bool", "enum", "sum") else 0
                cost += sum(weight[ft] for _, ft in c.fields)
                if cost == weight[tid]:
                    choice[tid] = j
                    break
        self.min_weight = weight
        values: dict = {}

        def build(tid):
            if tid in values:
                return values[tid]
            decl = by[tid]
            sel = choice[tid]
            if decl.kind == "integer":
                values[tid] = sel
                return sel
            ctor = decl.ctors[sel]
            fv = tuple(build(ft) for _, ft in ctor.fields)
            if decl.kind == "bool":
                v = sel == 1
            elif decl.kind == "product":
                v = tuple(fv)
            elif not ctor.fields:
                v = ctor.name
            else:
                v = (ctor.name, *fv)
            values[tid] = v
            return v

        for tid in table.productive:
            build(tid)
        self.min_value = values


# ============================================================ 独立匹配器

def match_branch(by, branch_decl, root_tid, value):
    """值是否匹配某分支（任一 or 行结构匹配且守卫在绑定下为真）。"""
    for rn in branch_decl.rows:
        env: dict = {}
        if _match_node(by, rn, root_tid, value, env) and \
                _eval_guard(branch_decl.guard, branch_decl.index, env):
            return True
    return False


def _match_node(by, node, tid, value, env):
    tag = node[0]
    decl = by[tid]
    if tag == "W":
        return True
    if tag == "B":
        save = dict(env)
        if _match_node(by, node[2], tid, value, env):
            env[node[1]] = value
            return True
        env.clear()
        env.update(save)
        return False
    if tag == "I":
        return (isinstance(value, int) and not isinstance(value, bool)
                and iv_member(node[1], value))
    if tag == "T":
        if decl.kind != "product" or not isinstance(value, tuple):
            return False
        fields = decl.ctors[0].fields
        return len(value) == len(fields) and all(
            _match_node(by, node[1][j], fields[j][1], value[j], env)
            for j in range(len(fields)))
    if tag == "C":
        idx = node[1][1]
        ctor = decl.ctors[idx]
        if decl.kind == "bool":
            return value is (idx == 1)
        label = value[0] if isinstance(value, tuple) else value
        if label != ctor.name:
            return False
        fvals = () if isinstance(value, str) else value[1:]
        return len(fvals) == len(ctor.fields) and all(
            _match_node(by, node[2][j], ctor.fields[j][1], fvals[j], env)
            for j in range(len(ctor.fields)))
    raise AssertionError(tag)


def _eval_guard(ast, branch, env):
    tag = ast[0]
    if tag == "K":
        return ast[1]
    if tag == "A":
        z = env.get(ast[1][1])  # (branch, name) -> name
        return z is not None and iv_member(ast[2], z)
    if tag == "!":
        return not _eval_guard(ast[1], branch, env)
    a = _eval_guard(ast[1], branch, env)
    b = _eval_guard(ast[2], branch, env)
    return (a and b) if tag == "&" else (a or b)


# ============================================================ 证书核对

def verify_certificate(types, raw_patterns, result) -> dict:
    """用独立引擎与独立匹配器复核 result。返回摘要 dict；不符抛异常。"""
    if getattr(result, "status", None) == "unknown":
        raise CertificateRejected(
            "不能核对 unknown 结果：预算内未得出结论，不存在可核对的闭合证书；"
            "请调大 budget 重新分析")
    table = build_type_table(types, version=2)
    by = table_by_id(table)
    decls = [normalize_branch_v2(table, raw, i, f"patterns[{i}]")
             for i, raw in enumerate(raw_patterns)]

    eng = IndepMatrixEngine(table)

    # ---- 1. 逐分支独立重算有用性
    prior_rows: frozenset = frozenset()
    expected = []
    for d in decls:
        tgt = frozenset((p, b, g, True) for p, b, g, _t in eng.root_rows([d]))
        r = eng.target(prior_rows | tgt, (table.root,))
        wit = None if r is None else r[2][0]
        expected.append((r is not None, wit))
        new_rows = frozenset(row for row in eng.root_rows([d]) if row[2] != K_FALSE)
        prior_rows = prior_rows | new_rows

    if len(result.branches) != len(expected):
        raise CertificateRejected("证书分支数与输入不符")
    for got, (useful, wit) in zip(result.branches, expected):
        if bool(got.useful) != useful or got.useful is None:
            raise CertificateRejected(
                f"分支 {got.index} 有用性不一致：证书 {got.useful}，独立 {useful}")
        if useful:
            if got.witness != wit:
                raise CertificateRejected(
                    f"分支 {got.index} 最小见证不一致：证书 {got.witness!r}，独立 {wit!r}")
            if any(match_branch(by, decls[k], table.root, wit)
                   for k in range(got.index)):
                raise CertificateRejected(f"分支 {got.index} 见证被前序分支匹配")
            if not match_branch(by, decls[got.index], table.root, wit):
                raise CertificateRejected(f"分支 {got.index} 见证不被本分支匹配")

    # ---- 2. 整体遗漏独立重算
    root_r = eng.cover(prior_rows, (table.root,))
    exp_status = "exhaustive" if root_r is None else "incomplete"
    exp_wit = None if root_r is None else root_r[2][0]
    if result.status != exp_status:
        raise CertificateRejected(
            f"结论不一致：证书 {result.status}，独立 {exp_status}")
    if result.uncovered != exp_wit:
        raise CertificateRejected(
            f"最小遗漏见证不一致：证书 {result.uncovered!r}，独立 {exp_wit!r}")
    if exp_wit is not None and any(
            match_branch(by, d, table.root, exp_wit) for d in decls):
        raise CertificateRejected("最小遗漏见证竟然被某分支匹配")

    summary = {"conclusion": exp_status, "uncovered": exp_wit,
               "states_checked": 0, "edges_checked": 0}
    if result.certificate is not None:
        _verify_closure(eng, table, result.certificate, prior_rows)
        summary["states_checked"] = len(result.certificate["states"])
        summary["edges_checked"] = sum(
            len(a["edges"]) for a in result.certificate["state_answers"])
    return summary


def _state_signature(eng, rows, sig):
    return _freeze({
        "columns": [eng.by[t].name for t in sig],
        "rows": sorted(({
            "branch": b,
            "target": tgt,
            "pattern_chain": [_render_pat(p) for p in pats],
            "guard": render_ast(g),
        } for pats, b, g, tgt in rows),
            key=lambda d: (d["branch"], d["target"], repr(d["pattern_chain"]),
                           repr(d["guard"]))),
    })


def _render_pat(p):
    if p == "W":
        return "_"
    tag = p[0]
    if tag == "END":
        return {"alive": True}
    if tag == "IE":
        d = {"iv": [[a, b] for a, b in p[1]]}
        if p[2]:
            d["binds"] = list(p[2])
        return {"int": d}
    if tag == "T":
        return {"tuple": [_render_pat(a) for a in p[2]]}
    return {"ctor": p[1], "args": [_render_pat(a) for a in p[2]]}


def _verify_closure(eng, table, cert, root_rows):
    if cert.get("format") != "patcov-coverage-cert/v2":
        raise CertificateRejected("证书格式标识不符")
    by = eng.by
    for name in table.order:
        d = by[table.ids[name]]
        c = cert["types"].get(name)
        if c is None or c.get("kind") != d.kind:
            raise CertificateRejected(f"证书类型 {name!r} 声明不符")
        if d.kind == "integer" and list(d.int_range) != c.get("range"):
            raise CertificateRejected(f"证书整数类型 {name!r} range 不符")
    if cert["root"] != by[table.root].name:
        raise CertificateRejected("证书根类型不符")
    noncon = set(cert.get("nonconstructive", []))
    expect = {by[t].name for t in by if t not in table.productive}
    if noncon != expect:
        raise CertificateRejected(
            f"不可构造类型集合不符：证书 {sorted(noncon)}，独立 {sorted(expect)}")

    cert_states = cert["states"]
    cert_answers = {a["state"]: a for a in cert["state_answers"]}

    def find_state(rows, sig):
        key = ("c",) + eng._ckey(rows, sig)
        rec = eng.cover_states.get(eng._ckey(rows, sig))
        if rec is None:
            # 驱动独立引擎计算该状态
            eng.cover(rows, sig)
            rec = eng.cover_states[eng._ckey(rows, sig)]
        return rec

    visited = set()

    def walk(cid, rows, sig):
        if cid in visited:
            return
        visited.add(cid)
        cstate = cert_states.get(str(cid))
        if cstate is None:
            raise CertificateRejected(f"证书缺少状态 {cid}")
        if _freeze(cstate) != _state_signature(eng, rows, sig):
            raise CertificateRejected(
                f"证书状态 {cid} 与独立引擎状态签名不一致：{cstate}")
        rec = eng.cover_states[eng._ckey(rows, sig)]
        outcome, weight, wkey, edges, _sig, _rows = rec
        ans = cert_answers.get(cid)
        if ans is None:
            raise CertificateRejected(f"证书状态 {cid} 缺少答案记录")
        if ans["outcome"] != outcome:
            raise CertificateRejected(
                f"状态 {cid} 结论不符：证书 {ans['outcome']}，独立 {outcome}")
        if outcome == "witness":
            if ans.get("weight") != weight:
                raise CertificateRejected(
                    f"状态 {cid} 最优构造器节点数不符：{ans.get('weight')} vs {weight}")
            if list(ans.get("witness_key", [])) != list(wkey):
                raise CertificateRejected(
                    f"状态 {cid} 见证键不符：{ans.get('witness_key')} vs {wkey}")
        cert_edges = ans.get("edges", [])
        _check_edges(cid, cert_edges, edges, eng, rows, sig, walk)

    walk(cert["root_state"], root_rows, (table.root,))
    if visited != {int(x) for x in cert_states}:
        raise CertificateRejected("证书含从根不可达或多余的状态")
    _bellman_check(eng)


def _edge_label(e):
    via = e["via"]
    if via == "constructor":
        return ("ctor", e["constructor_index"])
    if via == "integer_segment":
        return ("int", tuple(e["segment"]), e["representative"])
    return (via,)


def _check_edges(cid, cert_edges, indep_edges, eng, rows, sig, walk):
    a = sorted((_edge_label(e) for e in cert_edges), key=repr)
    b = sorted((_edge_label(e) for e in indep_edges), key=repr)
    if a != b:
        raise CertificateRejected(
            f"状态 {cid} 后继不符：证书 {a}，独立 {b}")
    tid = sig[0] if sig else None
    decl = eng.by[tid] if tid is not None else None
    for ce, ie in zip(sorted(cert_edges, key=lambda e: repr(_edge_label(e))),
                      sorted(indep_edges, key=lambda e: repr(_edge_label(e)))):
        via = ie["via"]
        if via == "constructor":
            j = ie["constructor_index"]
            nrows = eng._specialize_ctor(rows, tid, j)
            nsig = eng._child_sig(tid, j, sig[1:])
            walk(ce["target"], nrows, nsig)
        elif via == "integer_segment":
            lo, hi = ie["segment"]
            z = ie["representative"]
            _check_segment(cid, tid, eng, lo, hi, z)
            _segs, hvk = eng._int_segments(rows, tid)
            spec = eng._specialize_int(rows, z, hvk)
            walk(ce["target"], eng._pop_wild(spec), sig[1:])
        elif via == "wild_pop":
            walk(ce["target"], eng._pop_wild(rows), sig[1:])
        else:
            if "target" in ce:
                raise CertificateRejected(f"状态 {cid} 终态边不应有后继")


def _check_segment(cid, tid, eng, lo, hi, z):
    lo_typ, hi_typ = eng.by[tid].int_range
    if lo < lo_typ or hi > hi_typ:
        raise CertificateRejected(f"状态 {cid} 整数段超出类型区间")
    if z != segment_min_new_order(lo, hi):
        raise CertificateRejected(f"状态 {cid} 段代表不是新序最小整数")
    if lo > hi:
        raise CertificateRejected(f"状态 {cid} 空整数段")


def _bellman_check(eng: IndepMatrixEngine):
    """每个见证状态必须满足局部最优（Bellman）方程。"""
    for rec in eng.cover_states.values():
        outcome, weight, wkey, edges, sig, rows = rec
        vias = [e["via"] for e in edges]
        if outcome == "covered":
            if "rows_empty" in vias or "unit_witness" in vias:
                raise CertificateRejected("covered 状态不得含见证终态边")
            continue
        tid = sig[0] if sig else None
        decl = eng.by[tid] if tid is not None else None
        if vias == ["rows_empty"]:
            w, k, _ = eng._min_chain(sig)
            cand = (w, k)
        elif vias == ["unit_witness"]:
            cand = (0, ())
        elif "wild_pop" in vias:
            r = eng.cover(eng._pop_wild(rows), sig[1:])
            hv = eng.min_value[tid]
            cand = (eng.min_weight[tid] + r[0],
                    value_key(eng.by, tid, hv) + r[1]) if r is not None else None
        elif decl.kind == "integer":
            cand = None
            segs, hvk = eng._int_segments(rows, tid)
            for lo, hi, z in segs:
                spec = eng._specialize_int(rows, z, hvk)
                r = eng.cover(eng._pop_wild(spec), sig[1:])
                if r is not None:
                    c = (r[0], int_key(z) + r[1])
                    if cand is None or c < cand:
                        cand = c
        else:
            cand = None
            for j in eng.constructive_ctors(tid):
                spec = eng._specialize_ctor(rows, tid, j)
                r = eng.cover(spec, eng._child_sig(tid, j, sig[1:]))
                if r is not None:
                    prefix = () if decl.kind == "product" else (j,)
                    c = ((1 if decl.kind in ("bool", "enum", "sum") else 0) + r[0],
                         prefix + r[1])
                    if cand is None or c < cand:
                        cand = c
        if cand is None:
            raise CertificateRejected("witness 状态无见证后继")
        if cand[0] != weight or list(cand[1]) != list(wkey):
            raise CertificateRejected(
                f"Bellman 方程不符：记录 ({weight},{wkey})，方程值 {cand}")
    # 整数段划分完备性（无重叠无缝隙覆盖类型闭区间）
    for rec in eng.cover_states.values():
        edges = rec[3]
        sig = rec[4]
        ints = [e for e in edges if e["via"] == "integer_segment"]
        if not ints:
            continue
        lo_typ, hi_typ = eng.by[sig[0]].int_range
        segs = sorted((e["segment"][0], e["segment"][1]) for e in ints)
        if segs[0][0] != lo_typ or segs[-1][1] != hi_typ:
            raise CertificateRejected("整数段未覆盖整个类型区间")
        for (l1, h1), (l2, h2) in zip(segs, segs[1:]):
            if h1 + 1 != l2:
                raise CertificateRejected("整数段有重叠或缝隙")
