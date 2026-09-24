"""v2 核心：递归类型 + 整数闭区间上的有限状态覆盖/有用性引擎。

与 v1（:mod:`patcov.solver`，共享有序决策 DAG）并存，互不共享搜索缓存，
也不被证书检查器 :mod:`patcov.certcheck` 使用（检查器另有一份独立实现）。

有限状态，而非固定深度展开
==========================

问题状态 = 模式矩阵（行的有限集合）+ 列类型序列。每次特化消耗一个*非通配
模式头*（C/T/IE），通配头只复制通配子行；全通配列直接从列序列删去。沿任一
搜索路径非通配头严格减少，故可达状态构成**有限 DAG**——递归类型不需要任何
深度阈值即可精确判定（递归类型上的经典 Maranget 原理）。

整数列按全部相关区间端点切成有限个极大段，段内每个守卫的真值恒定，段代表
整数取该段在新序（0,-1,1,-2,2,...）下的最小值。

行 = (模式 tuple, branch, 守卫id, is_target)；模式按列对齐，行尾带一个 END
哨兵（模式格耗尽、列也耗尽时的存活标记），使“无存活行”与“有存活行覆盖
单位值 ()”可区分。

见证次序：先构造器节点数（bool/enum/sum 构造器计 1，product 与整数计 0），
再按前序键（构造器声明序编号；整数按绝对值、同绝对值负数在先）。

预算：每个记忆化调用计一个工作步；耗尽抛 :class:`BudgetStop`，由入口转换
为 ``unknown`` 结果并附未决状态，绝不谎报穷尽。
"""

from __future__ import annotations

from .guards import (
    K_FALSE,
    K_TRUE,
    ast_atom_intervals,
    ast_subst,
    cut_segments,
    int_key,
    iv_member,
    render_ast,
    segment_min_new_order,
)
from .model import value_key

W = ("W",)
END = ("END",)


class BudgetStop(Exception):
    """内部：预算耗尽。pending 为尚未得出结论的状态签名（frozenset）。"""

    def __init__(self, pending):
        self.pending = pending


class V2Solver:
    def __init__(self, table, by_id, budget: int):
        self.table = table
        self.by_id = by_id
        self.budget = budget
        self.used = 0
        self.productive = table.productive

        self._gid_map: dict = {}
        self._gid_rev: list = []
        self.g_true = self.intern_guard(K_TRUE)

        self._cover_memo: dict = {}
        self._targ_memo: dict = {}
        self._inprogress: set = set()
        # 闭合关系：key -> (rows, sig, outcome, succ_desc, answer)
        self.closure: dict = {}

        self._min_value: dict = {}
        self._min_weight: dict = {}
        self._precompute_min_values()

    # ============================================================== 基础设施
    def tick(self):
        self.used += 1
        if self.used > self.budget:
            raise BudgetStop(frozenset(self._inprogress))

    def intern_guard(self, ast) -> int:
        gid = self._gid_map.get(ast)
        if gid is None:
            gid = len(self._gid_rev)
            self._gid_map[ast] = gid
            self._gid_rev.append(ast)
        return gid

    def guard(self, gid):
        return self._gid_rev[gid]

    @staticmethod
    def _dedup(rows):
        return frozenset(rows)

    # ============================================================== 模式准备
    def prepare(self, node, tid: int):
        tag = node[0]
        decl = self.by_id[tid]
        if tag == "W":
            return W
        if tag == "T":
            fts = [ft for _, ft in decl.ctors[0].fields]
            return ("T", tuple(self.prepare(a, fts[j]) for j, a in enumerate(node[1])))
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
                    # 绑定通配 = 绑定该整数类型闭区间内的任意值
                    iv = (decl.int_range,)
                    return ("IE", iv, tuple(names))
                if cur[0] != "I":
                    raise AssertionError("整数列绑定内必须是整数模式")
                return ("IE", cur[1], tuple(names))
            return self.prepare(node[2], tid)
        raise AssertionError(f"未知模式节点 {tag!r}")

    def branch_rows(self, branch_decl, is_target: bool) -> frozenset:
        """一个分支的 or 行集合：每行 (pats, branch, gid, tgt)，尾带 END。"""
        gid = self.intern_guard(branch_decl.guard)
        rows = set()
        for root_node in branch_decl.rows:
            pnode = self.prepare(root_node, self.table.root)
            rows.add(((pnode, END), branch_decl.index, gid, is_target))
        return frozenset(rows)

    # ============================================================== 最小值
    def _precompute_min_values(self):
        """两阶段：先求每型最小构造器节点数（不动点），再在等权构造器中按
        声明序选第一个（保证平局取声明序在前的构造器）。"""
        by = self.by_id
        weight: dict = {}
        changed = True
        while changed:
            changed = False
            for tid in self.productive:
                decl = by[tid]
                if decl.kind == "integer":
                    if tid not in weight:
                        weight[tid] = 0
                        changed = True
                    continue
                best = None
                for c in decl.ctors:
                    if not all(ft in self.productive for _, ft in c.fields):
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

        choice: dict = {}
        for tid in self.productive:
            decl = by[tid]
            if decl.kind == "integer":
                lo, hi = decl.int_range
                choice[tid] = ("int", 0 if lo <= 0 <= hi else (hi if hi < 0 else lo))
                continue
            for j, c in enumerate(decl.ctors):
                if not all(ft in self.productive for _, ft in c.fields):
                    continue
                cost = 1 if decl.kind in ("bool", "enum", "sum") else 0
                cost += sum(weight[ft] for _, ft in c.fields)
                if cost == weight[tid]:
                    choice[tid] = ("ctor", j)
                    break

        values: dict = {}

        def build(tid):
            if tid in values:
                return values[tid]
            decl = by[tid]
            kind, sel = choice[tid]
            if kind == "int":
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

        for tid in self.productive:
            build(tid)
        self._min_value = values
        self._min_weight = weight

    def constructive_ctors(self, tid: int):
        decl = self.by_id[tid]
        return tuple(j for j, c in enumerate(decl.ctors)
                     if all(ft in self.productive for _, ft in c.fields))

    @staticmethod
    def _assemble(decl, idx, fv):
        if decl.kind == "bool":
            return idx == 1
        name = decl.ctors[idx].name
        if decl.kind == "product":
            return tuple(fv)
        if not fv:
            return name
        return (name, *fv)

    # ============================================================== 特化
    @staticmethod
    def _pop_wild(rows):
        return frozenset((pats[1:], b, g, t) for pats, b, g, t in rows)

    def _spec_ctor(self, rows, tid, j):
        decl = self.by_id[tid]
        arity = len(decl.ctors[j].fields)
        out = set()
        for pats, b, g, tgt in rows:
            head, tail = pats[0], pats[1:]
            if head == W:
                child = (W,) * arity
            elif head[0] == "T":
                if decl.kind != "product" or j != 0:
                    continue
                child = head[1]
            elif head[0] == "C":
                if head[1] != j:
                    continue
                child = head[2]
            else:
                continue  # IE/END 不可能出现在非末列构造器位置
            out.add((child + tail, b, g, tgt))
        return frozenset(out)

    def _int_info(self, rows):
        iv_sets, hvk = [], set()
        for pats, b, _g, _t in rows:
            head = pats[0]
            if head[0] == "IE":
                iv_sets.append(head[1])
                for name in head[2]:
                    hvk.add((b, name))
        for pats, b, g, _t in rows:
            if g != self.g_true and self.guard(g)[0] != "K":
                for key in hvk:
                    iv_sets.extend(ast_atom_intervals(self.guard(g), key))
        return iv_sets, hvk

    def _spec_int(self, rows, z, hvk):
        out = set()
        for pats, b, gid, tgt in rows:
            head, tail = pats[0], pats[1:]
            g = self.guard(gid)
            if head == W or head == END:
                out.add((pats, b, gid, tgt))
            elif head[0] == "IE":
                if not iv_member(head[1], z):
                    continue
                ng = g
                for name in head[2]:
                    if (b, name) in hvk:
                        ng = ast_subst(ng, (b, name), z)
                if ng == K_FALSE:
                    continue
                ngid = gid if ng == g else self.intern_guard(ng)
                out.add(((W,) + tail, b, ngid, tgt))
        return frozenset(out)

    def _int_segments(self, rows, tid):
        decl = self.by_id[tid]
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

    def _min_chain(self, sig):
        w, k, vals = 0, (), []
        for t in sig:
            w += self._min_weight[t]
            hv = self._min_value[t]
            k += value_key(self.by_id, t, hv)
            vals.append(hv)
        return w, k, tuple(vals)

    def _all_wild(self, rows):
        return all(pats[0] == W for pats, _b, _g, _t in rows)

    @staticmethod
    def _has_target(rows):
        return any(t for _p, _b, _g, t in rows)

    @staticmethod
    def _has_prior(rows):
        return any(not t for _p, _b, _g, t in rows)

    # ============================================================== 覆盖
    def cover(self, rows, sig, record: bool = False):
        """None=已覆盖；否则 (weight, key, value_tuple按列序)。"""
        if not sig:
            # 列空间={()}：尾哨兵 END 存活即覆盖；无存活行则 () 未覆盖
            alive = bool(rows)
            ans = None if alive else (0, (), ())
            if record:
                self.closure[(False, rows, sig)] = (
                    rows, sig, "covered" if alive else "witness",
                    ({"via": "unit_covered" if alive else "unit_witness"},),
                    ans)
            return ans
        key = (False, rows, sig)
        if key in self._cover_memo:
            return self._cover_memo[key]
        if key in self._inprogress:
            raise AssertionError("覆盖状态图出现环：非通配头计数不变量被破坏")
        self._inprogress.add(key)
        self.tick()
        succ = []
        try:
            ans = self._cover_body(rows, sig, succ, record)
        finally:
            self._inprogress.discard(key)
        self._cover_memo[key] = ans
        if record:
            self.closure[key] = (rows, sig,
                                 "covered" if ans is None else "witness",
                                 tuple(succ), ans)
        return ans

    def _cover_body(self, rows, sig, succ, record):
        tid, rest = sig[0], sig[1:]
        decl = self.by_id[tid]
        if tid not in self.productive:
            succ.append({"via": "empty_domain"})
            return None
        if not rows:
            w, k, vals = self._min_chain(sig)
            succ.append({"via": "rows_empty"})
            return (w, k, vals)
        if self._all_wild(rows):
            r = self.cover(self._pop_wild(rows), rest, record=record)
            succ.append({"via": "wild_pop", "rows": self._pop_wild(rows), "sig": rest})
            if r is None:
                return None
            hv = self._min_value[tid]
            return (self._min_weight[tid] + r[0],
                    value_key(self.by_id, tid, hv) + r[1],
                    (hv,) + r[2])
        if decl.kind == "integer":
            return self._cover_int(rows, sig, succ, record)
        best = None
        wadd = 1 if decl.kind in ("bool", "enum", "sum") else 0
        for j in self.constructive_ctors(tid):
            spec = self._spec_ctor(rows, tid, j)
            csig = tuple(ft for _, ft in decl.ctors[j].fields) + rest
            r = self.cover(spec, csig, record=record)
            succ.append({"via": "constructor", "j": j, "rows": spec, "sig": csig})
            if r is None:
                continue
            arity = len(decl.ctors[j].fields)
            v0 = self._assemble(decl, j, r[2][:arity])
            # product 为匿名唯一构造器，前序键不占编号；其余拼构造器声明序
            prefix = () if decl.kind == "product" else (j,)
            cand = (wadd + r[0], prefix + r[1], (v0,) + r[2][arity:])
            if best is None or (cand[0], cand[1]) < (best[0], best[1]):
                best = cand
        return best

    def _cover_int(self, rows, sig, succ, record):
        tid, rest = sig[0], sig[1:]
        segs, hvk = self._int_segments(rows, tid)
        best = None
        for lo, hi, z in segs:
            spec = self._spec_int(rows, z, hvk)
            tails = self._pop_wild(spec)
            r = self.cover(tails, rest, record=record)
            succ.append({"via": "integer_segment", "lo": lo, "hi": hi, "z": z,
                         "rows": tails, "sig": rest})
            if r is None:
                continue
            cand = (r[0], int_key(z) + r[1], (z,) + r[2])
            if best is None or (cand[0], cand[1]) < (best[0], best[1]):
                best = cand
        return best

    # ============================================================== 有用性
    def target(self, rows, sig):
        """带目标行的有用性。None=无用；否则 (weight,key,value_tuple)。"""
        if not self._has_target(rows):
            return None
        if not sig:
            if self._has_prior(rows):
                return None
            return (0, (), ())
        key = (True, rows, sig)
        if key in self._targ_memo:
            return self._targ_memo[key]
        if key in self._inprogress:
            raise AssertionError("有用性状态图出现环")
        self._inprogress.add(key)
        self.tick()
        try:
            ans = self._target_body(rows, sig)
        finally:
            self._inprogress.discard(key)
        self._targ_memo[key] = ans
        return ans

    def _target_body(self, rows, sig):
        tid, rest = sig[0], sig[1:]
        decl = self.by_id[tid]
        if tid not in self.productive:
            return None
        if self._all_wild(rows):
            r = self.target(self._pop_wild(rows), rest)
            if r is None:
                return None
            hv = self._min_value[tid]
            return (self._min_weight[tid] + r[0],
                    value_key(self.by_id, tid, hv) + r[1],
                    (hv,) + r[2])
        if decl.kind == "integer":
            return self._target_int(rows, sig)
        best = None
        wadd = 1 if decl.kind in ("bool", "enum", "sum") else 0
        for j in self.constructive_ctors(tid):
            spec = self._spec_ctor(rows, tid, j)
            if not self._has_target(spec):
                continue
            csig = tuple(ft for _, ft in decl.ctors[j].fields) + rest
            r = self.target(spec, csig)
            if r is None:
                continue
            arity = len(decl.ctors[j].fields)
            v0 = self._assemble(decl, j, r[2][:arity])
            prefix = () if decl.kind == "product" else (j,)
            cand = (wadd + r[0], prefix + r[1], (v0,) + r[2][arity:])
            if best is None or (cand[0], cand[1]) < (best[0], best[1]):
                best = cand
        return best

    def _target_int(self, rows, sig):
        tid, rest = sig[0], sig[1:]
        segs, hvk = self._int_segments(rows, tid)
        best = None
        for lo, hi, z in segs:
            spec = self._spec_int(rows, z, hvk)
            if not self._has_target(spec):
                continue
            r = self.target(self._pop_wild(spec), rest)
            if r is None:
                continue
            cand = (r[0], int_key(z) + r[1], (z,) + r[2])
            if best is None or (cand[0], cand[1]) < (best[0], best[1]):
                best = cand
        return best

    # ============================================================== 渲染
    def render_state(self, rows, sig):
        return {
            "columns": [self.by_id[t].name for t in sig],
            "rows": sorted(({
                "branch": b,
                "target": tgt,
                "pattern_chain": [self._render_pat(p) for p in pats],
                "guard": render_ast(self.guard(g)),
            } for pats, b, g, tgt in rows),
                key=lambda d: (d["branch"], d["target"],
                               repr(d["pattern_chain"]), repr(d["guard"]))),
        }

    def _render_pat(self, p):
        tag = p[0]
        if tag == "W":
            return "_"
        if tag == "END":
            return {"alive": True}
        if tag == "IE":
            d = {"iv": [[a, b] for a, b in p[1]]}
            if p[2]:
                d["binds"] = list(p[2])
            return {"int": d}
        if tag == "T":
            return {"tuple": [self._render_pat(a) for a in p[1]]}
        return {"ctor": p[1], "args": [self._render_pat(a) for a in p[2]]}
