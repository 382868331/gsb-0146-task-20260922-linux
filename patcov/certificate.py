"""覆盖证书装配（v2）。

成功时产出的证书是**纯数据**（仅 str/int/bool/None/list/dict），包含：

- 规范化后的类型声明、根类型、不可构造类型集合；
- 结论与最小见证；
- **有限状态闭合关系**：根状态、全部可达状态、每个状态的后继
  （构造器分支 / 整数段 / 全通配弹出 / 自由终态 / 空值域终态 / 单位终态）；
- 每个见证状态的最优（构造器节点数，前序键）。

证书中的结论字段只是*声明*；独立检查器 :mod:`patcov.certcheck` 从类型与
分支重建自己的引擎，自行计算结论、见证与最优代价，逐条核对闭合关系与
Bellman 最优方程，不信任本证书内任何布尔标志。
"""

from __future__ import annotations

CERT_FORMAT = "patcov-coverage-cert/v2"


def build_certificate(solver, table, root_sig, root_rows, status, uncovered,
                      root_ans) -> dict:
    by_id = solver.by_id
    root_key = (False, frozenset(root_rows), tuple(root_sig))

    # ---- 1. 从根沿闭合记录 BFS 编号
    state_ids: dict = {root_key: 0}
    order = [root_key]
    qi = 0
    while qi < len(order):
        k = order[qi]
        qi += 1
        rec = solver.closure.get(k)
        if rec is None:
            raise AssertionError("证书错误：闭合关系缺少状态记录")
        for edge in rec[3]:
            tk = _edge_target(edge)
            if tk is None:
                continue
            if tk not in solver.closure:
                raise AssertionError("证书错误：后继状态不在闭合关系中（非闭合）")
            if tk not in state_ids:
                state_ids[tk] = len(state_ids)
                order.append(tk)

    # ---- 2. 类型声明（规范表面）
    types_out = {}
    for name in table.order:
        d = by_id[table.ids[name]]
        if d.kind == "enum":
            types_out[name] = {"kind": "enum",
                               "constructors": [c.name for c in d.ctors]}
        elif d.kind == "product":
            types_out[name] = {"kind": "product",
                               "fields": [[fn, by_id[ft].name]
                                          for fn, ft in d.ctors[0].fields]}
        elif d.kind == "integer":
            types_out[name] = {"kind": "integer",
                               "range": [d.int_range[0], d.int_range[1]]}
        else:
            types_out[name] = {"kind": "sum",
                               "variants": [[c.name,
                                             [[fn, by_id[ft].name]
                                              for fn, ft in c.fields]]
                                            for c in d.ctors]}

    # ---- 3. 状态与转移
    states_out = {}
    state_answers = []
    for k in order:
        sid = state_ids[k]
        rows, sig, outcome, edges, ans = solver.closure[k]
        states_out[str(sid)] = solver.render_state(rows, tuple(sig))
        entry = {"state": sid, "outcome": outcome, "edges": []}
        if ans is not None:
            entry["weight"] = ans[0]
            entry["witness_key"] = list(ans[1])
        for e in edges:
            ee = {"via": e["via"]}
            if e["via"] == "constructor":
                ee["constructor_index"] = e["j"]
                ee["target"] = state_ids[_edge_target(e)]
            elif e["via"] == "integer_segment":
                ee["segment"] = [e["lo"], e["hi"]]
                ee["representative"] = e["z"]
                ee["target"] = state_ids[_edge_target(e)]
            elif e["via"] == "wild_pop":
                ee["target"] = state_ids[_edge_target(e)]
            entry["edges"].append(ee)
        state_answers.append(entry)

    return {
        "format": CERT_FORMAT,
        "types": types_out,
        "root": by_id[table.root].name,
        "nonconstructive": sorted(
            by_id[t].name for t in by_id if t not in table.productive),
        "conclusion": status,
        "witness": _render_value(uncovered),
        "root_state": 0,
        "states": states_out,
        "state_answers": state_answers,
    }


def _edge_target(e):
    via = e["via"]
    if via == "constructor":
        return (False, frozenset(e["rows"]), tuple(e["sig"]))
    if via == "integer_segment":
        return (False, frozenset(e["rows"]), tuple(e["sig"]))
    if via == "wild_pop":
        return (False, frozenset(e["rows"]), tuple(e["sig"]))
    return None


def _render_value(v):
    if isinstance(v, (bool, int, str)) or v is None:
        return v
    if isinstance(v, tuple):
        return [_render_value(x) for x in v]
    raise AssertionError(f"不可序列化的见证值 {v!r}")
