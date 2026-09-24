"""类型表与模式的规范化、校验，以及见证值与字典序键。

公开输入形态（详见 README）：

类型声明 ``types`` 为有序 ``dict``，第一项是被测（scrutinee）根类型：

- ``{"kind": "bool"}``              —— 仅内置名 ``"Bool"``，用户不得重声明
- ``{"kind": "enum", "constructors": ["A", "B"]}``
- ``{"kind": "product", "fields": [("x", "Bool"), ("y", "Color")]}``
- ``{"kind": "sum", "variants": [("C", [("f", "Bool")]), ("D", [])]}``

模式（每个分支一个，针对根类型）：

- ``"_"``                 通配
- ``"A"``                 零元构造器（Bool 用字符串 ``"True"`` / ``"False"``）
- ``("C", p1, ..., pn)``  构造器模式
- ``[p1, ..., pn]``       积（元组）模式
- ``("or", p1, ..., pn)`` 或模式（允许重复备选，语义为集合求并）

见证值（分析输出）：

- Bool：Python ``bool``；枚举/零元变体：标签字符串
- 积：``tuple``；带字段变体：``(标签, 字段值...)``
"""

from __future__ import annotations

from dataclasses import dataclass

from .errors import CoverageError
from .guards import compile_guard, int_key

# ---------------------------------------------------------------- 常量与上限

MAX_CONSTRUCTORS = 128
MAX_NESTING = 12
# 单个构造器（含 product 的匿名构造器）字段数上限。
# 这是受 Python 原生递归栈约束的工程边界：位置序列长度按字段数增长，
# 嵌套深度 ≤12，故最坏序列长度 ≤ 12×该上限；求解器据此设置递归上限，
# 保证上限内输入在 Linux 原生 Python 3.14 上不会触发 RecursionError。
MAX_FIELD_ARITY = 10_000
# 单分支模式节点上限：与字段数上限同量级，主要约束宽积的显式字段模式。
MAX_PATTERN_NODES = 1_000_000
# v2：整数类型 range 端点与整数字面量/区间模式端点的绝对值上限。
# 题面要求端点绝对值可达 10^9；留到 10^18 作为工程边界（精确整数运算）。
MAX_INT_ENDPOINT_ABS = 10**18
BOOL_NAME = "Bool"
WILD = "_"
OR_TAG = "or"
BIND_TAG = "bind"
WHEN_TAG = "when"
RANGE_TAG = "range"
RESERVED_LABELS = (WILD, OR_TAG, BIND_TAG, WHEN_TAG, RANGE_TAG)
INTEGER_KIND = "integer"


# ---------------------------------------------------------------- 模式内部表示

# ('W',)                         通配
# ('C', ctor_gid, args)          构造器，ctor_gid = (类型名, 输入次序编号)
# ('T', args)                    积
# ('O', alts)                    或
# args/alts 均为节点 tuple

_WILD_NODE = ("W",)


def P_wild() -> tuple:
    return _WILD_NODE


def P_con(gid, args: tuple) -> tuple:
    return ("C", gid, args)


def P_tuple(args: tuple) -> tuple:
    return ("T", args)


def P_or(alts: tuple) -> tuple:
    return ("O", alts)


# ---------------------------------------------------------------- 类型表

@dataclass(frozen=True)
class Ctor:
    name: str
    fields: tuple          # tuple[tuple[字段名, 类型id], ...]
    decl: str              # 所属类型名


@dataclass(frozen=True)
class TDecl:
    name: str
    kind: str              # 'bool' | 'enum' | 'sum' | 'product' | 'integer'
    ctors: tuple           # bool/enum/sum 的构造器；product 为单个匿名 Ctor；integer 为 ()
    field_names: tuple     # product 的字段名；其余为 ()
    int_range: tuple = ()  # integer：闭区间 (lo, hi)


@dataclass(frozen=True)
class TypeTable:
    decls: dict            # name -> TDecl
    ids: dict              # name -> int
    root: int
    order: tuple           # 用户声明顺序（不含内置 Bool）
    version: int = 1       # 1=旧契约（拒绝递归/整数）；2=递归+整数+守卫
    productive: frozenset = frozenset()  # v2：可构造（值域非空）类型 id 集


def _err(path, msg: str) -> CoverageError:
    return CoverageError(f"{path}: {msg}" if path else msg)


def _check_name(path, value, what: str) -> str:
    if not isinstance(value, str) or not value:
        raise _err(path, f"{what}必须是非空字符串，得到 {type(value).__name__!s}")
    return value


def table_by_id(table: TypeTable) -> dict:
    return {table.ids[n]: table.decls[n] for n in list(table.order) + [BOOL_NAME]}


def build_type_table(types, *, version: int = 1) -> TypeTable:
    """校验并构造类型表。第一项声明为根（scrutinee）类型。

    首项允许且仅允许是 ``{"Bool": {"kind": "bool"}}``，表示根直接取内置 Bool
    （这不是重声明：内置 Bool 仍由库提供，且该写法只能出现在首项）。

    :param version:
        - ``1``（默认，旧契约）：只允许 bool/enum/product/sum，类型必须
          **非递归**，嵌套深度 ≤ 12；递归类型在输入阶段被拒绝。
        - ``2``：允许**递归/互递归**类型与 ``kind: "integer"`` 整数闭区间
          类型；所有声明先做**生产性检查**（最小不动点），不可构造类型合法
          存在，但其值域为空。
    """
    if version not in (1, 2):
        raise _err("version", f"仅支持版本 1 或 2，得到 {version!r}")
    if not isinstance(types, dict):
        raise _err("types", f"必须是 dict（有序映射，首项为根类型），得到 {type(types).__name__}")
    if not types:
        raise _err("types", "类型表为空，至少需要一个根类型声明")

    ids: dict[str, int] = {}
    decls: dict[str, TDecl] = {}

    # 识别内置 Bool 根
    bool_root = False
    for idx0, (bname, bspec) in enumerate(types.items()):
        if bname == BOOL_NAME:
            if idx0 != 0:
                raise _err(f"types 的键 {idx0}",
                          f"{BOOL_NAME} 为内置保留名；只允许作为首项（根）且声明为 {{'kind': 'bool'}}")
            if bspec != {"kind": "bool"}:
                raise _err(f"types[{BOOL_NAME!r}]",
                          f"内置 {BOOL_NAME} 根的声明必须恰好是 {{'kind': 'bool'}}")
            bool_root = True

    order = tuple(n for n in types.keys() if n != BOOL_NAME)

    for idx, name in enumerate(order):
        spec = types[name]
        _check_name(f"types 的键 {idx}", name, "类型名")
        if not isinstance(spec, dict):
            raise _err(f"types[{name!r}]", f"声明必须是 dict，得到 {type(spec).__name__}")
        allowed = {"bool", "enum", "product", "sum"} | ({"integer"} if version == 2 else set())
        if not isinstance(spec.get("kind"), str) or spec.get("kind") not in allowed:
            raise _err(
                f"types[{name!r}]",
                f"'kind' 必须是 {sorted(allowed)} 之一，得到 {spec.get('kind')!r}"
                + ("（integer 类型需要 version=2）" if spec.get("kind") == "integer" else ""),
            )
        if spec["kind"] == "bool":
            raise _err(f"types[{name!r}]", f"用户类型不得声明 kind='bool'；请直接引用内置 {BOOL_NAME}，或改用 enum")
        allowed_keys = {"kind", "constructors", "fields", "variants"}
        if version == 2:
            allowed_keys.add(RANGE_TAG)
        extra = set(spec) - allowed_keys
        if extra:
            raise _err(f"types[{name!r}]", f"出现未知键 {sorted(extra)}；合法键为 {sorted(allowed_keys)}")
        ids[name] = idx

    total_ctors = 0
    for name in order:
        spec = types[name]
        p = f"types[{name!r}]"
        kind = spec["kind"]

        if kind == "integer":
            if set(spec) != {"kind", RANGE_TAG}:
                raise _err(p, "integer 声明只能含 kind 与 'range'")
            rng = spec[RANGE_TAG]
            if not isinstance(rng, (list, tuple)) or len(rng) != 2:
                raise _err(p + f"[{RANGE_TAG!r}]", "整数类型必须给出闭区间 [左端点, 右端点] 二元组")
            lo, hi = rng
            if isinstance(lo, bool) or not isinstance(lo, int):
                raise _err(p + f"[{RANGE_TAG!r}][0]",
                           f"左端点必须是 int（拒绝 bool），得到 {type(lo).__name__}")
            if isinstance(hi, bool) or not isinstance(hi, int):
                raise _err(p + f"[{RANGE_TAG!r}][1]",
                           f"右端点必须是 int（拒绝 bool），得到 {type(hi).__name__}")
            if abs(lo) > MAX_INT_ENDPOINT_ABS or abs(hi) > MAX_INT_ENDPOINT_ABS:
                raise _err(p + f"[{RANGE_TAG!r}]",
                           f"整数端点绝对值超过上限 {MAX_INT_ENDPOINT_ABS}：[{lo}, {hi}]")
            if lo > hi:
                raise _err(p + f"[{RANGE_TAG!r}]",
                           f"空整数闭区间 [{lo}, {hi}]：左端不得大于右端；"
                           "空值域请用不可构造的递归类型表达")
            decls[name] = TDecl(name, "integer", (), (), (lo, hi))

        elif kind == "enum":
            if set(spec) != {"kind", "constructors"}:
                raise _err(p, "enum 声明只能含 kind 与 constructors")
            raw = spec["constructors"]
            if not isinstance(raw, (list, tuple)) or not raw:
                raise _err(p + "['constructors']", "必须是非空列表")
            labels = []
            for j, lab in enumerate(raw):
                lp = f"{p}['constructors'][{j}]"
                _check_name(lp, lab, "构造器名")
                if lab in RESERVED_LABELS:
                    raise _err(lp, f"构造器名 {lab!r} 为模式语法保留字")
                if lab in labels:
                    raise _err(lp, f"构造器名 {lab!r} 在该类型内重复")
                labels.append(lab)
            decls[name] = TDecl(name, "enum", tuple(Ctor(lab, (), name) for lab in labels), ())
            total_ctors += len(labels)

        elif kind == "product":
            if set(spec) != {"kind", "fields"}:
                raise _err(p, "product 声明只能含 kind 与 fields")
            raw = spec["fields"]
            if not isinstance(raw, (list, tuple)) or not raw:
                raise _err(p + "['fields']", "必须是非空列表")
            fields = []
            seen = set()
            for j, item in enumerate(raw):
                fp = f"{p}['fields'][{j}]"
                if not isinstance(item, (list, tuple)) or len(item) != 2:
                    raise _err(fp, "字段必须是 [字段名, 类型名] 二元组")
                fname, ftype = item
                _check_name(fp + "[0]", fname, "字段名")
                _check_name(fp + "[1]", ftype, "类型名")
                if fname in seen:
                    raise _err(fp + "[0]", f"字段名 {fname!r} 在该类型内重复")
                seen.add(fname)
                fields.append((fname, ftype))
            if len(fields) > MAX_FIELD_ARITY:
                raise _err(
                    p + "['fields']",
                    f"product 字段数 {len(fields)} 超过上限 {MAX_FIELD_ARITY}（受原生递归栈约束的接口边界）",
                )
            anon = Ctor("·tuple·", tuple(fields), name)
            decls[name] = TDecl(name, "product", (anon,), tuple(f[0] for f in fields))

        else:  # sum
            if set(spec) != {"kind", "variants"}:
                raise _err(p, "sum 声明只能含 kind 与 variants")
            raw = spec["variants"]
            if not isinstance(raw, (list, tuple)) or not raw:
                raise _err(p + "['variants']", "必须是非空列表")
            ctors = []
            seen_labels = set()
            for j, item in enumerate(raw):
                vp = f"{p}['variants'][{j}]"
                if not isinstance(item, (list, tuple)) or len(item) != 2:
                    raise _err(vp, "变体必须是 [构造器名, [[字段名, 类型名], ...]] 二元组")
                lab, fraw = item
                _check_name(vp + "[0]", lab, "构造器名")
                if lab in RESERVED_LABELS:
                    raise _err(vp + "[0]", f"构造器名 {lab!r} 为模式语法保留字")
                if lab in seen_labels:
                    raise _err(vp + "[0]", f"构造器名 {lab!r} 在该类型内重复")
                seen_labels.add(lab)
                if not isinstance(fraw, (list, tuple)):
                    raise _err(vp + "[1]", "字段列表必须是列表（允许空列表）")
                fields = []
                seen_f = set()
                for k, fi in enumerate(fraw):
                    fp2 = f"{vp}[1][{k}]"
                    if not isinstance(fi, (list, tuple)) or len(fi) != 2:
                        raise _err(fp2, "字段必须是 [字段名, 类型名] 二元组")
                    fname, ftype = fi
                    _check_name(fp2 + "[0]", fname, "字段名")
                    _check_name(fp2 + "[1]", ftype, "类型名")
                    if fname in seen_f:
                        raise _err(fp2 + "[0]", f"字段名 {fname!r} 在该变体内重复")
                    seen_f.add(fname)
                    fields.append((fname, ftype))
                if len(fields) > MAX_FIELD_ARITY:
                    raise _err(
                        vp + "[1]",
                        f"变体 {lab!r} 字段数 {len(fields)} 超过上限 {MAX_FIELD_ARITY}（受原生递归栈约束的接口边界）",
                    )
                ctors.append(Ctor(lab, tuple(fields), name))
            decls[name] = TDecl(name, "sum", tuple(ctors), ())
            total_ctors += len(ctors)

    if total_ctors > MAX_CONSTRUCTORS:
        raise _err(
            "types",
            f"用户声明的构造器总数为 {total_ctors}，超过上限 {MAX_CONSTRUCTORS}（不含内置 Bool 的 2 个）",
        )

    # 内置 Bool（编号排在用户类型之后）
    bool_id = len(order)
    ids[BOOL_NAME] = bool_id
    decls[BOOL_NAME] = TDecl(
        BOOL_NAME,
        "bool",
        (Ctor("False", (), BOOL_NAME), Ctor("True", (), BOOL_NAME)),
        (),
    )

    def resolve(tname: str, path: str) -> int:
        if tname not in ids:
            raise _err(path, f"未知类型 {tname!r}；已声明：{sorted(ids)}")
        return ids[tname]

    # 字段类型引用：字符串名 -> id
    for name in order:
        d = decls[name]
        if d.kind == "product":
            c = d.ctors[0]
            fields = tuple(
                (fn, resolve(ft, f"types[{name!r}]['fields'][{j}][1]"))
                for j, (fn, ft) in enumerate(c.fields)
            )
            decls[name] = TDecl(name, "product", (Ctor(c.name, fields, name),), d.field_names)
        elif d.kind == "sum":
            newctors = []
            for j, c in enumerate(d.ctors):
                fields = tuple(
                    (fn, resolve(ft, f"types[{name!r}]['variants'][{j}][1][{k}][1]"))
                    for k, (fn, ft) in enumerate(c.fields)
                )
                newctors.append(Ctor(c.name, fields, name))
            decls[name] = TDecl(name, "sum", tuple(newctors), ())

    provisional_root = ids[BOOL_NAME] if bool_root else ids[order[0]]
    by_id = table_by_id(TypeTable(decls, ids, provisional_root, order, version))

    def _child_tids(d: TDecl):
        return [ft for c in d.ctors for _, ft in c.fields]

    if version == 1:
        # 递归检测 + 嵌套深度（深度按字段展开的最长链计，1 = 一层复合）
        visiting: set[int] = set()
        done: dict[int, int] = {}

        def depth(tid: int) -> int:
            if tid in done:
                return done[tid]
            d = by_id[tid]
            if tid in visiting:
                raise _err(f"types[{d.name!r}]", "检测到递归类型；契约只允许有限非递归类型")
            visiting.add(tid)
            worst = 0
            children = _child_tids(d)
            if children:
                worst = 1 + max(depth(ch) for ch in children)
            visiting.discard(tid)
            done[tid] = worst
            if worst > MAX_NESTING:
                raise _err(f"types[{d.name!r}]", f"类型嵌套深度 {worst} 超过上限 {MAX_NESTING}")
            return worst

        for name in order:
            depth(ids[name])
        productive = frozenset(ids[n] for n in order) | frozenset([ids[BOOL_NAME]])
    else:
        # v2：生产性 = 存在有限深度的构造器展开。按最小不动点迭代：
        #   bool / integer 可构造；
        #   product 可构造当且仅当其所有字段类型可构造；
        #   enum/sum 可构造当且仅当至少有一个构造器，其所有字段类型可构造。
        # 起始即不可达（无论多少步迭代都不亮）的类型即为空递归类型，
        # 合法存在但值域为空。内置 Bool 恒可构造，作为迭代种子。
        productive = {ids[BOOL_NAME]}
        changed = True
        while changed:
            changed = False
            for name in order:
                tid = ids[name]
                if tid in productive:
                    continue
                d = by_id[tid]
                if d.kind in ("bool", "integer"):
                    ok = True
                elif d.kind == "product":
                    # 无字段积可立即构造；否则要求所有字段类型均可构造
                    ok = all(ft in productive for _, ft in d.ctors[0].fields)
                else:  # enum / sum
                    ok = any(all(ft in productive for _, ft in c.fields)
                             for c in d.ctors)
                if ok:
                    productive.add(tid)
                    changed = True
        productive = frozenset(productive) | frozenset([ids[BOOL_NAME]])

    root = ids[BOOL_NAME] if bool_root else ids[order[0]]
    return TypeTable(decls=decls, ids=ids, root=root, order=order,
                     version=version, productive=productive)


# ---------------------------------------------------------------- 模式规范化

def normalize(table: TypeTable, pat, tid: int, path: str) -> tuple:
    """把公开模式形态校验并转为内部节点；path 用于错误定位（如 patterns[3]）。"""
    by_id = table_by_id(table)
    counter = [0]

    def go(p, t: int, pth: str) -> tuple:
        counter[0] += 1
        if counter[0] > MAX_PATTERN_NODES:
            raise _err(path, f"模式节点数超过上限 {MAX_PATTERN_NODES}（路径 {pth}）")
        decl = by_id[t]

        if p == WILD:
            return P_wild()
        if isinstance(p, bool) or not isinstance(p, (str, list, tuple)):
            raise _err(
                pth,
                "模式必须是 '_'(通配)、str(零元构造器)、list(积模式) 或 tuple(构造器/or 模式)，"
                f"得到 {type(p).__name__}",
            )
        if isinstance(p, list):
            if decl.kind != "product":
                raise _err(pth, f"出现积模式 [...]，但该位置类型 {decl.name!r} 不是 product")
            fields = decl.ctors[0].fields
            if len(p) != len(fields):
                raise _err(pth, f"积模式字段数 {len(p)} 与类型 {decl.name!r} 的 {len(fields)} 个字段不符")
            return P_tuple(tuple(go(x, fields[j][1], f"{pth}[{j}]") for j, x in enumerate(p)))
        if isinstance(p, str):
            return make_con(p, (), decl, pth, go)

        if not p:
            raise _err(pth, "空 tuple 不是合法模式（构造器模式至少含标签，or 至少含一个备选）")
        tag = p[0]
        if not isinstance(tag, str):
            raise _err(pth + "[0]", f"tuple 模式首元素必须是构造器标签或 'or'，得到 {type(tag).__name__}")
        if tag == OR_TAG:
            if len(p) < 2:
                raise _err(pth, "or 模式至少需要一个备选")
            return P_or(tuple(go(x, t, f"{pth}[{j + 1}]") for j, x in enumerate(p[1:])))
        return make_con(tag, p[1:], decl, pth, go)

    def make_con(tag: str, raw_args, decl: TDecl, pth: str, go) -> tuple:
        if decl.kind == "product":
            raise _err(pth, f"类型 {decl.name!r} 是 product，此处必须用积模式 [...]，不能用构造器 {tag!r}")
        idx = next((j for j, c in enumerate(decl.ctors) if c.name == tag), None)
        if idx is None:
            names = [c.name for c in decl.ctors]
            raise _err(pth, f"构造器 {tag!r} 不属于类型 {decl.name!r}；该类型构造器（按输入次序）：{names}")
        ctor = decl.ctors[idx]
        if len(raw_args) != len(ctor.fields):
            raise _err(pth, f"构造器 {tag!r} 需要 {len(ctor.fields)} 个字段模式，得到 {len(raw_args)} 个")
        args = tuple(go(x, ctor.fields[j][1], f"{pth}[{j + 1}]") for j, x in enumerate(raw_args))
        return P_con((decl.name, idx), args)

    root_node = go(pat, tid, path)
    _check_depth(root_node, by_id, tid, path)
    return root_node


def _check_depth(node, by_id, tid, path):
    """模式嵌套深度，与类型同口径：根为 1，逐层 +1，上限 12。"""
    def depth(n, t) -> int:
        tag = n[0]
        decl = by_id[t]
        if tag == "W":
            return 1
        if tag == "O":
            return 1 + max(depth(a, t) for a in n[1])
        if tag == "T":
            fts = decl.ctors[0].fields
            inner = (depth(a, fts[j][1]) for j, a in enumerate(n[1]))
            return 1 + (max(inner) if n[1] else 0)
        cidx = n[1][1]
        ctor = decl.ctors[cidx]
        if not ctor.fields:
            return 1
        return 1 + max(depth(a, ctor.fields[j][1]) for j, a in enumerate(n[2]))

    d = depth(node, tid)
    if d > MAX_NESTING:
        raise _err(path, f"模式嵌套深度 {d} 超过上限 {MAX_NESTING}")


# ================================================================== v2 模式

# v2 模式语法嵌套深度上限（仅计语法层；递归类型的重复展开由求解器有限状态处理，
# 不受此限）。受 CPython 原生递归栈约束的工程边界。
MAX_V2_PATTERN_DEPTH = 1_000

# 内部 v2 模式节点（除沿用 W/C/T/O 外）：
# ('I', intervals)      整数模式：字面量或闭区间之并（已与类型 range 求交）
# ('B', name, arg)      绑定：arg 匹配的值绑定到 name（对匹配透明）


@dataclass(frozen=True)
class BranchDecl:
    """v2 单个源分支规范化结果。

    :ivar index: 分支下标
    :ivar rows: 顶层 or（及嵌套 or）展开后的候选根模式元组；任一行匹配且
        守卫为真即本分支匹配
    :ivar guard: 守卫内部 AST（所有行共用；变量 intern 为 (index, name)）
    :ivar binds: 绑定名 -> 类型 id（frozenset 项为 (name, tid)）
    """

    index: int
    rows: tuple
    guard: object
    binds: frozenset


def _v2_node_int(iv: tuple) -> tuple:
    return ("I", iv)


def _v2_node_bind(name: str, arg: tuple) -> tuple:
    return ("B", name, arg)


def normalize_branch_v2(table: TypeTable, raw, branch_index: int, path: str) -> BranchDecl:
    """规范化一个 v2 分支。

    ``raw`` 为裸模式（守卫恒真）或 ``{"pattern": 模式, "when": 守卫表达式}``。
    步骤：
      1. 递归收集绑定并校验同一 or 各备选的绑定名称/类型一致；
      2. 校验整数模式（拒绝 bool、端点落在类型闭区间内）；
      3. 编译守卫（只允许引用本分支绑定的整数变量）；
      4. 把 or 展开为候选行（守卫各备选互不可见，与 v1 语义一致）。
    """
    if isinstance(raw, dict) and not isinstance(raw, (list, tuple)) and \
            not isinstance(raw, str) and not isinstance(raw, bool):
        extra = set(raw) - {"pattern", "when"}
        if extra:
            raise _err(path, f"分支字典只允许键 'pattern'/'when'，出现 {sorted(extra)}")
        if "pattern" not in raw:
            raise _err(path, "分支字典必须含 'pattern' 键")
        spec = raw["pattern"]
        guard_raw = raw.get("when", "true")
    else:
        spec = raw
        guard_raw = "true"

    by_id = table_by_id(table)
    counter = [0]
    depth_counter = [0]

    # ------- 第一步：收集绑定（在 or 各备选间强制一致）
    def collect_binds(p, t: int, pth: str):
        """返回该（未展开）模式在*单一匹配路径*上引入的绑定 {name: tid}。"""
        if p == WILD:
            return {}
        if isinstance(p, bool):
            raise _err(pth, "模式位置不接受 bool（整数模式请用 int，True/False 仅属于 Bool）")
        if isinstance(p, int):
            if by_id[t].kind != "integer":
                raise _err(pth, f"整数字面量模式只能用于 integer 类型，位置类型为 {by_id[t].name!r}")
            return {}
        if not isinstance(p, (str, list, tuple)):
            raise _err(pth, f"模式必须是 '_'/str/list/tuple/int，得到 {type(p).__name__}")
        if isinstance(p, list):
            return _collect_product(p, t, pth)
        if isinstance(p, str):
            return {}
        if not p:
            raise _err(pth, "空 tuple 不是合法模式")
        tag = p[0]
        if not isinstance(tag, str):
            raise _err(pth + "[0]", f"tuple 模式首元素必须是字符串，得到 {tag!r}")
        if tag == BIND_TAG:
            if len(p) != 3:
                raise _err(pth, "绑定形态为 ('bind', 名称, 子模式)")
            name = p[1]
            if not isinstance(name, str) or not name:
                raise _err(pth + "[1]", f"绑定名必须是非空字符串，得到 {name!r}")
            if name in RESERVED_LABELS:
                raise _err(pth + "[1]", f"绑定名 {name!r} 为模式语法保留字")
            inner = collect_binds(p[2], t, f"{pth}[2]")
            if name in inner:
                raise _err(pth + "[1]", f"绑定名 {name!r} 在同一匹配路径上重复")
            merged = dict(inner)
            merged[name] = t
            return merged
        if tag == OR_TAG:
            if len(p) < 2:
                raise _err(pth, "or 模式至少需要一个备选")
            alt_maps = []
            for j, alt in enumerate(p[1:]):
                alt_maps.append(collect_binds(alt, t, f"{pth}[{j + 1}]"))
            base = alt_maps[0]
            for j, m in enumerate(alt_maps[1:], start=1):
                if set(m) != set(base):
                    raise _err(
                        f"{pth}[{j + 1}]",
                        f"or 备选绑定名称不一致：备选 0 绑定 {sorted(base)}，"
                        f"备选 {j} 绑定 {sorted(m)}；同一 or 各边必须绑定相同名称",
                    )
                for n_, tt in m.items():
                    if base[n_] != tt:
                        raise _err(
                            f"{pth}[{j + 1}]",
                            f"or 备选绑定 {n_!r} 的类型不一致："
                            f"{by_id[base[n_]].name!r} vs {by_id[tt].name!r}",
                        )
            # 继续向内收集：or 引入的绑定对外可见（每条路径都绑定）
            return dict(base)
        if tag == RANGE_TAG:
            if by_id[t].kind != "integer":
                raise _err(pth, f"区间模式只能用于 integer 类型，位置类型为 {by_id[t].name!r}")
            return {}
        # 构造器
        return _collect_ctor(p, t, pth)

    def _collect_product(p, t: int, pth: str):
        decl = by_id[t]
        if decl.kind != "product":
            raise _err(pth, f"出现积模式 [...]，但类型 {decl.name!r} 不是 product")
        fields = decl.ctors[0].fields
        if len(p) != len(fields):
            raise _err(pth, f"积模式字段数 {len(p)} 与 {len(fields)} 不符")
        merged = {}
        for j, x in enumerate(p):
            for n_, tt in collect_binds(x, fields[j][1], f"{pth}[{j}]").items():
                if n_ in merged:
                    raise _err(f"{pth}[{j}]", f"绑定名 {n_!r} 在同一匹配路径上重复")
                merged[n_] = tt
        return merged

    def _collect_ctor(p, t: int, pth: str):
        decl = by_id[t]
        if decl.kind == "product":
            raise _err(pth, f"类型 {decl.name!r} 是 product，必须用积模式 [...]")
        if decl.kind == "integer":
            raise _err(pth, f"类型 {decl.name!r} 是 integer，此处不能用构造器 {p[0]!r}")
        idx = next((j for j, c in enumerate(decl.ctors) if c.name == p[0]), None)
        if idx is None:
            raise _err(pth, f"构造器 {p[0]!r} 不属于类型 {decl.name!r}")
        ctor = decl.ctors[idx]
        if len(p) - 1 != len(ctor.fields):
            raise _err(pth, f"构造器 {p[0]!r} 需要 {len(ctor.fields)} 个字段模式，得到 {len(p) - 1} 个")
        merged = {}
        for j, x in enumerate(p[1:]):
            for n_, tt in collect_binds(x, ctor.fields[j][1], f"{pth}[{j + 1}]").items():
                if n_ in merged:
                    raise _err(f"{pth}[{j + 1}]", f"绑定名 {n_!r} 在同一匹配路径上重复")
                merged[n_] = tt
        return merged

    binds = collect_binds(spec, table.root, path)

    # 守卫只允许引用绑定的*整数*变量
    int_binds = {n: t for n, t in binds.items() if by_id[t].kind == "integer"}
    guard_ast = compile_guard(guard_raw, int_binds, path, branch_index)

    # ------- 第二步：结构校验 + or 展开
    def go(p, t: int, pth: str, syntactic_depth: int):
        counter[0] += 1
        if counter[0] > MAX_PATTERN_NODES:
            raise _err(path, f"模式节点数超过上限 {MAX_PATTERN_NODES}（路径 {pth}）")
        if syntactic_depth > MAX_V2_PATTERN_DEPTH:
            raise _err(path, f"模式语法嵌套深度超过上限 {MAX_V2_PATTERN_DEPTH}（路径 {pth}）")
        decl = by_id[t]

        if p == WILD:
            return (P_wild(),)
        if isinstance(p, bool):
            raise _err(pth, "模式位置不接受 bool（整数模式请用 int，True/False 仅属于 Bool）")
        if isinstance(p, int):
            if decl.kind != "integer":
                raise _err(pth, f"整数字面量只能用于 integer 类型，位置类型为 {decl.name!r}")
            lo, hi = decl.int_range
            if not (lo <= p <= hi):
                raise _err(pth, f"整数字面量 {p} 超出类型 {decl.name!r} 的闭区间 [{lo}, {hi}]")
            return (_v2_node_int(((p, p),)),)
        if not isinstance(p, (str, list, tuple)):
            raise _err(pth, f"模式必须是 '_'/str/list/tuple/int，得到 {type(p).__name__}")
        if isinstance(p, list):
            if decl.kind != "product":
                raise _err(pth, f"积模式 [...] 只能用于 product，类型 {decl.name!r} 不是")
            fields = decl.ctors[0].fields
            if len(p) != len(fields):
                raise _err(pth, f"积模式字段数 {len(p)} 与 {len(fields)} 不符")
            return _expand([fields[j][1] for j in range(len(p))],
                           [p[j] for j in range(len(p))], pth,
                           lambda child_nodes: P_tuple(tuple(child_nodes)),
                           syntactic_depth)
        if isinstance(p, str):
            return _go_ctor(p, (), decl, t, pth, syntactic_depth)
        if not p:
            raise _err(pth, "空 tuple 不是合法模式")
        tag = p[0]
        if not isinstance(tag, str):
            raise _err(pth + "[0]", "tuple 模式首元素必须是字符串")
        if tag == OR_TAG:
            out = []
            for j, alt in enumerate(p[1:]):
                out.extend(go(alt, t, f"{pth}[{j + 1}]", syntactic_depth + 1))
            return tuple(out)
        if tag == BIND_TAG:
            name = p[1]
            alts = go(p[2], t, f"{pth}[2]", syntactic_depth + 1)
            return tuple(_v2_node_bind(name, a) for a in alts)
        if tag == RANGE_TAG:
            if len(p) != 3:
                raise _err(pth, "区间模式形态为 ('range', 左端点, 右端点)")
            if decl.kind != "integer":
                raise _err(pth, f"区间模式只能用于 integer，位置类型为 {decl.name!r}")
            a, b = p[1], p[2]
            if isinstance(a, bool) or not isinstance(a, int):
                raise _err(pth + "[1]", f"区间左端点必须是 int（拒绝 bool）")
            if isinstance(b, bool) or not isinstance(b, int):
                raise _err(pth + "[2]", f"区间右端点必须是 int（拒绝 bool）")
            if abs(a) > MAX_INT_ENDPOINT_ABS or abs(b) > MAX_INT_ENDPOINT_ABS:
                raise _err(pth, f"区间端点绝对值超过上限 {MAX_INT_ENDPOINT_ABS}")
            if a > b:
                raise _err(pth, f"区间左端 {a} 不得大于右端 {b}")
            lo, hi = decl.int_range
            if b < lo or a > hi:
                raise _err(pth, f"区间 [{a}, {b}] 与类型 {decl.name!r} 的 [{lo}, {hi}] 无交集")
            iv = ((max(a, lo), min(b, hi)),)
            return (_v2_node_int(iv),)
        return _go_ctor(tag, p[1:], decl, t, pth, syntactic_depth)

    def _expand(child_types, raw_children, pth, build, syntactic_depth):
        """把若干子位置的 or 展开候选做笛卡尔积（构造器/积字段同时展开）。"""
        per = []
        for j, (ct, rc) in enumerate(zip(child_types, raw_children)):
            per.append(go(rc, ct, f"{pth}[{j}]", syntactic_depth + 1))
        result = []

        def cross(k, acc):
            if k == len(per):
                result.append(build(tuple(acc)))
                return
            for node in per[k]:
                acc.append(node)
                cross(k + 1, acc)
                acc.pop()

        cross(0, [])
        return tuple(result)

    def _go_ctor(tag, raw_args, decl: TDecl, t: int, pth: str, syntactic_depth):
        if decl.kind == "product":
            raise _err(pth, f"类型 {decl.name!r} 是 product，必须用积模式 [...]")
        if decl.kind == "integer":
            raise _err(pth, f"类型 {decl.name!r} 是 integer，不能用构造器 {tag!r}")
        idx = next((j for j, c in enumerate(decl.ctors) if c.name == tag), None)
        if idx is None:
            names = [c.name for c in decl.ctors]
            raise _err(pth, f"构造器 {tag!r} 不属于 {decl.name!r}；构造器（按输入次序）：{names}")
        ctor = decl.ctors[idx]
        if len(raw_args) != len(ctor.fields):
            raise _err(pth, f"构造器 {tag!r} 需要 {len(ctor.fields)} 个字段模式，得到 {len(raw_args)} 个")
        child_types = [ft for _, ft in ctor.fields]
        return _expand(child_types, list(raw_args), pth,
                       lambda child_nodes: P_con((decl.name, idx), tuple(child_nodes)),
                       syntactic_depth)

    rows = go(spec, table.root, path, 1)
    return BranchDecl(
        index=branch_index,
        rows=tuple(rows),
        guard=guard_ast,
        binds=frozenset(binds.items()),
    )


# ---------------------------------------------------------------- 见证值与字典序

def value_key(by_id: dict, tid: int, value) -> tuple:
    """见证值的字典序键：构造器输入次序编号优先，随后按字段次序拼接。

    同一类型在“同一构造器编号”下各变体字段键等长，可直接拼接比较；
    不同构造器在编号处即分出大小。
    """
    decl = by_id[tid]
    if decl.kind == "bool":
        if not isinstance(value, bool):
            raise AssertionError("内部错误：Bool 见证应为 bool")
        return (1 if value else 0,)
    if decl.kind == "integer":
        if isinstance(value, bool) or not isinstance(value, int):
            raise AssertionError("内部错误：整数见证应为 int")
        return int_key(value)
    if decl.kind == "product":
        fields = decl.ctors[0].fields
        if not isinstance(value, tuple) or len(value) != len(fields):
            raise AssertionError("内部错误：积见证字段数不符")
        key = ()
        for j, (_, ft) in enumerate(fields):
            key += value_key(by_id, ft, value[j])
        return key
    # enum / sum
    if isinstance(value, tuple):
        label = value[0]
    else:
        label = value
    idx = next((j for j, c in enumerate(decl.ctors) if c.name == label), None)
    if idx is None:
        raise AssertionError(f"内部错误：未知构造器标签 {label!r}")
    ctor = decl.ctors[idx]
    if not ctor.fields:
        if isinstance(value, tuple):
            raise AssertionError("内部错误：零元构造器见证应为标签字符串")
        return (idx,)
    if not isinstance(value, tuple) or len(value) != 1 + len(ctor.fields):
        raise AssertionError("内部错误：带字段变体见证字段数不符")
    key = (idx,)
    for j, (_, ft) in enumerate(ctor.fields):
        key += value_key(by_id, ft, value[j + 1])
    return key


def min_value(by_id: dict, tid: int):
    """该类型字典序最小的值。"""
    decl = by_id[tid]
    if decl.kind == "bool":
        return False
    if decl.kind == "product":
        return tuple(min_value(by_id, ft) for _, ft in decl.ctors[0].fields)
    ctor = decl.ctors[0]
    fvs = [min_value(by_id, ft) for _, ft in ctor.fields]
    if not fvs:
        return ctor.name
    return (ctor.name, *fvs)
