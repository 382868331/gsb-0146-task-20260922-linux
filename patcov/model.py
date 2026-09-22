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
BOOL_NAME = "Bool"
WILD = "_"
OR_TAG = "or"
RESERVED_LABELS = (WILD, OR_TAG)


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
    kind: str              # 'bool' | 'enum' | 'sum' | 'product'
    ctors: tuple           # bool/enum/sum 的构造器；product 为单个匿名 Ctor
    field_names: tuple     # product 的字段名；其余为 ()


@dataclass(frozen=True)
class TypeTable:
    decls: dict            # name -> TDecl
    ids: dict              # name -> int
    root: int
    order: tuple           # 用户声明顺序（不含内置 Bool）


def _err(path, msg: str) -> CoverageError:
    return CoverageError(f"{path}: {msg}" if path else msg)


def _check_name(path, value, what: str) -> str:
    if not isinstance(value, str) or not value:
        raise _err(path, f"{what}必须是非空字符串，得到 {type(value).__name__!s}")
    return value


def table_by_id(table: TypeTable) -> dict:
    return {table.ids[n]: table.decls[n] for n in list(table.order) + [BOOL_NAME]}


def build_type_table(types) -> TypeTable:
    """校验并构造类型表。第一项声明为根（scrutinee）类型。

    首项允许且仅允许是 ``{"Bool": {"kind": "bool"}}``，表示根直接取内置 Bool
    （这不是重声明：内置 Bool 仍由库提供，且该写法只能出现在首项）。
    """
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
        if not isinstance(spec.get("kind"), str) or spec.get("kind") not in {"bool", "enum", "product", "sum"}:
            raise _err(
                f"types[{name!r}]",
                f"'kind' 必须是 'bool'|'enum'|'product'|'sum' 之一，得到 {spec.get('kind')!r}",
            )
        if spec["kind"] == "bool":
            raise _err(f"types[{name!r}]", f"用户类型不得声明 kind='bool'；请直接引用内置 {BOOL_NAME}，或改用 enum")
        extra = set(spec) - {"kind", "constructors", "fields", "variants"}
        if extra:
            raise _err(f"types[{name!r}]", f"出现未知键 {sorted(extra)}；合法键为 kind/constructors/fields/variants")
        ids[name] = idx

    total_ctors = 0
    for name in order:
        spec = types[name]
        p = f"types[{name!r}]"
        kind = spec["kind"]

        if kind == "enum":
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
    by_id = table_by_id(TypeTable(decls, ids, provisional_root, order))

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
        children = [ft for c in d.ctors for _, ft in c.fields]
        if children:
            worst = 1 + max(depth(ch) for ch in children)
        visiting.discard(tid)
        done[tid] = worst
        if worst > MAX_NESTING:
            raise _err(f"types[{d.name!r}]", f"类型嵌套深度 {worst} 超过上限 {MAX_NESTING}")
        return worst

    for name in order:
        depth(ids[name])

    root = ids[BOOL_NAME] if bool_root else ids[order[0]]
    return TypeTable(decls=decls, ids=ids, root=root, order=order)


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
