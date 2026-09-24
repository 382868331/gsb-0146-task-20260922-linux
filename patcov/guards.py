"""整数闭区间集合与分支守卫（guard）的编译、折叠。

守卫语言（v2，公开语法，全部为普通 Python 数据）::

    "true" / "false"                         布尔常量
    ["<=", "x", 5]                           已绑定整数 x 与*常数*比较
    ["<", ...] | [">", ...] | [">=", ...] | ["==", ...] | ["!=", ...]
    ["in", "x", [-3, 3]]                     整数闭区间守卫（端点均含）
    ["and", g1, g2]                          布尔组合（and/or 可多个参数）
    ["or", g1, g2, g3]
    ["not", g]

硬约束（输入校验，违例抛 :class:`patcov.CoverageError`）：

- 比较的左侧必须是*已由同分支顶层 or 模式绑定*的整数变量名，右侧必须是
  Python ``int`` 常数（拒绝 ``bool``，因为 ``bool`` 是 ``int`` 子类）；
  不允许变量间比较、不允许算术表达式；
- 闭区间 ``["in", x, [a, b]]`` 要求 a、b 为整数且 a <= b；
- 守卫语法节点数有明确输入上限 :data:`MAX_GUARD_NODES`。

内部 AST（hash-cons 由求解器完成；本模块只产/消元组）::

    ("K", True|False)                    常量
    ("A", var_key, intervals)            原子：var_key ∈ intervals
    ("&", a, b) | ("|", a, b) | ("!", a)

``var_key`` 为 ``(branch_index, name)``：不同分支的同名变量独立。
每个原子是*单个*整数变量上的区间集合（一元谓词），因此守卫中不存在
跨变量耦合；这正是逐列精确分段可判定的基础。

区间集合 ``intervals`` 是排序、两两不相交、相邻已合并的闭区间元组，
每元素 ``(lo, hi)``，``None`` 端点表示 ±∞：

    ()                      空集
    ((None, None),)         整数全集 Z
    ((None, c),)            (-∞, c]
    ((a, None),)            [a, +∞)
    ((a, b),)               [a, b]

整数见证的次序（题面：按绝对值，同绝对值负数先于正数）::

    0, -1, 1, -2, 2, -3, 3, ...
"""

from __future__ import annotations

from .errors import CoverageError

# ------------------------------------------------------------------ 输入上限

MAX_GUARD_NODES = 200
# 单个顶层 or 模式允许声明的绑定数。
MAX_BINDS = 32
# 守卫常数与整数闭区间端点的绝对值上限（工程边界；题面要求至少可达 10^9）。
MAX_GUARD_CONST_ABS = 10**18

IV_ALL = ((None, None),)
IV_EMPTY: tuple = ()

_COMPARATORS = {"<", "<=", ">", ">=", "==", "!="}


def _err(path, msg: str) -> CoverageError:
    return CoverageError(f"{path}: {msg}" if path else msg)


# ------------------------------------------------------------------ 区间集合

def iv_single(a: int, b: int) -> tuple:
    return ((a, b),)


def iv_norm(ivs) -> tuple:
    """排序、去重、合并相交/相邻的闭整数区间。"""
    pts = sorted(((a, b) for a, b in ivs),
                 key=lambda ab: (ab[0] is not None, ab[0]))
    out: list = []
    for lo, hi in pts:
        if not out:
            out.append([lo, hi])
            continue
        llo, lhi = out[-1]
        # 闭整数区间：[.,lhi] 与 [lhi+1,.] 之间无整数空隙，应合并。
        if lo is not None and (lhi is None or lo <= lhi + 1):
            if hi is None or (lhi is not None and hi > lhi):
                out[-1][1] = hi
        else:
            out.append([lo, hi])
    return tuple((a, b) for a, b in out)


def iv_union(a: tuple, b: tuple) -> tuple:
    if not a:
        return b
    if not b:
        return a
    return iv_norm(a + b)


def _clip(lo, hi, bound_lo, bound_hi):
    """[lo,hi] ∩ [bound_lo,bound_hi]，None 为无穷。"""
    nlo = lo if bound_lo is None else (
        bound_lo if lo is None else max(lo, bound_lo))
    nhi = hi if bound_hi is None else (
        bound_hi if hi is None else min(hi, bound_hi))
    if nlo is not None and nhi is not None and nlo > nhi:
        return None
    return (nlo, nhi)


def iv_intersect(a: tuple, b: tuple) -> tuple:
    if not a or not b:
        return IV_EMPTY
    out = []
    i = j = 0
    while i < len(a) and j < len(b):
        part = _clip(*a[i], *b[j])
        if part is not None:
            out.append(part)
        if a[i][1] is None:
            j += 1
        elif b[j][1] is None:
            i += 1
        elif a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return tuple(out)


def iv_subtract(a: tuple, b: tuple) -> tuple:
    """a \\ b，结果仍是排序不相交闭区间。"""
    if not a:
        return IV_EMPTY
    if not b:
        return a
    out: list = []
    j = 0
    for lo, hi in a:
        cur_lo, cur_hi = lo, hi
        k = j
        while k < len(b):
            blo, bhi = b[k]
            if blo is not None and cur_hi is not None and blo > cur_hi:
                break
            if bhi is None or (cur_lo is not None and bhi >= cur_lo):
                # [blo,bhi] 与 [cur_lo,cur_hi] 相交
                if blo is not None and (cur_lo is None or blo > cur_lo):
                    out.append((cur_lo, blo - 1))
                if bhi is None:
                    cur_lo, cur_hi = None, None
                    break
                cur_lo = bhi + 1
                if cur_hi is not None and cur_lo > cur_hi:
                    cur_lo, cur_hi = None, None
                    break
            k += 1
        if cur_lo is not None or cur_hi is not None:
            out.append((cur_lo, cur_hi))
    return iv_norm(out)


def iv_member(iv: tuple, z: int) -> bool:
    for lo, hi in iv:
        if lo is not None and z < lo:
            return False
        if hi is not None and z > hi:
            continue
        return True
    return False


def iv_is_empty(iv: tuple) -> bool:
    return not iv


def segment_min_new_order(lo, hi) -> int:
    """闭整数段 [lo,hi]（None=无穷）中按新序（0,-1,1,-2,2,...）的最小整数。"""
    if lo is None or lo <= 0:
        if hi is None or hi >= 0:
            return 0
        return hi  # 全负段：最接近 0 的是右端点
    return lo  # 全正段：左端点


def iv_min_new_order(iv: tuple):
    """整个区间集合在新序下的最小整数；空集返回 None。"""
    if not iv:
        return None
    best = None
    best_key = None
    for lo, hi in iv:
        z = segment_min_new_order(lo, hi)
        key = (abs(z), 0 if z < 0 else 1)
        if best_key is None or key < best_key:
            best, best_key = z, key
    return best


def int_key(z: int) -> tuple:
    """整数叶子的字典序键：绝对值优先，同绝对值负数先于正数。"""
    return (abs(z), 0 if z < 0 else 1)


def cut_segments(sets):
    """把若干区间集合作为切分约束，返回 Z 的有限个极大段。

    每个输入集合在每个输出段内的成员关系恒定。段表示为
    ``(lo, hi, rep)``；``rep`` 为该段按整数新序的代表值。
    段按 rep 的新序排列（0, -1, 1, ...），无空段。
    """
    points: set = set()
    for iv in sets:
        for lo, hi in iv:
            if lo is not None:
                points.add(lo)
                points.add(lo - 1)
            if hi is not None:
                points.add(hi)
                points.add(hi + 1)
    bounds = sorted(points)
    raw = []
    prev = None  # 上一个有限分界
    for e in bounds:
        # 段 (prev, e) 内的整数：[prev+1, e-1]
        lo = None if prev is None else prev + 1
        hi = e - 1
        if lo is None or lo <= hi:
            raw.append((lo, hi))
        raw.append((e, e))
        prev = e
    raw.append((None if prev is None else prev + 1, None))
    # 合并相邻段（成员关系在调用方按段重算；此处保留全部边界点即可，
    # 相邻段对调用方可能行为相同，调用方按结果状态自行去重）
    seen = {}
    out = []
    for lo, hi in raw:
        key = (lo, hi)
        if key in seen:
            continue
        seen[key] = True
        out.append((lo, hi, segment_min_new_order(lo, hi)))
    out.sort(key=lambda s: int_key(s[2]))
    return out


# ------------------------------------------------------------------ 守卫 AST

K_TRUE = ("K", True)
K_FALSE = ("K", False)


def ast_is_const(ast, value: bool) -> bool:
    return ast == ("K", value)


def _fold(op: str, a, b=None):
    if op == "!":
        if a == K_TRUE:
            return K_FALSE
        if a == K_FALSE:
            return K_TRUE
        if a[0] == "!":
            return a[1]
        return ("!", a)
    if a[0] == "K" and (b is None or b[0] == "K"):
        if op == "&":
            return ("K", a[1] and b[1])
        return ("K", a[1] or b[1])
    if op == "&":
        if a == K_FALSE or b == K_FALSE:
            return K_FALSE
        if a == K_TRUE:
            return b
        if b == K_TRUE:
            return a
        return ("&", a, b)
    # |
    if a == K_TRUE or b == K_TRUE:
        return K_TRUE
    if a == K_FALSE:
        return b
    if b == K_FALSE:
        return a
    return ("|", a, b)


def ast_not(a):
    return _fold("!", a)


def ast_and(a, b):
    return _fold("&", a, b)


def ast_or(a, b):
    return _fold("|", a, b)


def ast_atom(key, iv: tuple):
    if not iv:
        return K_FALSE
    if iv == IV_ALL:
        return K_TRUE
    return ("A", key, iv)


def ast_vars(ast, out: set):
    """收集 AST 中出现的变量 key。"""
    tag = ast[0]
    if tag == "K":
        return
    if tag == "A":
        out.add(ast[1])
    elif tag == "!":
        ast_vars(ast[1], out)
    else:
        ast_vars(ast[1], out)
        ast_vars(ast[2], out)


def ast_subst(ast, key, z: int):
    """把变量 key 的所有原子按具体整数 z 折叠（其他变量保留），常量化简。"""
    tag = ast[0]
    if tag == "K":
        return ast
    if tag == "A":
        if ast[1] != key:
            return ast
        return ("K", iv_member(ast[2], z))
    if tag == "!":
        return ast_not(ast_subst(ast[1], key, z))
    a = ast_subst(ast[1], key, z)
    b = ast_subst(ast[2], key, z)
    return ast_and(a, b) if tag == "&" else ast_or(a, b)


def ast_atom_intervals(ast, key) -> tuple:
    """取 AST 中某变量的（语法层）约束区间并不多；分段时调用方逐原子取。

    返回该变量出现的所有原子区间集合（调用方只需端点，故返回区间列表）。
    """
    res = []

    def go(n):
        t = n[0]
        if t == "K":
            return
        if t == "A":
            if n[1] == key:
                res.append(n[2])
        elif t == "!":
            go(n[1])
        else:
            go(n[1])
            go(n[2])

    go(ast)
    return res


# ------------------------------------------------------------------ 编译/校验

def _check_const(value, path: str):
    if isinstance(value, bool) or not isinstance(value, int):
        raise _err(path, f"守卫常数必须是 int（拒绝 bool），得到 {type(value).__name__}: {value!r}")
    if abs(value) > MAX_GUARD_CONST_ABS:
        raise _err(path, f"守卫常数绝对值超过上限 {MAX_GUARD_CONST_ABS}：{value}")
    return value


def _check_var(name, binds: dict, path: str):
    if not isinstance(name, str) or not name:
        raise _err(path, f"守卫变量必须是非空字符串，得到 {name!r}")
    if name not in binds:
        raise _err(path, f"守卫变量 {name!r} 未由本分支顶层 or 模式绑定；已绑定：{sorted(binds)}")
    return name


def compile_guard(raw, binds: dict, path: str, branch: int):
    """把公开守卫表达式编译为内部 AST。

    :param binds: 顶层 or 绑定名 -> 整数类型 id
    :param branch: 分支下标（变量 intern 为 (branch, name)）
    """
    counter = [0]

    def go(node, pth: str):
        counter[0] += 1
        if counter[0] > MAX_GUARD_NODES:
            raise _err(path, f"守卫节点数超过上限 {MAX_GUARD_NODES}")
        if isinstance(node, str) and node in ("true", "false"):
            return K_TRUE if node == "true" else K_FALSE
        if not isinstance(node, (list, tuple)) or not node:
            raise _err(pth, "守卫必须是非空 list/tuple 表达式或 'true'/'false'")
        op = node[0]
        if not isinstance(op, str):
            raise _err(pth + "[0]", f"守卫操作符必须是字符串，得到 {op!r}")
        if op in ("and", "or"):
            if len(node) < 2:
                raise _err(pth, f"{op!r} 至少需要一个参数")
            acc = go(node[1], f"{pth}[1]")
            for k in range(2, len(node)):
                rhs = go(node[k], f"{pth}[{k}]")
                acc = ast_and(acc, rhs) if op == "and" else ast_or(acc, rhs)
            return acc
        if op == "not":
            if len(node) != 2:
                raise _err(pth, "'not' 恰好需要一个参数")
            return ast_not(go(node[1], pth + "[1]"))
        if op == "in":
            if len(node) != 3:
                raise _err(pth, "'in' 形态为 ['in', 变量, [左端点, 右端点]]")
            name = _check_var(node[1], binds, pth + "[1]")
            rng = node[2]
            if not isinstance(rng, (list, tuple)) or len(rng) != 2:
                raise _err(pth + "[2]", "闭区间必须是 [左端点, 右端点] 二元组")
            a = _check_const(rng[0], pth + "[2][0]")
            b = _check_const(rng[1], pth + "[2][1]")
            if a > b:
                raise _err(pth + "[2]", f"闭区间左端 {a} 不得大于右端 {b}")
            return ("A", (branch, name), ((a, b),))
        if op in _COMPARATORS:
            if len(node) != 3:
                raise _err(pth, f"{op!r} 形态为 [{op!r}, 变量, 常数]")
            name = _check_var(node[1], binds, pth + "[1]")
            c = _check_const(node[2], pth + "[2]")
            key = (branch, name)
            if op == "<=":
                iv = ((None, c),)
            elif op == ">=":
                iv = ((c, None),)
            elif op == "<":
                iv = ((None, c - 1),)
            elif op == ">":
                iv = ((c + 1, None),)
            elif op == "==":
                iv = ((c, c),)
            else:  # !=
                iv = ((None, c - 1), (c + 1, None))
            return ast_atom(key, iv_norm(iv))
        raise _err(pth + "[0]",
                   f"未知守卫操作符 {op!r}；合法：and/or/not/in/<,<=,>,>=,==,!=")

    if raw is None:
        return K_TRUE
    return go(raw, path + ".when")


# ------------------------------------------------------------------ AST 反编译

def _var_token(key) -> str:
    """证书中变量的规范记号：``b{分支下标}:{名字}``。"""
    return f"b{key[0]}:{key[1]}"


def render_ast(ast):
    """把（可能已部分折叠的）内部 AST 反编译为公开守卫形态（纯数据）。

    原子若含多个区间，渲染为若干单区间表达式的 or。用于覆盖证书：
    检查器用自己的独立解析器重新解析，不信任求解器标志。
    """
    tag = ast[0]
    if tag == "K":
        return "true" if ast[1] else "false"
    if tag == "A":
        key, iv = ast[1], ast[2]
        v = _var_token(key)
        parts = []
        for lo, hi in iv:
            if lo is None and hi is None:
                parts.append("true")
            elif lo is None:
                parts.append(["<=", v, hi])
            elif hi is None:
                parts.append([">=", v, lo])
            elif lo == hi:
                parts.append(["==", v, lo])
            else:
                parts.append(["in", v, [lo, hi]])
        if len(parts) == 1:
            return parts[0]
        return ["or", *parts]
    if tag == "!":
        return ["not", render_ast(ast[1])]
    return [tag, render_ast(ast[1]), render_ast(ast[2])]
