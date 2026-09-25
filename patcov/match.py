"""编译/执行入口：把模式序列编译为 :class:`Matcher`，对具体值求首个命中分支与绑定。

- :func:`compile_matcher` 复用与 :func:`patcov.analyze` 完全相同的类型表与
  模式规范化（含绑定规则校验），一次编译、可重复执行；
- :meth:`Matcher.match` 先校验值是否属于根类型（不合法即 :class:`CoverageError`，
  路径可定位），再按**源顺序**逐分支直接递归匹配，返回首个命中分支下标与
  该分支产生的绑定字典；无命中显式返回 ``None``；
- 每个分支的尝试使用全新的绑定字典，or 的每个备选亦然——失败尝试的半成品
  绑定随字典一起丢弃，不存在跨分支/跨调用串用的捕获槽；模式节点为不可变
  tuple，可在分支间安全共享；
- 匹配是模式规模的线性递归，不展开值空间、不建查找表。
"""

from __future__ import annotations

from dataclasses import dataclass

from .errors import CoverageError
from .model import build_type_table, normalize_patterns, table_by_id


@dataclass(frozen=True)
class MatchResult:
    """一次命中的结果。

    :ivar branch: 命中分支的下标（源顺序，从 0 开始）
    :ivar bindings: 该分支产生的绑定字典（名 -> 值，按绑定发生顺序插入；
        as 绑定在其子模式命中之后记录，故嵌套名先于外层 as 名）
    """

    branch: int
    bindings: dict


class Matcher:
    """已编译的模式序列。不可变；:meth:`match` 可重复调用、互不干扰。"""

    def __init__(self, table, norm_patterns):
        self._table = table
        self._by_id = table_by_id(table)
        self._patterns = tuple(norm_patterns)

    @property
    def branch_count(self) -> int:
        return len(self._patterns)

    def match(self, value):
        """返回首个命中分支的 :class:`MatchResult`；无命中返回 ``None``。

        :raises CoverageError: ``value`` 不是根类型的合法值（错误信息含路径）
        """
        validate_value(self._by_id, self._table.root, value, "value")
        root = self._table.root
        for i, node in enumerate(self._patterns):
            out: dict = {}
            if self._hit(node, value, root, out):
                return MatchResult(branch=i, bindings=out)
        return None

    def _hit(self, node, value, tid: int, out: dict) -> bool:
        """node 是否匹配 value；命中时把绑定并入 out。value 已通过类型校验。"""
        tag = node[0]
        if tag == "W":
            return True
        if tag == "V":
            out[node[1]] = value
            return True
        if tag == "A":
            if self._hit(node[2], value, tid, out):
                out[node[1]] = value
                return True
            return False
        decl = self._by_id[tid]
        if tag == "T":
            fields = decl.ctors[0].fields
            for j, sub in enumerate(node[1]):
                if not self._hit(sub, value[j], fields[j][1], out):
                    return False
            return True
        if tag == "O":
            for alt in node[1]:
                tmp: dict = {}
                if self._hit(alt, value, tid, tmp):
                    out.update(tmp)
                    return True
            return False
        # 'C'：构造器
        idx = node[1][1]
        ctor = decl.ctors[idx]
        if decl.kind == "bool":
            return value is (idx == 1)
        if decl.kind == "enum":
            return value == ctor.name
        # sum
        if not ctor.fields:
            return value == ctor.name
        if not (isinstance(value, tuple) and value and value[0] == ctor.name):
            return False
        for j, sub in enumerate(node[2]):
            if not self._hit(sub, value[j + 1], ctor.fields[j][1], out):
                return False
        return True


def validate_value(by_id: dict, tid: int, value, path: str) -> None:
    """校验 value 是类型 tid 的合法值（形态与见证值相同）；否则 CoverageError。"""
    decl = by_id[tid]
    kind = decl.kind
    if kind == "bool":
        if type(value) is not bool:
            raise CoverageError(
                f"{path}: 类型 'Bool' 的值必须是 bool，得到 {type(value).__name__}"
            )
        return
    if kind == "product":
        fields = decl.ctors[0].fields
        if not isinstance(value, tuple) or len(value) != len(fields):
            raise CoverageError(
                f"{path}: 类型 {decl.name!r} 的值必须是 {len(fields)} 元 tuple，"
                f"得到 {value!r}"
            )
        for j, (_, ft) in enumerate(fields):
            validate_value(by_id, ft, value[j], f"{path}[{j}]")
        return
    if kind == "enum":
        labels = {c.name for c in decl.ctors}
        if not isinstance(value, str) or value not in labels:
            raise CoverageError(
                f"{path}: 类型 {decl.name!r} 的值必须是其构造器标签之一 "
                f"{sorted(labels)}，得到 {value!r}"
            )
        return
    # sum
    if isinstance(value, str):
        label, fvals = value, ()
    elif isinstance(value, tuple) and value and isinstance(value[0], str):
        label, fvals = value[0], value[1:]
    else:
        raise CoverageError(
            f"{path}: 类型 {decl.name!r} 的值必须是变体标签字符串或 "
            f"(标签, 字段值...) tuple，得到 {value!r}"
        )
    ctor = next((c for c in decl.ctors if c.name == label), None)
    if ctor is None:
        names = [c.name for c in decl.ctors]
        raise CoverageError(
            f"{path}: 标签 {label!r} 不属于类型 {decl.name!r}；该类型变体（按输入次序）：{names}"
        )
    if len(fvals) != len(ctor.fields):
        raise CoverageError(
            f"{path}: 变体 {label!r} 需要 {len(ctor.fields)} 个字段值，得到 {len(fvals)} 个"
        )
    for j, (_, ft) in enumerate(ctor.fields):
        validate_value(by_id, ft, fvals[j], f"{path}[{j + 1}]")


def compile_matcher(types, patterns) -> Matcher:
    """编译模式序列为可重复执行的 :class:`Matcher`。

    类型声明、模式形态与绑定规则的校验和 :func:`patcov.analyze` 完全一致；
    非法输入抛 :class:`CoverageError`（信息含可定位路径），不产生部分结果。
    """
    table = build_type_table(types)
    norm = normalize_patterns(table, patterns)
    return Matcher(table, norm)


__all__ = [
    "compile_matcher",
    "Matcher",
    "MatchResult",
    "validate_value",
]
