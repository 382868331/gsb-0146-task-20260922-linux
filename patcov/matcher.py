"""编译后模式的执行引擎：对一个已校验的合法值直接递归匹配。

与覆盖求解器（solver.py，回答“是否存在未覆盖值”）不同，本模块回答“这个具体值
命中哪个分支、捕获了什么”。它：

- **不展开值域**：每次匹配只沿值本身的结构递归下降，复杂度与值的规模成线性；
- **严格按源顺序**：分支由调用方（:func:`patcov.analyze.run`）按编译顺序逐一尝试，
  命中第一个即返回，不改变源顺序优先级；
- **捕获槽不跨分支共享**：每次匹配传入全新的绑定字典，分支之间互不复用；
- or 各备选在规范化阶段已保证绑定名字集合与类型完全一致，执行时按备选次序尝试，
  命中备选的绑定即为该分支绑定（用独立子字典，失败备选不污染成功备选的捕获）。

绑定节点（``B`` 变量绑定 / ``A`` as 绑定）在覆盖分析中按通配处理；这里在“匹配
任意值”的同时把该位置的完整子树值写入绑定字典。
"""

from __future__ import annotations


class MatchPlan:
    """无状态匹配计划：类型表固定，匹配所需的全部可变状态由调用方给的绑定字典承载。"""

    def __init__(self, by_id: dict):
        self.by_id = by_id

    def match(self, node: tuple, tid: int, value, binds: dict) -> bool:
        """node 是否匹配 tid 类型的值 value；成功时向 binds 写入捕获并返回 True。

        失败时 binds 可能已被部分写入——调用方在分支失败时整体丢弃该字典
        （每次 run 使用全新字典；or 备选使用独立子字典），故失败痕迹不会外泄。
        """
        tag = node[0]

        if tag == "W":
            return True

        if tag == "B":
            # 变量绑定：匹配任意值（覆盖分析中的通配），捕获完整子树。
            binds[node[1]] = value
            return True

        decl = self.by_id[tid]

        if tag == "A":
            # as 绑定：内层匹配成功时，把内层所在位置的完整子树一并绑给名字。
            _, name, inner = node
            if self.match(inner, tid, value, binds):
                binds[name] = value
                return True
            return False

        if tag == "O":
            # 备选按源（输入）次序尝试；用独立子字典，使失败备选不串用捕获槽。
            for alt in node[1]:
                sub: dict = {}
                if self.match(alt, tid, value, sub):
                    # 规范化已保证各备选绑定集合一致且与外层名字不相交。
                    binds.update(sub)
                    return True
            return False

        if tag == "T":
            fields = decl.ctors[0].fields
            if not isinstance(value, tuple) or len(value) != len(fields):
                return False
            for j, (arg, (_, ft)) in enumerate(zip(node[1], fields)):
                if not self.match(arg, ft, value[j], binds):
                    return False
            return True

        # tag == "C"：构造器模式（Bool/enum/带字段 sum 变体统一处理）
        idx = node[1][1]
        ctor = decl.ctors[idx]
        if decl.kind == "bool":
            # False=0, True=1；value 已由 validate_value 保证为 bool
            return (1 if value else 0) == idx
        if not ctor.fields:
            # 零元：值为标签字符串
            return isinstance(value, str) and value == ctor.name
        # 带字段变体：值为 (标签, 字段值...)
        if not isinstance(value, tuple) or len(value) != 1 + len(ctor.fields):
            return False
        if value[0] != ctor.name:
            return False
        for j, (arg, (_, ft)) in enumerate(zip(node[2], ctor.fields)):
            if not self.match(arg, ft, value[j + 1], binds):
                return False
        return True
