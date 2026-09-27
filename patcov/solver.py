"""核心算法：共享有序决策 DAG 上的覆盖与有用性。

被覆盖的值集合是若干“模式矩形”之并（每行模式 = 各位置匹配集的积）。
若把行合并成单列覆盖会丢失行内跨字段关联（例如行 (True, _) 与 (_, True)
并不能覆盖 (False, False)），因此本模块用一棵*有序决策图*精确表示该并集：

- 变量次序为值树的前序：先判当前类型的构造器（输入次序），再依次判其各字段。
- 决策节点（以整数 id 引用并 hash-cons）：
    id 0 = TOP  该位置序列的所有值均已覆盖（全集，全局共享）
    id 1 = BOT  空集（全局共享）
    分支节点：在某类型上按构造器分支，children[j] 为构造器 j 的子节点；子节点
      面对的“位置序列”是 构造器 j 的字段类型 后接 当前剩余兄弟类型。
- 类型序列、模式序列均用 *interned 持久链表*（整数 id，O(1) cons/取尾），
  见证以后序 list 沿同一路径透传，宽积逐层下降为严格线性，无二次方复制。
- 通配模式不直接等于 TOP：它覆盖当前类型的全部构造器，但后续兄弟位置仍受
  同行其余模式约束，故按构造器惰性展开；从不枚举任何值域笛卡尔积。
  n 个 Bool 的积只产生 O(n) 规模的处理，而不是 2^n。
- 全部节点共享记忆化；预算耗尽抛 CoverageBudgetExhausted，与“已证明无遗漏
  值”（返回 None）严格区分。
"""

from __future__ import annotations

from .errors import CoverageBudgetExhausted
from .model import P_wild, value_key

TOP_ID = 0
BOT_ID = 1
EMPTY = -1  # 空链表 id


class DecisionSolver:
    def __init__(self, table, by_id, budget: int):
        self.table = table
        self.by_id = by_id
        self.budget = budget
        self.used = 0

        # 类型序列持久链表：head(tid) + tail(id)
        self._sig_map: dict[tuple, int] = {}
        self._sig_head: list[int] = []
        self._sig_tail: list[int] = []

        # 模式序列持久链表：head(模式小整数 id) + tail(id)；并记录后缀是否全通配
        self._q_map: dict[tuple, int] = {}
        self._q_head: list[int] = []
        self._q_tail: list[int] = []
        self._q_allwild: list[bool] = []
        self._pat_id: dict[int, int] = {}   # id(模式节点) -> 模式小整数 id
        self._pat_node: list = []
        self._wild_pid = self._intern_pat(P_wild())

        # 决策节点
        self._node_tid: list[int] = [-1, -1]
        self._node_sig: list[int] = [EMPTY, EMPTY]
        self._node_children: list = [(), ()]
        self._branch_map: dict[tuple, int] = {}

        # 记忆化（键均为整数）
        self._add_memo: dict[tuple, int] = {}
        self._use_memo: dict[tuple, object] = {}
        self._free_memo: dict[tuple, int] = {}
        self._match_memo: dict[tuple, int] = {}
        self._NONE = object()

        # 见证持久 cons 链表：head 值 + tail id；空为 EMPTY(-1)。
        # 不可变、可安全共享给多个记忆化调用方；cons O(1)。
        self._v_map: dict[tuple, int] = {}
        self._v_head: list = []
        self._v_tail: list[int] = []

        # (字段类型 tuple, q 尾) -> 全通配序列
        self._wild_q_cache: dict[tuple, int] = {}

        # 基数（精确计数，Python 大整数；永不逐值展开）
        self._type_card: dict[int, int] = {}
        self._sig_card: dict[int, int] = {EMPTY: 1}  # EMPTY 序列只含单位元 ()

    # --------------------------------------------------------------- 基础设施
    def _tick(self):
        self.used += 1
        if self.used > self.budget:
            raise CoverageBudgetExhausted(self.used, self.budget)

    def _intern_pat(self, pat) -> int:
        key = id(pat)
        pid = self._pat_id.get(key)
        if pid is None:
            pid = len(self._pat_node)
            self._pat_id[key] = pid
            self._pat_node.append(pat)
        return pid

    def sig_cons(self, tid: int, tail: int) -> int:
        key = (tid, tail)
        sid = self._sig_map.get(key)
        if sid is None:
            sid = len(self._sig_head)
            self._sig_map[key] = sid
            self._sig_head.append(tid)
            self._sig_tail.append(tail)
        return sid

    def q_cons_pid(self, pid: int, tail: int) -> int:
        key = (pid, tail)
        qid = self._q_map.get(key)
        if qid is None:
            qid = len(self._q_head)
            tail_wild = True if tail == EMPTY else self._q_allwild[tail]
            self._q_map[key] = qid
            self._q_head.append(pid)
            self._q_tail.append(tail)
            self._q_allwild.append(pid == self._wild_pid and tail_wild)
        return qid

    def q_cons(self, pat, tail: int) -> int:
        return self.q_cons_pid(self._intern_pat(pat), tail)

    @staticmethod
    def _field_types(ctor) -> tuple:
        return tuple(ft for _, ft in ctor.fields)

    def child_sig(self, ctor, rest_sig: int) -> int:
        """构造器字段类型序列 + rest（自右向左 cons，O(arity)）。"""
        sid = rest_sig
        for _, ft in reversed(ctor.fields):
            sid = self.sig_cons(ft, sid)
        return sid

    def child_q(self, args: tuple, qtail: int) -> int:
        """参数模式序列 + 尾部模式（自右向左 cons，O(arity)）。"""
        qid = qtail
        for a in reversed(args):
            qid = self.q_cons(a, qid)
        return qid

    def wilds(self, types: tuple, tail: int) -> int:
        """在 tail 前接 types 对应的全通配模式序列。"""
        key = (types, tail)
        qid = self._wild_q_cache.get(key)
        if qid is None:
            qid = tail
            for _ in reversed(types):
                qid = self.q_cons_pid(self._wild_pid, qid)
            self._wild_q_cache[key] = qid
        return qid

    def _branch(self, tid: int, sig_id: int, children: tuple) -> int:
        if all(ch == TOP_ID for ch in children):
            return TOP_ID
        if all(ch == BOT_ID for ch in children):
            return BOT_ID
        key = (tid, sig_id, children)
        nid = self._branch_map.get(key)
        if nid is None:
            nid = len(self._node_tid)
            self._branch_map[key] = nid
            self._node_tid.append(tid)
            self._node_sig.append(sig_id)
            self._node_children.append(children)
        return nid

    @staticmethod
    def _assemble(decl, idx, field_values: tuple):
        """把构造器编号与字段值（前序）组装为公开见证形态。"""
        if decl.kind == "bool":
            return idx == 1  # False=0, True=1
        name = decl.ctors[idx].name
        if decl.kind == "product":
            return tuple(field_values)
        if not field_values:
            return name
        return (name, *field_values)

    @staticmethod
    def _wild_args(arity: int) -> tuple:
        return (P_wild(),) * arity if arity else ()

    def v_cons(self, value, tail: int) -> int:
        key = (value, tail)
        vid = self._v_map.get(key)
        if vid is None:
            vid = len(self._v_head)
            self._v_map[key] = vid
            self._v_head.append(value)
            self._v_tail.append(tail)
        return vid

    def v_split(self, vid: int, n: int):
        """拆出见证链表前 n 个值（前序 tuple）与第 n 个之后的尾链表 id。"""
        out = []
        for _ in range(n):
            out.append(self._v_head[vid])
            vid = self._v_tail[vid]
        return tuple(out), vid

    # --------------------------------------------------------------- 精确计数
    def cardinality_type(self, tid: int) -> int:
        """单个类型的值总数：各构造器（字段值数之积）之和。"""
        n = self._type_card.get(tid)
        if n is not None:
            return n
        decl = self.by_id[tid]
        total = 0
        for ctor in decl.ctors:
            per = 1
            for _, ft in ctor.fields:
                per *= self.cardinality_type(ft)
            total += per
        self._type_card[tid] = total
        return total

    def cardinality_sig(self, sig_id: int) -> int:
        """类型序列的值元组总数：各位置类型基数之积。"""
        n = self._sig_card.get(sig_id)
        if n is not None:
            return n
        n = self.cardinality_type(self._sig_head[sig_id])
        n *= self.cardinality_sig(self._sig_tail[sig_id])
        self._sig_card[sig_id] = n
        return n

    def count_free(self, node: int, sig_id: int) -> int:
        """sig 序列中*未被 node 覆盖*的值元组数（精确，不展开值域）。

        分支节点按构造器分解：未覆盖数 = 各构造器子节点未覆盖数之和；
        构造器字段与兄弟位置在同一子序列上做积，全部为大整数符号运算。
        """
        if node == TOP_ID:
            return 0
        if sig_id == EMPTY:
            return 1 if node == BOT_ID else 0
        if node == BOT_ID:
            return self.cardinality_sig(sig_id)
        key = (node, sig_id)
        cached = self._free_memo.get(key)
        if cached is not None:
            return cached
        self._tick()
        tid = self._sig_head[sig_id]
        rest_sig = self._sig_tail[sig_id]
        ctors = self.by_id[tid].ctors
        total = 0
        for j, child in enumerate(self._node_children[node]):
            csig = self.child_sig(ctors[j], rest_sig)
            total += self.count_free(child, csig)
        self._free_memo[key] = total
        return total

    def count_match(self, node: int, sig_id: int, qid: int) -> int:
        """sig 序列中*被 qid 匹配且未被 node 覆盖*的值元组数（精确）。

        与 useful 同构但做计数：构造器分支求和、字段序列做积；or 不做
        容斥展开，而是顺序把已处理备选并入 node，新增量天然互不重叠。
        """
        if qid == EMPTY:
            return 1 if node == BOT_ID else 0
        if node == TOP_ID:
            return 0
        if self._q_allwild[qid]:
            return self.count_free(node, sig_id)
        key = (node, sig_id, qid)
        cached = self._match_memo.get(key)
        if cached is not None:
            return cached
        self._tick()

        tid = self._sig_head[sig_id]
        rest_sig = self._sig_tail[sig_id]
        decl = self.by_id[tid]
        ctors = decl.ctors
        is_bot = node == BOT_ID
        old_children = () if is_bot else self._node_children[node]
        p = self._pat_node[self._q_head[qid]]
        qtail = self._q_tail[qid]
        ptag = p[0]

        if ptag == "W":
            total = 0
            for j, ctor in enumerate(ctors):
                csig = self.child_sig(ctor, rest_sig)
                cq = self.wilds(self._field_types(ctor), qtail)
                old_ch = BOT_ID if is_bot else old_children[j]
                # 通配矩形 = 全部构造器的字段全通配之并。
                # 若展开后子序列为全通配，直接取该子节点的未覆盖数
                #（count_free 对 BOT 立即返回基数，不会形成互递归环）。
                if self._q_allwild[cq]:
                    total += self.count_free(old_ch, csig)
                else:
                    total += self.count_match(old_ch, csig, cq)
        elif ptag in ("C", "T"):
            idx = 0 if ptag == "T" else p[1][1]
            args = p[1] if ptag == "T" else p[2]
            ctor = ctors[idx]
            csig = self.child_sig(ctor, rest_sig)
            cq = self.child_q(args, qtail)
            old_ch = BOT_ID if is_bot else old_children[idx]
            total = self.count_match(old_ch, csig, cq)
        else:  # or：顺序并集，逐备选只计入“新增加”的量，杜绝重复计数
            total = 0
            unode = node
            for alt in p[1]:
                aq = self.q_cons(alt, qtail)
                total += self.count_match(unode, sig_id, aq)
                unode = self.add(unode, sig_id, aq)

        self._match_memo[key] = total
        return total

    # --------------------------------------------------------------- 加入覆盖
    def add(self, node: int, sig_id: int, qid: int) -> int:
        """返回 node ∪ ⟦qid 模式序列⟧。"""
        if qid == EMPTY:
            return TOP_ID  # 空序列只有 ()，模式行匹配它即覆盖
        if node == TOP_ID:
            return TOP_ID
        if self._q_allwild[qid]:
            return TOP_ID  # 剩余位置全通配：该矩形即全集，无需逐构造器展开
        key = (node, sig_id, qid)
        cached = self._add_memo.get(key)
        if cached is not None:
            return cached
        self._tick()

        tid = self._sig_head[sig_id]
        rest_sig = self._sig_tail[sig_id]
        decl = self.by_id[tid]
        ctors = decl.ctors
        is_bot = node == BOT_ID
        old_children = () if is_bot else self._node_children[node]
        p = self._pat_node[self._q_head[qid]]
        qtail = self._q_tail[qid]
        ptag = p[0]

        if ptag == "W":
            new_children = []
            for j, ctor in enumerate(ctors):
                csig = self.child_sig(ctor, rest_sig)
                cq = self.wilds(self._field_types(ctor), qtail)
                old_ch = BOT_ID if is_bot else old_children[j]
                new_children.append(self.add(old_ch, csig, cq))
            result = self._branch(tid, sig_id, tuple(new_children))

        elif ptag in ("C", "T"):
            idx = 0 if ptag == "T" else p[1][1]
            args = p[1] if ptag == "T" else p[2]
            ctor = ctors[idx]
            csig = self.child_sig(ctor, rest_sig)
            cq = self.child_q(args, qtail)
            old_ch = BOT_ID if is_bot else old_children[idx]
            new_ch = self.add(old_ch, csig, cq)
            built = [BOT_ID] * len(ctors) if is_bot else list(old_children)
            built[idx] = new_ch
            result = self._branch(tid, sig_id, tuple(built))

        else:  # or：备选求并（重复备选天然幂等）
            result = node
            for alt in p[1]:
                result = self.add(result, sig_id, self.q_cons(alt, qtail))
                if result == TOP_ID:
                    break

        self._add_memo[key] = result
        return result

    # --------------------------------------------------------------- 有用性
    def useful(self, node: int, sig_id: int, qid: int):
        """返回*前序*见证 cons 链表 id：head 为当前（序列首位）位置的值，
        随后依次为字段与兄弟位置；不存在未覆盖值时返回 None。
        链表不可变，可被多个记忆化调用方安全共享；cons O(1)。"""
        if qid == EMPTY:
            return EMPTY if node == BOT_ID else None
        if node == TOP_ID:
            return None
        key = (node, sig_id, qid)
        cached = self._use_memo.get(key, self._NONE)
        if cached is not self._NONE:
            return cached
        self._tick()
        ans = self._useful(node, sig_id, qid)
        self._use_memo[key] = ans
        return ans

    def _useful(self, node: int, sig_id: int, qid: int):
        tid = self._sig_head[sig_id]
        rest_sig = self._sig_tail[sig_id]
        decl = self.by_id[tid]
        ctors = decl.ctors
        is_bot = node == BOT_ID
        old_children = () if is_bot else self._node_children[node]
        p = self._pat_node[self._q_head[qid]]
        qtail = self._q_tail[qid]
        ptag = p[0]

        def try_ctor(j: int, args: tuple):
            ctor = ctors[j]
            csig = self.child_sig(ctor, rest_sig)
            cq = self.child_q(args, qtail)
            old_ch = BOT_ID if is_bot else old_children[j]
            return self.useful(old_ch, csig, cq)

        if ptag in ("W", "T", "C"):
            if ptag == "W":
                order = range(len(ctors))
            elif ptag == "T":
                order = (0,)
            else:
                order = (p[1][1],)
            for j in order:
                ctor = ctors[j]
                arity = len(ctor.fields)
                if ptag == "W":
                    args = self._wild_args(arity)
                elif ptag == "T":
                    args = p[1]
                else:
                    args = p[2]
                r = try_ctor(j, args)
                if r is not None:
                    # r 前 arity 个为该构造器字段值（前序），其后为兄弟位置值
                    field_values, tail = self.v_split(r, arity)
                    v0 = self._assemble(decl, j, field_values)
                    return self.v_cons(v0, tail)
            return None

        # or：各备选互不可见（分支内备选之间不判冗余）；取字典序最小见证
        best = None
        best_key = None
        for alt in p[1]:
            r = self.useful(node, sig_id, self.q_cons(alt, qtail))
            if r is not None:
                rk = self._prelist_key(sig_id, r)
                if best_key is None or rk < best_key:
                    best, best_key = r, rk
        return best

    def _prelist_key(self, sig_id: int, vid: int) -> tuple:
        """前序类型序列与前序见证链表对齐，拼字典序键。"""
        key = ()
        s = sig_id
        while s != EMPTY:
            key += value_key(self.by_id, self._sig_head[s], self._v_head[vid])
            vid = self._v_tail[vid]
            s = self._sig_tail[s]
        return key

    # --------------------------------------------------------------- 入口辅助
    def root_sig(self, tid: int) -> int:
        return self.sig_cons(tid, EMPTY)

    def root_q(self, pat) -> int:
        return self.q_cons(pat, EMPTY)

    def root_value(self, vid: int):
        """根位置（单位置序列）的见证值。"""
        return self._v_head[vid]

    def smallest_uncovered(self, node: int, sig_id: int) -> int:
        """node 下字典序最小未覆盖见证链表 id；全集（node==TOP）时返回 None。"""
        types = []
        s = sig_id
        while s != EMPTY:
            types.append(self._sig_head[s])
            s = self._sig_tail[s]
        qid = self.wilds(tuple(types), EMPTY)
        return self.useful(node, sig_id, qid)
