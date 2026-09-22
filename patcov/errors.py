"""覆盖见证库的异常类型。

错误语义约定：
- 凡是违反输入契约（结构、类型、次序、上限、NaN/Infinity、bool 冒充整数等），
  一律抛出 :class:`CoverageError`，错误信息带可定位路径，不做静默纠正。
- 算法搜索预算耗尽抛出 :class:`CoverageBudgetExhausted`。它与“已证明不存在
  遗漏值”（``uncovered is None``）是两种完全不同的结果：前者是“未能在预算内
  得出结论”，后者是“已证明穷举”。
"""

from __future__ import annotations


class CoverageError(ValueError):
    """输入违反公开契约。信息中包含可定位的路径。"""


class CoverageBudgetExhausted(RuntimeError):
    """搜索预算（工作量计数）耗尽，结论未知——并非证明穷举。"""

    def __init__(self, used: int, budget: int):
        super().__init__(
            f"覆盖分析搜索预算耗尽：已使用 {used} 个工作量单位，预算为 {budget}；"
            "这表示未能在预算内得出结论，不表示模式已穷举。可调大 budget 重试。"
        )
        self.used = used
        self.budget = budget
