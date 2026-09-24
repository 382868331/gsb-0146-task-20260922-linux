"""v2 测试共享构造器（非测试模块，不被 unittest 发现）。"""

from __future__ import annotations


def list_types():
    return {"List": {"kind": "sum", "variants": [
        ("Nil", []),
        ("Cons", [("h", "Bool"), ("t", "List")]),
    ]}}


def exact_length_list(n):
    """恰好 n 个 Cons 的列表模式（Nil 长度 0 单独）。"""
    p = "Nil"
    for _ in range(n):
        p = ("Cons", "_", p)
    return p


def all_false_list(n):
    """n 个 False 后接 Nil 的值（Bool 见证值是 Python bool）。"""
    v = "Nil"
    for _ in range(n):
        v = ("Cons", False, v)
    return v


def all_false_list_pattern(n):
    """n 个 "False" 后接 Nil 的模式（Bool 模式用字符串，与 v1 同语法）。"""
    p = "Nil"
    for _ in range(n):
        p = ("Cons", "False", p)
    return p


TREE_TYPES = {"Tree": {"kind": "sum", "variants": [
    ("Leaf", []),
    ("Node", [("l", "Tree"), ("v", "Bool"), ("r", "Tree")]),
]}}
