"""Coverage witnesses for finite non-recursive algebraic data types.

Public contract (see README for the full specification):

Types (mapping ``name -> definition``; the order of constructors and fields is
taken from the input and defines the lexicographic value order):

* ``bool`` is a built-in type with two constructors, ``False < True``.
* Enum:      ``{"kind": "enum", "constructors": ["A", "B"]}``
* Product:   ``{"kind": "product", "fields": [["x", "color"], ["y", "bool"]]}``
              A product has a single anonymous constructor.
* Labeled sum:``{"kind": "sum", "constructors": [["A", []], ["B", [["f", "t"]]]]}``

Patterns:

* ``{"kind": "wildcard"}``
* ``{"kind": "ctor", "name": "C", "fields": [...]}`` -- ``name`` omitted for
  products; fields omitted/empty for nullary constructors.  Bare Python booleans
  are accepted as shorthand for the two ``bool`` constructors.
* ``{"kind": "or", "patterns": [...]}``

Values (witnesses) are JSON-like:

* bool:        ``True`` / ``False``
* enum:        ``"C"``
* sum:         ``{"C": [field values ...]}`` (``{"C": []}`` when nullary)
* product:     ``[field values ...]``

Ordering (used for every "smallest witness"): constructors in declaration
order, ``False < True``, fields left to right, recursively.

The algorithm is the classical pattern-matrix decomposition (specialisation /
default matrices, after Maranget).  The value space is never enumerated: a
wildcard column branches over its *constructors* and a constructor column
descends into its fields.  Or-patterns are expanded per clause (bounded by the
clause size); no Cartesian product of values is ever built.
"""

from __future__ import annotations

__all__ = ["analyze", "CoverageError", "BudgetExhausted", "LIMITS"]

MAX_CONSTRUCTORS_TOTAL = 128
MAX_NESTING_DEPTH = 12
DEFAULT_BUDGET = 1_000_000

LIMITS = {
    "max_constructors_total": MAX_CONSTRUCTORS_TOTAL,
    "max_nesting_depth": MAX_NESTING_DEPTH,
    "default_budget": DEFAULT_BUDGET,
}

_BOOL = "bool"
_WILD = ("wild",)


class CoverageError(ValueError):
    """The input violates the documented contract.

    The message always carries a path that locates the offending value, e.g.
    ``patterns[2].fields[0]: unknown constructor 'X' for type 'Color'``.
    """


class BudgetExhausted(RuntimeError):
    """The search budget ran out before usefulness could be decided.

    This is distinct from a proven result: an exhaustive match returns
    ``uncovered=None``; this exception means "not proved within the budget".
    """


# ---------------------------------------------------------------------------
# Internal type descriptors
#
#   ("bool",)
#   ("enum",    type_name, (ctor_name, ...))
#   ("product", type_name, ((field_name, ref), ...))
#   ("sum",     type_name, (((ctor_name, ((field_name, ref), ...)), ...)))
#
# refs are strings, resolved lazily through the environment mapping.
# ---------------------------------------------------------------------------


def _ctor_count(desc) -> int:
    kind = desc[0]
    if kind == "bool":
        return 0  # the built-in is not counted against the declared limit
    if kind == "enum" or kind == "sum":
        return len(desc[2])
    return 1  # product


def _ctors(desc):
    kind = desc[0]
    if kind == "bool":
        return ("False", "True")
    if kind == "enum" or kind == "sum":
        return tuple(c[0] if kind == "sum" else c for c in desc[2])
    return (desc[1],)  # product: one anonymous constructor, named after type


def _arity(desc, idx):
    kind = desc[0]
    if kind in ("bool", "enum"):
        return 0
    if kind == "product":
        return len(desc[2])
    return len(desc[2][idx][1])


def _field_refs(desc, idx):
    kind = desc[0]
    if kind == "product":
        return tuple(r for _, r in desc[2])
    if kind == "sum":
        return tuple(r for _, r in desc[2][idx][1])
    return ()


def _wrap(desc, idx, fields):
    kind = desc[0]
    if kind == "bool":
        return idx == 1
    if kind == "enum":
        return desc[2][idx]
    if kind == "product":
        return list(fields)
    return {desc[2][idx][0]: list(fields)}


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _scan_finite(obj, path):
    """Reject every float.  Non-finite floats get the dedicated NaN/Inf error.

    This mathematical subset contains no numeric domain, so any float is
    invalid input; NaN/Infinity are reported explicitly.
    """
    if isinstance(obj, bool):
        return
    if isinstance(obj, float):
        if obj != obj:
            raise CoverageError(f"{path}: NaN is not a valid finite number")
        if obj in (float("inf"), float("-inf")):
            raise CoverageError(f"{path}: Infinity is not a valid finite number")
        raise CoverageError(
            f"{path}: floating point values are not accepted in this subset"
        )
    if isinstance(obj, dict):
        for k in obj:
            if not isinstance(k, str):
                raise CoverageError(f"{path}: object keys must be strings")
            _scan_finite(obj[k], f"{path}[{k!r}]")
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            _scan_finite(v, f"{path}[{i}]")


def _parse_fields(raw, type_name, ctor_name, path):
    if not isinstance(raw, (list, tuple)):
        raise CoverageError(f"{path}: field list must be a list")
    out = []
    for i, item in enumerate(raw):
        fpath = f"{path}[{i}]"
        if isinstance(item, str):
            out.append((f"field{i}", item))
        elif isinstance(item, (list, tuple)) and len(item) == 2 and all(
            isinstance(x, str) for x in item
        ):
            out.append((item[0], item[1]))
        else:
            raise CoverageError(
                f"{fpath}: field must be a type-reference string or "
                f"a [name, type-reference] pair"
            )
    names = [n for n, _ in out]
    if len(set(names)) != len(names):
        raise CoverageError(f"{path}: duplicate field name in {type_name}.{ctor_name}")
    return tuple(out)


def _build_env(raw_types):
    if not isinstance(raw_types, dict):
        raise CoverageError("types: expected a mapping of type name to definition")
    env = {}
    for name, definition in raw_types.items():
        path = f"types[{name!r}]"
        if not isinstance(name, str) or not name:
            raise CoverageError("types: type names must be non-empty strings")
        if name == _BOOL:
            raise CoverageError(f"{path}: {_BOOL!r} is a reserved built-in type name")
        if not isinstance(definition, dict):
            raise CoverageError(f"{path}: definition must be an object")
        kind = definition.get("kind")
        if kind == "enum":
            ctors = definition.get("constructors")
            if not isinstance(ctors, (list, tuple)) or not ctors:
                raise CoverageError(
                    f"{path}.constructors: enum must declare a non-empty list"
                )
            for i, c in enumerate(ctors):
                if not isinstance(c, str):
                    raise CoverageError(f"{path}.constructors[{i}]: expected a name")
            if len(set(ctors)) != len(ctors):
                raise CoverageError(f"{path}.constructors: duplicate constructor name")
            env[name] = ("enum", name, tuple(ctors))
        elif kind == "product":
            fields = _parse_fields(
                definition.get("fields", []), name, name, f"{path}.fields"
            )
            env[name] = ("product", name, fields)
        elif kind == "sum":
            raw_ctors = definition.get("constructors")
            if not isinstance(raw_ctors, (list, tuple)) or not raw_ctors:
                raise CoverageError(
                    f"{path}.constructors: sum must declare a non-empty list"
                )
            ctors = []
            seen = set()
            for i, item in enumerate(raw_ctors):
                cpath = f"{path}.constructors[{i}]"
                if (
                    not isinstance(item, (list, tuple))
                    or len(item) != 2
                    or not isinstance(item[0], str)
                ):
                    raise CoverageError(
                        f"{cpath}: expected a [name, fields] constructor pair"
                    )
                cname, raw_fields = item
                if cname in seen:
                    raise CoverageError(
                        f"{cpath}: duplicate constructor name {cname!r}"
                    )
                seen.add(cname)
                fields = _parse_fields(raw_fields, name, cname, f"{cpath}[1]")
                ctors.append((cname, fields))
            env[name] = ("sum", name, tuple(ctors))
        else:
            raise CoverageError(
                f"{path}.kind: expected one of 'enum', 'product', 'sum'"
            )

    total = sum(_ctor_count(d) for d in env.values())
    if total > MAX_CONSTRUCTORS_TOTAL:
        raise CoverageError(
            f"types: {total} constructors declared, limit is "
            f"{MAX_CONSTRUCTORS_TOTAL}"
        )

    # Reference resolution, cycle rejection and depth computation.
    depths: dict[str, int] = {_BOOL: 1}
    state: dict[str, int] = {}  # 1 = visiting, 2 = done

    def resolve(ref, path):
        if not isinstance(ref, str):
            raise CoverageError(f"{path}: type reference must be a string")
        if ref != _BOOL and ref not in env:
            raise CoverageError(f"{path}: unknown type {ref!r}")

    def depth(ref, path, stack):
        if ref in depths:
            return depths[ref]
        if state.get(ref) == 1:
            cycle = " -> ".join(stack + [ref])
            raise CoverageError(f"types: recursive types are not allowed ({cycle})")
        state[ref] = 1
        desc = env[ref]
        if desc[0] == "enum":
            d = 1
        elif desc[0] == "product":
            children = [(n, r) for n, r in desc[2]]
            d = 1 + max(
                (depth(r, f"types[{ref!r}].fields[{i}]", stack + [ref])
                 for i, (n, r) in enumerate(children)),
                default=0,
            )
        else:
            worst = 0
            for ci, (cname, fields) in enumerate(desc[2]):
                for fi, (_, r) in enumerate(fields):
                    worst = max(
                        worst,
                        depth(
                            r,
                            f"types[{ref!r}].constructors[{ci}][1][{fi}]",
                            stack + [ref],
                        ),
                    )
            d = 1 + worst
        state[ref] = 2
        depths[ref] = d
        if d > MAX_NESTING_DEPTH:
            raise CoverageError(
                f"{path}: nesting depth {d} exceeds limit {MAX_NESTING_DEPTH}"
            )
        return d

    for name, desc in env.items():
        if desc[0] == "product":
            for _, r in desc[2]:
                resolve(r, f"types[{name!r}].fields")
        elif desc[0] == "sum":
            for cname, fields in desc[2]:
                for _, r in fields:
                    resolve(r, f"types[{name!r}].{cname} fields")
        depth(name, f"types[{name!r}]", [])
    return env


# ---------------------------------------------------------------------------
# Pattern parsing.  Normalised patterns:
#   ("wild",)
#   ("ctor", index, (sub-pattern, ...))
#   ("or",   (alternative, ...))
# ---------------------------------------------------------------------------


def _parse_pattern(obj, ref, env, path):
    if isinstance(obj, bool):  # before int; bool shorthand for bool ctors
        if ref != _BOOL:
            raise CoverageError(f"{path}: boolean literal used for type {ref!r}")
        return ("ctor", 1 if obj else 0, ())
    if not isinstance(obj, dict):
        raise CoverageError(
            f"{path}: pattern must be an object (or a boolean for type 'bool')"
        )
    kind = obj.get("kind")
    if kind in ("wildcard", "wild", "_"):
        return _WILD
    if kind == "or":
        alts_raw = obj.get("patterns")
        if not isinstance(alts_raw, (list, tuple)) or not alts_raw:
            raise CoverageError(
                f"{path}.patterns: or-pattern requires a non-empty list"
            )
        alts = tuple(
            _parse_pattern(a, ref, env, f"{path}.patterns[{i}]")
            for i, a in enumerate(alts_raw)
        )
        return ("or", alts)
    if kind != "ctor":
        raise CoverageError(
            f"{path}.kind: expected 'wildcard', 'ctor' or 'or'"
        )

    desc = env[ref] if ref != _BOOL else ("bool",)
    dkind = desc[0]
    name = obj.get("name")
    raw_fields = obj.get("fields", [])
    if not isinstance(raw_fields, (list, tuple)):
        raise CoverageError(f"{path}.fields: expected a list")

    if dkind == "bool":
        if name not in ("True", "False"):
            raise CoverageError(
                f"{path}.name: bool constructor must be 'True' or 'False'"
            )
        if raw_fields:
            raise CoverageError(f"{path}: bool constructors take no fields")
        return ("ctor", 1 if name == "True" else 0, ())

    if dkind == "enum":
        names = desc[2]
        if not isinstance(name, str) or name not in names:
            raise CoverageError(
                f"{path}.name: unknown constructor {name!r} for enum {ref!r}; "
                f"expected one of {', '.join(names)}"
            )
        if raw_fields:
            raise CoverageError(
                f"{path}: enum constructor {name!r} takes no fields"
            )
        return ("ctor", names.index(name), ())

    if dkind == "product":
        if name is not None:
            raise CoverageError(
                f"{path}.name: product type {ref!r} has one anonymous "
                f"constructor; omit 'name'"
            )
        refs = tuple(r for _, r in desc[2])
        if len(raw_fields) != len(refs):
            raise CoverageError(
                f"{path}: product {ref!r} expects {len(refs)} fields, "
                f"got {len(raw_fields)}"
            )
        subs = tuple(
            _parse_pattern(p, r, env, f"{path}.fields[{i}]")
            for i, (p, r) in enumerate(zip(raw_fields, refs))
        )
        return ("ctor", 0, subs)

    # sum
    names = tuple(c[0] for c in desc[2])
    if not isinstance(name, str) or name not in names:
        raise CoverageError(
            f"{path}.name: unknown constructor {name!r} for sum {ref!r}; "
            f"expected one of {', '.join(names)}"
        )
    idx = names.index(name)
    refs = tuple(r for _, r in desc[2][idx][1])
    if len(raw_fields) != len(refs):
        raise CoverageError(
            f"{path}: constructor {ref}.{name} expects {len(refs)} fields, "
            f"got {len(raw_fields)}"
        )
    subs = tuple(
        _parse_pattern(p, r, env, f"{path}.fields[{i}]")
        for i, (p, r) in enumerate(zip(raw_fields, refs))
    )
    return ("ctor", idx, subs)


# ---------------------------------------------------------------------------
# Values and ordering
# ---------------------------------------------------------------------------


def _min_value(pat, ref, env):
    tag = pat[0]
    if tag == "wild":
        return _min_type_value(ref, env)
    if tag == "or":
        best = None
        best_key = None
        for alt in pat[1]:
            v = _min_value(alt, ref, env)
            k = _value_key(v, ref, env)
            if best_key is None or k < best_key:
                best, best_key = v, k
        return best
    desc = env[ref] if ref != _BOOL else ("bool",)
    idx = pat[1]
    refs = _field_refs(desc, idx)
    return _wrap(desc, idx, [_min_value(s, r, env) for s, r in zip(pat[2], refs)])


def _min_type_value(ref, env, memo=None):
    if memo is None:
        memo = {}
    if ref in memo:
        return memo[ref]
    desc = env[ref] if ref != _BOOL else ("bool",)
    idx0 = 0
    refs = _field_refs(desc, idx0)
    v = _wrap(desc, idx0, [_min_type_value(r, env, memo) for r in refs])
    memo[ref] = v
    return v


def _value_key(v, ref, env):
    """Total lexicographic key, comparing only values of the same type."""
    if ref == _BOOL:
        return (1 if v else 0,)
    desc = env[ref]
    kind = desc[0]
    if kind == "enum":
        return (desc[2].index(v),)
    if kind == "product":
        refs = tuple(r for _, r in desc[2])
        return tuple(_value_key(fv, r, env) for fv, r in zip(v, refs))
    # sum: single-key object
    (cname, fvals) = next(iter(v.items()))
    cnames = tuple(c[0] for c in desc[2])
    idx = cnames.index(cname)
    refs = tuple(r for _, r in desc[2][idx][1])
    return (idx, *[_value_key(fv, r, env) for fv, r in zip(fvals, refs)])


def _seq_key(values, refs, env):
    return tuple(_value_key(v, r, env) for v, r in zip(values, refs))


# ---------------------------------------------------------------------------
# Pattern matrix algorithm
# ---------------------------------------------------------------------------


def _specialize(rows, refs, col, desc, idx):
    """Specialise the matrix on column ``col`` for constructor ``idx``.

    Wildcard-headed rows survive with ``arity`` fresh wildcard columns;
    matching constructor rows survive with their sub-pattern columns; rows
    headed by another constructor disappear.  No value is ever enumerated.
    """
    arity = _arity(desc, idx)
    wild_cols = (_WILD,) * arity
    out = []
    for row in rows:
        h = row[col]
        if h[0] == "wild":
            out.append(row[:col] + wild_cols + row[col + 1 :])
        elif h[0] == "ctor" and h[1] == idx:
            out.append(row[:col] + tuple(h[2]) + row[col + 1 :])
    new_refs = refs[:col] + _field_refs(desc, idx) + refs[col + 1 :]
    return out, new_refs


def _dnf(pat):
    """Expand one clause into or-free rows (one column at the root)."""
    tag = pat[0]
    if tag == "wild":
        return [(pat,)]
    if tag == "or":
        out = []
        for alt in pat[1]:
            out.extend(_dnf(alt))
        return out
    field_choices = [_dnf(f) for f in pat[2]]

    def build(i, acc):
        if i == len(field_choices):
            yield (("ctor", pat[1], tuple(acc)),)
            return
        for row in field_choices[i]:
            yield from build(i + 1, acc + [row[0]])

    return list(build(0, []))


def _default_column(rows, q, refs, col):
    """Default matrix: keep wildcard-headed rows, dropping column ``col``.

    Called only when every head at the column (rows and query) is a wildcard,
    so every row is retained.  The dropped query column contributes the
    smallest value of its type to any later witness.
    """
    return (
        [r[:col] + r[col + 1 :] for r in rows],
        q[:col] + q[col + 1 :],
        refs[:col] + refs[col + 1 :],
    )


class _Solver:
    def __init__(self, env, budget):
        self.env = env
        self.budget = budget
        self.nodes = 0

    def _bump(self):
        self.nodes += 1
        if self.nodes > self.budget:
            raise BudgetExhausted(
                f"search budget of {self.budget} matrix nodes exhausted; "
                f"usefulness undecided (distinct from a proven exhaustive match)"
            )

    def useful(self, rows, refs, q):
        """Smallest tuple accepted by the query vector ``q`` but by no row of
        the pattern matrix, or ``None`` when the query is fully covered.

        Rows and query are column vectors aligned with ``refs``; columns are
        type references.  The matrix is refined by constructor specialisation
        (one constructor branch at a time) or by dropping an all-wildcard
        column -- the value space itself is never enumerated.
        """
        self._bump()

        if not refs:
            # Zero columns: the single empty tuple exists; a remaining row
            # covers it.
            return () if not rows else None
        if not rows:
            return tuple(_min_value(p, r, self.env) for p, r in zip(q, refs))

        # Pick a useful column: query ctor first (branch factor 1), then query
        # or, then the first column carrying a constructor in the matrix.
        col = None
        for i, h in enumerate(q):
            if h[0] == "ctor":
                col = i
                break
        if col is None:
            for i, h in enumerate(q):
                if h[0] == "or":
                    col = i
                    break
        if col is None:
            for i in range(len(refs)):
                if any(r[i][0] == "ctor" for r in rows):
                    col = i
                    break
        if col is None:
            # Every head is a wildcard.  Drop the first column; the matrix
            # still constrains all remaining columns.  The dropped column can
            # contribute its smallest value to the witness.
            dropped_ref = refs[0]
            d_rows, d_q, d_refs = _default_column(rows, q, refs, 0)
            w = self.useful(d_rows, d_refs, d_q)
            if w is None:
                return None
            return (_min_type_value(dropped_ref, self.env),) + w

        return self._solve_at(rows, refs, q, col)

    def _solve_at(self, rows, refs, q, col):
        ref = refs[col]
        desc = self.env[ref] if ref != _BOOL else ("bool",)
        head = q[col]

        def lift(sub_w, idx):
            """Reassemble a witness over the specialised columns into one
            aligned with the original columns."""
            arity = _arity(desc, idx)
            value = _wrap(desc, idx, sub_w[col : col + arity])
            return sub_w[:col] + (value,) + sub_w[col + arity :]

        if head[0] == "ctor":
            idx = head[1]
            s_rows, s_refs = _specialize(rows, refs, col, desc, idx)
            s_q = q[:col] + tuple(head[2]) + q[col + 1 :]
            w = self.useful(s_rows, s_refs, s_q)
            return None if w is None else lift(w, idx)

        if head[0] == "or":
            best = None
            best_key = None
            for alt in head[1]:
                w = self.useful(rows, refs, q[:col] + (alt,) + q[col + 1 :])
                if w is not None:
                    k = _seq_key(w, refs, self.env)
                    if best_key is None or k < best_key:
                        best, best_key = w, k
            return best

        # Query wildcard: branch over declared constructors in order.
        best = None
        best_key = None
        for idx in range(len(_ctors(desc))):
            s_rows, s_refs = _specialize(rows, refs, col, desc, idx)
            wild_cols = (_WILD,) * _arity(desc, idx)
            s_q = q[:col] + wild_cols + q[col + 1 :]
            w = self.useful(s_rows, s_refs, s_q)
            if w is not None:
                cand = lift(w, idx)
                k = _seq_key(cand, refs, self.env)
                if best_key is None or k < best_key:
                    best, best_key = cand, k
        return best


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def analyze(types, patterns, root=None, *, budget=DEFAULT_BUDGET):
    """Analyse pattern-match coverage.

    Returns a dict::

        {
          "root": <type reference>,
          "branches": [
            {"index": 0, "useful": bool, "redundant": bool,
             "witness": <smallest newly covered value, or null>},
            ...
          ],
          "exhaustive": bool,
          "uncovered": <smallest uncovered value, or null>,
        }

    Branches are judged in source order against the previously *useful*
    branches; a redundant clause covers nothing new and is not added to the
    context.  Witnesses are lexicographically smallest in the input's
    constructor/field order.

    ``budget`` must be a positive ``int`` (booleans rejected); when more than
    ``budget`` matrix nodes are explored, :class:`BudgetExhausted` is raised.
    That exception means "undecided" and never means "exhaustive".
    """
    _scan_finite(types, "types")
    _scan_finite(patterns, "patterns")
    if isinstance(budget, bool) or not isinstance(budget, int):
        raise CoverageError("budget: must be an integer (booleans are rejected)")
    if budget < 1:
        raise CoverageError("budget: must be a positive integer")

    env = _build_env(types)

    if root is None:
        if env:
            if len(env) == 1:
                root = next(iter(env))
            else:
                raise CoverageError(
                    "root: cannot infer the matched type; pass root=<type name>"
                )
        else:
            root = _BOOL
    if not isinstance(root, str):
        raise CoverageError("root: type reference must be a string")
    if root != _BOOL and root not in env:
        raise CoverageError(f"root: unknown type {root!r}")

    if not isinstance(patterns, (list, tuple)):
        raise CoverageError("patterns: expected a list of branch patterns")

    parsed = [
        _parse_pattern(p, root, env, f"patterns[{i}]")
        for i, p in enumerate(patterns)
    ]

    solver = _Solver(env, budget)
    refs = (root,)
    rows: list[tuple] = []
    branches = []
    for i, pat in enumerate(parsed):
        w = solver.useful(rows, refs, (pat,))
        useful = w is not None
        branches.append(
            {
                "index": i,
                "useful": useful,
                "redundant": not useful,
                "witness": None if w is None else w[0],
            }
        )
        if useful:
            rows.extend(_dnf(pat))

    tail = solver.useful(rows, refs, (_WILD,))
    uncovered = None if tail is None else tail[0]
    return {
        "root": root,
        "branches": branches,
        "exhaustive": uncovered is None,
        "uncovered": uncovered,
    }
