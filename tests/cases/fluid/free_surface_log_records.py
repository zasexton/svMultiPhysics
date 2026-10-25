"""Read the solver's diagnostic log lines in their old and their record format.

The free-surface services' log lines move to FE log records
(agent-status/refactor/logrecord-cleanup-proposal.md, approved 2026-10-24):

    old:  [svMultiPhysics::Application] Wet volume diagnostic step=0 time=0.0000000000000000e+00 ...
    new:  [hh:mm:ss.mmm] [R0] [INFO] ApplicationDriver: wet volume diagnostic=wet_volume step=0 time=0 ...

The map free_surface_log_record_map.json (next to this file) states, for
every converted line, how it reads before and after.  This module reads a
line in either format and gives the same record: the diagnostic word, the
owner and message of the new format, the fields under their new names, and
the child lines (the repeated groups of the old format).  A reader that
selects lines through line_diagnostic() and reads them through
line_fields() or records() works before and after a line converts.

Values are text, as printed; value() parses one.  values_equal() is the
comparison of the conversion gate: a conversion moves no value, but an old
line may print a value rounded (6 significant digits, 6 decimals) where the
record prints the exact double.
"""
from __future__ import annotations

import functools
import json
import math
import re
from pathlib import Path
from typing import Iterable

MAP_PATH = Path(__file__).resolve().with_name("free_surface_log_record_map.json")

_TS = re.compile(r"^\[\d\d:\d\d:\d\d\.\d+\] ")
_RANK = re.compile(r"^\[R(\d+)\] ")
_LEVEL = re.compile(r"^\[(INFO|WARN|WARNING|DEBUG|ERROR)\] ")
_KV = re.compile(r"(?<![\w.\[])([A-Za-z_][\w.]*)=('[^']*'|\"[^\"]*\"|[^\s\]]*)")
_DIAG = re.compile(r"(?<![\w.])diagnostic=([A-Za-z0-9_]+)")
_BRACKET = re.compile(r"\s*\[([^\[\]]*=[^\[\]]*)\]")
_INT = re.compile(r"[-+]?\d+")
_REAL = re.compile(r"[-+]?(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?|[-+]?(nan|inf)")
_SNAKE = re.compile(r"[a-z][a-z0-9_]*")
# Owner tags of the old lines: a head is also matched without them.
_OLD_TAGS = ("[svMultiPhysics::Application] ", "[svMultiPhysics::FE] ", "FESystem: ", "FESystem:")


def _core(head: str) -> str:
    for tag in _OLD_TAGS:
        if head.startswith(tag):
            return head[len(tag):].strip()
    return head.strip()


class Line:
    """One log line: rank, level, the text after the prefixes, its head (the
    text before the first field) and its fields in order."""

    __slots__ = ("raw", "rank", "level", "text", "head", "_fields", "diagnostic")

    def __init__(self, raw: str):
        self.raw = raw.rstrip("\n")
        text = _TS.sub("", self.raw)
        m = _RANK.match(text)
        self.rank = m.group(1) if m else None
        if m:
            text = text[m.end():]
        m = _LEVEL.match(text)
        self.level = m.group(1) if m else None
        if m:
            text = text[m.end():]
        self.text = text
        first = _KV.search(text)
        self.head = (text[: first.start()] if first else text).rstrip()
        self._fields = None
        d = _DIAG.search(text)
        self.diagnostic = d.group(1) if d else None

    @property
    def fields(self):
        if self._fields is None:
            self._fields = [(k, v) for k, v in _KV.findall(self.text)]
        return self._fields


class Record:
    """A record in the new format: owner, message, diagnostic, level, the
    fields (in order, without the diagnostic) and the child records."""

    __slots__ = ("entry", "owner", "message", "diagnostic", "level", "rank", "fields", "children", "lines")

    def __init__(self, entry, owner, message, diagnostic, level, rank, fields):
        self.entry = entry
        self.owner = owner
        self.message = message
        self.diagnostic = diagnostic
        self.level = level
        self.rank = rank
        self.fields = list(fields)
        self.children: list[Record] = []
        self.lines: list[str] = []

    def get(self, key, default=None):
        for k, v in self.fields:
            if k == key:
                return v
        return default

    def as_dict(self) -> dict:
        return dict(self.fields)

    def text(self) -> str:
        """The record's line as the FE logger prints it, without time, rank and level."""
        head = self.owner + ":" + (" " + self.message if self.message else "")
        parts = [head, "diagnostic=" + self.diagnostic] + ["%s=%s" % kv for kv in self.fields]
        return " ".join(parts)


class LogMap:
    def __init__(self, document: dict):
        self.entries = document["entries"]
        self.by_diagnostic = {}
        self.by_head = {}
        self.by_core = {}
        self.regex = []
        # Old lines without a key=value field that an entry reads (records()
        # skips other lines without '=').
        self.fieldless = tuple(e["before"]["fieldless_marker"] for e in self.entries
                               if e["before"].get("fieldless_marker"))
        for e in self.entries:
            before, after = e["before"], e["after"]
            if before.get("diagnostic"):
                self.by_diagnostic.setdefault(before["diagnostic"], (e, "parent"))
            self.by_diagnostic.setdefault(after["diagnostic"], (e, "parent"))
            for s in e.get("split") or []:
                self.by_diagnostic.setdefault(s["diagnostic"], (e, "child"))
            if e.get("child_diagnostic"):
                self.by_diagnostic.setdefault(e["child_diagnostic"], (e, "child"))
            # Further child words of an entry (e.g. a child line's own children).
            for word in e.get("other_children") or []:
                self.by_diagnostic.setdefault(word, (e, "child"))
            if before.get("head"):
                self.by_head.setdefault(before["head"], e)
                if _core(before["head"]):
                    self.by_core.setdefault(_core(before["head"]), e)
            if before.get("head_regex"):
                self.regex.append(("head", re.compile(before["head_regex"]), e))
            if before.get("line_regex"):
                self.regex.append(("line", re.compile(before["line_regex"]), e))

    def match(self, line: Line):
        """(entry, role) of a line in either format, or (None, None)."""
        if line.diagnostic is not None:
            hit = self.by_diagnostic.get(line.diagnostic)
            if hit is not None:
                return hit
            return None, None
        e = self.by_head.get(line.head) or self.by_core.get(_core(line.head))
        if e is not None:
            return e, "parent"
        for kind, rx, e in self.regex:
            if rx.match(line.head if kind == "head" else line.text):
                return e, "parent"
        return None, None


@functools.lru_cache(maxsize=4)
def load_map(path: str | None = None) -> LogMap:
    with open(path or MAP_PATH) as f:
        return LogMap(json.load(f))


def levels(entry: dict) -> list:
    """The levels the map allows for the entry's record (one, or a list)."""
    level = entry["after"]["level"]
    return list(level) if isinstance(level, list) else [level]


def _canonical(line: Line, entry: dict, role: str) -> Record:
    after = dict(entry["after"])
    after["level"] = levels(entry)[0]
    rank = line.rank if line.rank is not None else "0"
    if role == "child":
        fields = [(k, v) for k, v in line.fields if k != "diagnostic"]
        return Record(entry, after["owner"], after["message"], line.diagnostic, after["level"], rank, fields)
    text = line.text
    special = entry.get("special")
    children = []
    extra = []
    if special == "bracket_points":
        groups = _BRACKET.findall(text)
        text = _BRACKET.sub("", text)
        for k, g in enumerate(groups):
            children.append([("point", str(k))] + _KV.findall(g))
        m = re.match(entry["before"]["head_regex"], line.head)
        if m:
            extra.append(("constraint", "'%s'" % m.group("constraint")))
    if special == "input_deprecation_text" and line.diagnostic is None:
        m = re.match(entry["before"]["line_regex"], line.text)
        fields = []
        message = after["message"]
        if m:
            groups = m.groupdict()
            fields = [("setting", "'%s'" % groups["setting"]), ("location", "'%s'" % groups["location"]),
                      ("decision", groups.get("decision") or groups.get("decision2"))]
            if groups.get("replacement") is not None:
                fields.append(("replacement", "'%s'" % groups["replacement"]))
                message = (after.get("alternate_messages") or [message])[0]
            else:
                fields.append(("effect", "none"))
        return Record(entry, after["owner"], message, after["diagnostic"], after["level"], rank, fields)
    if special == "regex_fields" and line.diagnostic is None:
        m = re.match(entry["before"]["line_regex"], line.text)
        quoted = set(entry.get("quoted_fields") or [])
        fields = [(k, "'%s'" % v if k in quoted else v) for k, v in (m.groupdict().items() if m else [])]
        return Record(entry, after["owner"], after["message"], after["diagnostic"], after["level"], rank, fields)
    pairs = [(k, v) for k, v in _KV.findall(text) if k != "diagnostic"]
    if line.diagnostic is None:   # an old line: the fields the record adds
        pairs += [(k, v) for k, v in entry.get("added_fields") or []]
    renames = entry.get("renames") or {}
    pairs = [(renames.get(k, k), v) for k, v in pairs]
    if special == "interface_quantity_keys":
        kept, walls = [], {}
        for k, v in pairs:
            m = re.match(r"free_surface\.\d+\.wall\.(\d+)\.(\w+)$", k)
            if m:
                walls.setdefault(m.group(1), []).append((m.group(2), v))
                continue
            m = re.match(r"free_surface\.\d+\.(\w+)$", k)
            kept.append((m.group(1) if m else k, v))
        pairs = kept
        for b, ws in walls.items():
            children.append([("wall_marker", b)] + ws)
    groups = []
    split = entry.get("split") or []
    if split:
        starts = {s["start"]: s for s in split}
        kept, cur = [], None
        for k, v in pairs:
            if k in starts:
                cur = (starts[k], [(k, v)])
                groups.append(cur)
                continue
            if cur is not None and k.startswith(cur[0]["prefix"]) and not (
                    cur[0].get("exclude_prefix") and k.startswith(cur[0]["exclude_prefix"])):
                cur[1].append((k, v))
                continue
            cur = None
            kept.append((k, v))
        pairs = kept
    record = Record(entry, after["owner"], after["message"], after["diagnostic"], after["level"], rank, extra + pairs)
    ident = [(k, v) for k, v in record.fields if k in (entry.get("identifying") or [])]
    child_word = entry.get("child_diagnostic")
    for c in children:
        record.children.append(Record(entry, after["owner"], after["message"], child_word, after["level"], rank,
                                      ident + c))
    for spec, fields in groups:
        record.children.append(Record(entry, after["owner"], after["message"], spec["diagnostic"], after["level"],
                                      rank, ident + fields))
    return record


def line_diagnostic(raw: str, logmap: LogMap | None = None) -> str | None:
    """The diagnostic word of a line in either format: the new word of a
    converted line (also for its old text), else the line's own word."""
    logmap = logmap or load_map()
    if "=" not in raw and "WARNING: '" not in raw and not any(m in raw for m in logmap.fieldless):
        return None
    line = Line(raw)
    entry, role = logmap.match(line)
    if entry is None:
        return line.diagnostic
    return line.diagnostic if role == "child" else entry["after"]["diagnostic"]


def is_record(raw: str, diagnostic: str, logmap: LogMap | None = None) -> bool:
    """Whether a line, in either format, is a record with the word @p diagnostic."""
    return line_diagnostic(raw, logmap) == diagnostic


def line_record(raw: str, logmap: LogMap | None = None) -> Record | None:
    """The record of one line in either format (old repeated groups as
    children), or None for a line the map does not cover."""
    line = Line(raw)
    entry, role = (logmap or load_map()).match(line)
    if entry is None:
        return None
    record = _canonical(line, entry, role)
    record.lines.append(raw)
    return record


def line_fields(raw: str, logmap: LogMap | None = None) -> dict:
    """The fields of a line under their new names (a mapped line), or as
    printed (any other line); without the diagnostic."""
    record = line_record(raw, logmap)
    if record is None:
        return {k: v for k, v in Line(raw).fields if k != "diagnostic"}
    return record.as_dict()


def records(lines: Iterable[str] | str, diagnostic: str | None = None,
            logmap: LogMap | None = None) -> list[Record]:
    """The mapped records of a log in order, the child lines of the new
    format attached to their parent (the last one of the entry on that
    rank); only those with @p diagnostic when given."""
    logmap = logmap or load_map()
    if isinstance(lines, str):
        lines = lines.splitlines()
    out: list[Record] = []
    last: dict = {}
    for raw in lines:
        if "=" not in raw and "WARNING: '" not in raw and not any(m in raw for m in logmap.fieldless):
            continue
        line = Line(raw)
        entry, role = logmap.match(line)
        if entry is None:
            continue
        record = _canonical(line, entry, role)
        record.lines.append(raw)
        if role == "child":
            parent = last.get((entry["id"], record.rank))
            if parent is not None:
                parent.children.append(record)
                parent.lines.append(raw)
            continue
        last[(entry["id"], record.rank)] = record
        out.append(record)
    if diagnostic is not None:
        out = [r for r in out if r.diagnostic == diagnostic]
    return out


def legacy_line(record: Record) -> str:
    """The record in the old shape (old head, old key names, repeated groups
    and interface-prefixed keys merged back), with the record's value texts:
    for comparators written against the old lines.  Prefixes are the first
    line's."""
    entry = record.entry
    before = entry["before"]
    inverse = {v: k for k, v in (entry.get("renames") or {}).items()}
    fields = [(inverse.get(k, k), v) for k, v in record.fields]
    if entry.get("special") == "interface_quantity_keys":
        m = record.get("interface_marker")
        fields = [(k if k in ("accepted_step", "interface_marker") else "free_surface.%s.%s" % (m, k), v)
                  for k, v in fields]
    ident = set(entry.get("identifying") or [])
    for child in record.children:
        cf = [(inverse.get(k, k), v) for k, v in child.fields if k not in ident]
        if entry.get("special") == "interface_quantity_keys":
            b = child.get("wall_marker")
            m = record.get("interface_marker")
            cf = [("free_surface.%s.wall.%s.%s" % (m, b, k), v) for k, v in cf if k != "wall_marker"]
        fields.extend(cf)
    head = before.get("head") or record.owner + ":"
    parts = [head]
    if before.get("diagnostic"):
        parts.append("diagnostic=" + before["diagnostic"])
    parts.extend("%s=%s" % kv for kv in fields)
    first = record.lines[0] if record.lines else ""
    prefix = first[: len(first) - len(Line(first).text)] if first else ""
    return prefix + " ".join(parts)


def legacy_view(lines: Iterable[str], logmap: LogMap | None = None) -> list[str]:
    """The lines of a log with every new-format mapped record in the old
    shape (legacy_line; its child lines folded in); every other line as is."""
    logmap = logmap or load_map()
    out: list[str] = []
    pending: dict = {}
    for raw in lines:
        if "diagnostic=" not in raw:
            out.append(raw)
            continue
        line = Line(raw)
        entry, role = logmap.match(line)
        if entry is None or not _is_new_format(line, entry, role):
            out.append(raw)
            continue
        record = _canonical(line, entry, role)
        record.lines.append(raw)
        key = (entry["id"], record.rank)
        if role == "child":
            parent = pending.get(key)
            if parent is not None:
                parent[1].children.append(record)
                out[parent[0]] = legacy_line(parent[1])
            continue
        pending[key] = (len(out), record)
        out.append(legacy_line(record))
    return out


def _is_new_format(line: Line, entry: dict, role: str) -> bool:
    after = entry["after"]
    if role == "child":
        return True
    heads = {after["owner"] + ":" + (" " + message if message else "")
             for message in [after["message"]] + list(after.get("alternate_messages") or [])}
    return line.head in heads and line.fields[:1] == [("diagnostic", after["diagnostic"])]


def value(text: str):
    """A printed value: int, float, bool, None (none), or the text (quotes removed)."""
    if text.startswith("'") and text.endswith("'"):
        return text[1:-1]
    if text in ("true", "false"):
        return text == "true"
    if text == "none":
        return None
    if _INT.fullmatch(text):
        return int(text)
    if _REAL.fullmatch(text):
        return float(text)
    return text


def cxx_shortest(x: float) -> str:
    """std::to_chars(double) without a format: the shortest round-trip
    digits, fixed or scientific, whichever is shorter (fixed on a tie)."""
    if math.isnan(x):
        return "nan"
    if math.isinf(x):
        return "inf" if x > 0 else "-inf"
    r = repr(float(x))
    sign = "-" if r.startswith("-") else ""
    r = r.lstrip("-")
    if "e" in r:
        mant, exp = r.split("e")
        exp = int(exp)
    else:
        mant, exp = r, 0
    if "." in mant:
        ip, fp = mant.split(".")
    else:
        ip, fp = mant, ""
    digits = (ip + fp).lstrip("0")
    point = len(ip) + exp  # position of the decimal point in ip+fp
    lead = len(ip + fp) - len((ip + fp).lstrip("0"))
    point -= lead
    digits = digits.rstrip("0") or "0"
    if digits == "0":
        return sign + "0"
    n = len(digits)
    # fixed
    if point <= 0:
        fixed = "0." + "0" * (-point) + digits
    elif point >= n:
        fixed = digits + "0" * (point - n)
    else:
        fixed = digits[:point] + "." + digits[point:]
    e = point - 1
    sci = digits[0] + ("." + digits[1:] if n > 1 else "") + "e" + ("-" if e < 0 else "+") + "%02d" % abs(e)
    return sign + (fixed if len(fixed) <= len(sci) else sci)


def values_equal(base: str, new: str) -> bool:
    """The conversion gate's value comparison: the same text, the same
    double, or a new value that the base line's format prints as the base
    text (the old line printed it rounded)."""
    if base == new:
        return True
    b, n = value(base), value(new)
    if isinstance(b, bool) or isinstance(n, bool) or b is None or n is None:
        return False
    if not isinstance(b, (int, float)) or not isinstance(n, (int, float)):
        return False
    if float(b) == float(n):
        return True
    x = float(n)
    m = re.fullmatch(r"[-+]?(\d)\.?(\d*)[eE]([-+]?\d+)", base)
    if m:
        return ("%." + str(len(m.group(2))) + "e") % x == base
    if "." in base:
        decimals = len(base.split(".")[1])
        if ("%." + str(decimals) + "f") % x == base:
            return True
    for p in range(1, 18):
        if ("%." + str(p) + "g") % x == base:
            return True
    return False


def lint_line(raw: str) -> list[str]:
    """The conventions a converted line breaks (empty when none): one
    diagnostic, the first field; snake_case keys, each once; reals in the
    shortest round-trip text."""
    line = Line(raw)
    problems = []
    keys = [k for k, _ in line.fields]
    if keys.count("diagnostic") != 1:
        problems.append("diagnostic count %d" % keys.count("diagnostic"))
    elif keys[0] != "diagnostic":
        problems.append("diagnostic is not the first field")
    seen = set()
    for k, v in line.fields:
        if not _SNAKE.fullmatch(k):
            problems.append("key %s is not snake_case" % k)
        if k in seen:
            problems.append("key %s repeats" % k)
        seen.add(k)
        if _REAL.fullmatch(v) and not _INT.fullmatch(v) and v not in ("nan", "inf", "-inf"):
            if cxx_shortest(float(v)) != v:
                problems.append("real %s=%s is not the shortest text (%s)" % (k, v, cxx_shortest(float(v))))
    return problems
