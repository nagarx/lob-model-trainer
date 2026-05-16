"""#PY-291 lock test — every ``np.load()`` must pass ``allow_pickle=False``.

Without ``allow_pickle=False``, ``np.load()`` accepts pickled Python objects in
``.npy`` files, opening the door to remote code execution if a ``.npy`` file
arrives from an untrusted source (CVE-class hazard per hft-rules §8). This
test scans every ``np.load(`` callsite in ``src/``, ``tests/``, and
``scripts/`` (excluding ``scripts/archive/`` per hft-rules §4 fossil discipline)
and asserts the ``allow_pickle=False`` keyword appears within each call's
argument span.

Sister of FIND-110 lock-test at ``lob-backtester/tests/test_security/`` (commit
``20dbc8f`` 2026-05-14). Closes the RCE-via-malicious-NPY class for the
trainer surface where 18 callsites were silently unguarded before
2026-05-16 LATE Option B Path B' hygiene cycle (see
``CROSS_PIPELINE_VALIDATION_FINDINGS_2026_05_16.md`` §3 #PY-291).

Known limitation: the test scans for the textual pattern ``np.load(``. It does
not catch aliases like ``from numpy import load as _l; _l(...)`` or
fully-qualified ``numpy.load(...)``. The lob-model-trainer codebase convention
is ``import numpy as np``. If a future contributor introduces an alias,
extend the patterns below.
"""

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
NP_LOAD_RE = re.compile(r"np\.load\s*\(")
ALLOW_PICKLE_FALSE_RE = re.compile(r"allow_pickle\s*=\s*False")

# Fossil archive directories — explicitly excluded per hft-rules §4
# (pre-Phase-6D scripts retained for historical reproducibility only;
# NOT templates for new code per scripts/archive/README.md).
ARCHIVE_DIRS = {
    "scripts/archive",
}


def _extract_call_span(text: str, open_paren_idx: int) -> str:
    """Return the call's argument span: from ``(`` through the matching ``)``.

    Handles nested parens via depth tracking. Critical for multi-line calls
    like ``np.load(\\n    Path(p), mmap_mode=m, allow_pickle=False\\n)`` where
    the inner ``Path()`` parens must not confuse the matching algorithm.
    """
    depth = 0
    end = open_paren_idx
    for i in range(open_paren_idx, len(text)):
        c = text[i]
        if c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
            if depth == 0:
                end = i
                break
    return text[open_paren_idx : end + 1]


def _is_archive_path(path: Path) -> bool:
    """Return True if path is under any documented fossil archive."""
    rel_str = str(path.relative_to(REPO_ROOT))
    return any(rel_str.startswith(arc + "/") or rel_str.startswith(arc + "\\") for arc in ARCHIVE_DIRS)


class TestPy291AllowPickleFalseLock:
    """#PY-291 lock: every ``np.load()`` must pass ``allow_pickle=False``."""

    def test_every_np_load_passes_allow_pickle_false(self):
        """No ``np.load()`` callsite in src/, tests/, scripts/ (excluding
        scripts/archive/) may omit ``allow_pickle=False``.

        Removing ``allow_pickle=False`` from any callsite re-opens the
        pickle-RCE vector closed by #PY-291 (Option B Path B' hygiene cycle
        2026-05-16 LATE). If this test fails, the listed offenders MUST be
        hardened before merge.
        """
        offenders = []
        for sub in ("src", "tests", "scripts"):
            base = REPO_ROOT / sub
            if not base.exists():
                continue
            for py in base.rglob("*.py"):
                # Skip THIS file (would self-match the docstring + regex literals).
                if py.name == "test_np_load_allow_pickle_false.py":
                    continue
                # Skip fossil archives per hft-rules §4
                if _is_archive_path(py):
                    continue
                text = py.read_text()
                for m in NP_LOAD_RE.finditer(text):
                    # m.end() - 1 points at the `(` character.
                    span = _extract_call_span(text, m.end() - 1)
                    if not ALLOW_PICKLE_FALSE_RE.search(span):
                        line = text[: m.start()].count("\n") + 1
                        offenders.append(f"{py.relative_to(REPO_ROOT)}:{line}")
        assert not offenders, (
            "#PY-291 lock: every np.load() callsite must pass "
            "allow_pickle=False (prevents pickle-RCE on malicious .npy "
            "files; hft-rules §8). Offenders:\n  " + "\n  ".join(offenders)
        )

    def test_no_aliased_numpy_load_imports(self):
        """Defensive lock against aliased imports that bypass the np.load scan.

        Patterns like ``from numpy import load as _l`` or
        ``import numpy.load`` would defeat the regex-based scan above. We
        assert NEITHER pattern appears anywhere in production code.
        """
        forbidden_re = re.compile(
            r"from\s+numpy\s+import\s+.*\bload\b|import\s+numpy\.load\b"
        )
        offenders = []
        for sub in ("src", "tests", "scripts"):
            base = REPO_ROOT / sub
            if not base.exists():
                continue
            for py in base.rglob("*.py"):
                if py.name == "test_np_load_allow_pickle_false.py":
                    continue
                if _is_archive_path(py):
                    continue
                text = py.read_text()
                for m in forbidden_re.finditer(text):
                    line = text[: m.start()].count("\n") + 1
                    offenders.append(
                        f"{py.relative_to(REPO_ROOT)}:{line} ({m.group(0)})"
                    )
        assert not offenders, (
            "Aliased numpy.load imports defeat the np.load() regression-lock "
            "test. Use 'import numpy as np; np.load(...)' instead. Offenders:\n  "
            + "\n  ".join(offenders)
        )
