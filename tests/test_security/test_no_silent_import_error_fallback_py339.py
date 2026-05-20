"""#PY-339 lock test — simple_trainer.py must NOT silently fall back to
hardcoded contract indices on hft_contracts ImportError.

Pre-fix (2026-05-21; closed Cycle A-rev): ``simple_trainer.py:55-60`` had a
silent ``except ImportError:`` block that hardcoded ``MID_PRICE_IDX = 40``
+ ``SPREAD_BPS_IDX = 42`` as fallback. This was DEAD CODE under correct dep
install (hft-contracts is pinned dep; symbols exist since Phase Q.6.5 at
``hft_contracts/_generated.py:481+484``, re-exported via
``__init__.py:96-97 + 231-232``). BUT when dep resolution silently failed
(editable install drift / pin mismatch / partial install / circular import
during testing), the fallback substituted hardcoded constants instead of
failing loud — DOUBLE violation of hft-rules §0 (no hardcoded indices;
reuse-first SSoT) + §8 (never silently drop/clamp/fix without diagnostics).

Per §5 fail-fast, ImportError now propagates so operators see the real
upstream dep gap instead of silently consuming wrong indices that bypass
the canonical-hash SSoT.

Pattern mirrors FIND-110 (lob-backtester:
``tests/test_security/test_np_load_allow_pickle_false.py``): AST walk
catches the pattern at source level — robust against test isolation issues
that ``sys.modules`` monkey-patch would face.

See PHASE_P_BACKLOG.md #PY-339 and validation cycle 2026-05-21 Wave 1
Agent D + Wave 2 ground-truth verification.
"""

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
TRAINER_SRC = REPO_ROOT / "src" / "lobtrainer"


def _find_hardcoded_constant_assigns(tree: ast.AST, name: str, value: int) -> list:
    """Return list of (line, target_id) for ``<name> = <value>`` Assign nodes.

    Uses AST (not regex) so comments / docstrings / string-literal mentions
    are not false-positives. Catches:
        MID_PRICE_IDX = 40
        SPREAD_BPS_IDX = 42
    Does NOT catch (correctly):
        # MID_PRICE_IDX = 40  (comment)
        \"\"\"... MID_PRICE_IDX = 40 ...\"\"\"  (docstring)
        x = MID_PRICE_IDX  (read access)
        MID_PRICE_IDX = some_func()  (different value)
    """
    hits = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        # Check value is the exact integer constant
        if not (isinstance(node.value, ast.Constant) and node.value.value == value):
            continue
        # Check any target is the exact name
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == name:
                hits.append((node.lineno, target.id))
    return hits


class TestPy339NoSilentImportErrorFallback:
    """#PY-339 lock: no silent ImportError fallback for hft_contracts
    SIGNAL_PRICE_FEATURE_INDEX or SIGNAL_SPREAD_FEATURE_INDEX symbols.
    """

    def test_simple_trainer_has_direct_hft_contracts_imports(self):
        """Verify simple_trainer.py imports hft_contracts symbols directly
        WITHOUT try/except ImportError wrapper.

        Per §5 fail-fast: ImportError must propagate to surface the real
        dep gap (e.g., editable install drift, pin mismatch). Pre-#PY-339
        fallback consumed hardcoded constants silently bypassing SSoT.
        """
        simple_trainer = TRAINER_SRC / "training" / "simple_trainer.py"
        assert simple_trainer.exists(), (
            f"simple_trainer.py not found at {simple_trainer}"
        )

        source = simple_trainer.read_text()
        tree = ast.parse(source, filename=str(simple_trainer))

        # Find all `try: ... except ImportError: ...` blocks that contain
        # a hft_contracts import with SIGNAL_PRICE_FEATURE_INDEX or
        # SIGNAL_SPREAD_FEATURE_INDEX
        offenders = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Try):
                continue
            # Check if try body imports SIGNAL_*_FEATURE_INDEX from hft_contracts
            imports_signal_idx = False
            for stmt in node.body:
                if isinstance(stmt, ast.ImportFrom):
                    if stmt.module and "hft_contracts" in stmt.module:
                        for alias in stmt.names:
                            if alias.name in (
                                "SIGNAL_PRICE_FEATURE_INDEX",
                                "SIGNAL_SPREAD_FEATURE_INDEX",
                            ):
                                imports_signal_idx = True
                                break
                if imports_signal_idx:
                    break
            if not imports_signal_idx:
                continue
            # Check if there's an except ImportError handler
            for handler in node.handlers:
                if handler.type is None:
                    # bare except — also bad, catches everything
                    offenders.append(
                        f"line {node.lineno}: try/except (bare) wrapping "
                        f"SIGNAL_*_FEATURE_INDEX import (silently masks ImportError)"
                    )
                elif isinstance(handler.type, ast.Name) and handler.type.id == "ImportError":
                    offenders.append(
                        f"line {node.lineno}: try/except ImportError wrapping "
                        f"SIGNAL_*_FEATURE_INDEX import (silently masks dep gap; "
                        f"reintroduces #PY-339 silent SSoT bypass)"
                    )
                elif isinstance(handler.type, ast.Tuple):
                    # except (ImportError, ...) tuple
                    for elt in handler.type.elts:
                        if isinstance(elt, ast.Name) and elt.id == "ImportError":
                            offenders.append(
                                f"line {node.lineno}: try/except (ImportError, ...) "
                                f"wrapping SIGNAL_*_FEATURE_INDEX import"
                            )

        assert not offenders, (
            "#PY-339 lock: simple_trainer.py must NOT wrap "
            "SIGNAL_*_FEATURE_INDEX imports in try/except ImportError. "
            "Per hft-rules §5 (fail-fast) + §8 (no silent fallbacks) + §0 "
            "(no hardcoded indices; reuse-first SSoT). Offenders:\n  "
            + "\n  ".join(offenders)
        )

    def test_no_hardcoded_mid_price_idx_40_in_trainer_src(self):
        """Verify no source file in trainer hardcodes ``MID_PRICE_IDX = 40``
        as an Assign statement (canonical value lives in
        ``hft_contracts._generated`` SSoT). AST-based — ignores comments
        and docstring mentions per false-positive avoidance.
        """
        offenders = []
        for py in TRAINER_SRC.rglob("*.py"):
            if py.name == "__init__.py":
                continue
            try:
                tree = ast.parse(py.read_text(), filename=str(py))
            except SyntaxError:
                continue
            for lineno, _ in _find_hardcoded_constant_assigns(
                tree, "MID_PRICE_IDX", 40
            ):
                offenders.append(f"{py.relative_to(REPO_ROOT)}:{lineno}")

        assert not offenders, (
            "#PY-339 lock: no source file may hardcode `MID_PRICE_IDX = 40` "
            "as an Assign statement. Canonical value lives in "
            "hft_contracts._generated (auto-gen from "
            "contracts/pipeline_contract.toml SSoT). Offenders:\n  "
            + "\n  ".join(offenders)
        )

    def test_no_hardcoded_spread_bps_idx_42_in_trainer_src(self):
        """Verify no source file in trainer hardcodes ``SPREAD_BPS_IDX = 42``
        as an Assign statement. AST-based per #PY-339 lock pattern.
        """
        offenders = []
        for py in TRAINER_SRC.rglob("*.py"):
            if py.name == "__init__.py":
                continue
            try:
                tree = ast.parse(py.read_text(), filename=str(py))
            except SyntaxError:
                continue
            for lineno, _ in _find_hardcoded_constant_assigns(
                tree, "SPREAD_BPS_IDX", 42
            ):
                offenders.append(f"{py.relative_to(REPO_ROOT)}:{lineno}")

        assert not offenders, (
            "#PY-339 lock: no source file may hardcode `SPREAD_BPS_IDX = 42` "
            "as an Assign statement. Canonical value lives in "
            "hft_contracts._generated (auto-gen from "
            "contracts/pipeline_contract.toml SSoT). Offenders:\n  "
            + "\n  ".join(offenders)
        )

    def test_simple_trainer_imports_resolve_at_module_load(self):
        """Sanity: simple_trainer.py imports MID_PRICE_IDX + SPREAD_BPS_IDX
        successfully at module load time (validates fix didn't break imports).

        If hft_contracts is properly installed, this should always pass.
        If hft_contracts is missing, this will raise ImportError as designed
        (per #PY-339 fail-fast — no silent fallback).
        """
        from lobtrainer.training import simple_trainer as st
        assert hasattr(st, "MID_PRICE_IDX"), "MID_PRICE_IDX must be importable"
        assert hasattr(st, "SPREAD_BPS_IDX"), "SPREAD_BPS_IDX must be importable"
        # Verify they resolve to canonical SSoT values (currently 40 + 42 per
        # contracts/pipeline_contract.toml; this test will need update if
        # contract bumps the indices via TOML SSoT path).
        from hft_contracts import (
            SIGNAL_PRICE_FEATURE_INDEX,
            SIGNAL_SPREAD_FEATURE_INDEX,
        )
        assert st.MID_PRICE_IDX == SIGNAL_PRICE_FEATURE_INDEX, (
            "#PY-339: MID_PRICE_IDX must equal hft_contracts SSoT value. "
            f"Trainer: {st.MID_PRICE_IDX}, SSoT: {SIGNAL_PRICE_FEATURE_INDEX}"
        )
        assert st.SPREAD_BPS_IDX == SIGNAL_SPREAD_FEATURE_INDEX, (
            "#PY-339: SPREAD_BPS_IDX must equal hft_contracts SSoT value. "
            f"Trainer: {st.SPREAD_BPS_IDX}, SSoT: {SIGNAL_SPREAD_FEATURE_INDEX}"
        )
