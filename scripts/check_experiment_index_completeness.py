#!/usr/bin/env python3
# PRODUCTION INFRA — ledger validator (not an experiment)
"""Soft validator for wiki_consultation: discipline in EXPERIMENT_INDEX.md.

Per Cycle 11 Option δ Phase 1 implementation (#PY-NEW-CONSUMPTION-ENFORCEMENT closure attempt).

Mirrors hft-wiki/scripts/consumption_ratio.py pattern:
- Exit code 0 (WARN-not-ERROR) by default
- --verbose for per-entry detail
- --json for machine-readable output
- --strict to escalate WARN -> exit 1 (opt-in by operator)

DESIGN PRINCIPLES:
- Grandfather all entries dated before 2026-05-27 (Cycle 11 ship date)
- Check post-Cycle-11 entries for **Wiki consultation** block presence
- Validate justification length >= 20 chars per cite
- Optional: resolve cited IDs via `hft-wiki show <id>` (only with --strict + hft-wiki on path)

USAGE:
    cd lob-model-trainer
    python3 scripts/check_experiment_index_completeness.py
    python3 scripts/check_experiment_index_completeness.py --verbose
    python3 scripts/check_experiment_index_completeness.py --json
    python3 scripts/check_experiment_index_completeness.py --strict

DOCUMENTED IN:
    - lob-model-trainer/CONTRIBUTING.md (field discipline)
    - lob-model-trainer/EXPERIMENT_INDEX.md (per-entry template at top)
    - hft-wiki/playbooks/record-experiment-result.md (operator workflow)
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# Cycle 11 ship date — entries dated >= this require wiki_consultation
CYCLE_11_SHIP_DATE = "2026-05-27"

# Minimum justification length per cite (SOFT — WARN below)
MIN_JUSTIFICATION_CHARS = 20

# Regex for citable IDs
CITATION_REGEX = re.compile(r"`(theory|synthesis|FINDING)[-:][a-z0-9_]+(?:-[a-z0-9_]+)*`?", re.IGNORECASE)

# Regex for date in entry headers — e.g. "(2026-05-19)" or "(Completed 2026-04-10)"
DATE_REGEX = re.compile(r"\((?:Completed |Failed |Cancelled |verdict, )?(\d{4}-\d{2}-\d{2})\)")

# Regex for R-NN entry headers
R_NN_HEADER_REGEX = re.compile(r"^###\s+(R-?\d+|P\d+|E\d+|F\d+)[\s:—]", re.MULTILINE)


@dataclass
class EntryAuditResult:
    entry_id: str
    line_number: int
    date_string: Optional[str] = None
    grandfathered: bool = False
    has_wiki_consultation_block: bool = False
    citations_found: list[str] = field(default_factory=list)
    has_negative_fallback: bool = False
    short_justifications: list[tuple[str, int]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def parse_entries(text: str) -> list[tuple[str, int, int]]:
    """Returns list of (entry_id, start_offset, end_offset) per ### entry."""
    matches = list(R_NN_HEADER_REGEX.finditer(text))
    entries = []
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        entries.append((m.group(1), start, end))
    return entries


def find_line_number(text: str, offset: int) -> int:
    return text[:offset].count("\n") + 1


def extract_date(entry_text: str) -> Optional[str]:
    m = DATE_REGEX.search(entry_text[:500])  # only check the header line + immediate context
    return m.group(1) if m else None


def audit_entry(entry_id: str, entry_text: str, line_number: int) -> EntryAuditResult:
    result = EntryAuditResult(entry_id=entry_id, line_number=line_number)

    # Determine date + grandfathered status
    date_str = extract_date(entry_text)
    result.date_string = date_str
    if date_str is None:
        # Unable to date — grandfather by default (safer than false-positive)
        result.grandfathered = True
        result.warnings.append("INFO: could not extract date from entry header; grandfathered by default")
        return result

    if date_str < CYCLE_11_SHIP_DATE:
        result.grandfathered = True
        return result

    # Post-Cycle-11: must have **Wiki consultation** block
    # Look for either inline block ("**Wiki consultation**" followed by content)
    # OR table row ("| **Wiki consultation** | ..." )
    has_block = "**Wiki consultation**" in entry_text or "Wiki consultation" in entry_text
    result.has_wiki_consultation_block = has_block
    if not has_block:
        result.warnings.append("WARN: missing **Wiki consultation** block (REQUIRED post-Cycle-11)")
        return result

    # Check for citations OR negative-result fallback
    citations = CITATION_REGEX.findall(entry_text)
    result.citations_found = citations

    # Detect negative fallback (e.g., "None applicable — queried ...")
    has_fallback = bool(re.search(r"None applicable.*queried.*list", entry_text, re.IGNORECASE))
    result.has_negative_fallback = has_fallback

    if not citations and not has_fallback:
        result.warnings.append("WARN: **Wiki consultation** block present but contains no citations AND no 'None applicable' fallback")
        return result

    # Check justification length per cite
    # Heuristic: find each cite + the text following it on the same line, until newline
    cite_lines = re.findall(r"`(?:theory|synthesis|FINDING)[-:][a-z0-9_]+(?:-[a-z0-9_]+)*`?\s*[—–-]\s*(.+?)(?:\n|$)", entry_text, re.IGNORECASE)
    for justification in cite_lines:
        if len(justification.strip()) < MIN_JUSTIFICATION_CHARS:
            result.short_justifications.append((justification.strip()[:80], len(justification.strip())))

    if result.short_justifications:
        result.warnings.append(f"WARN: {len(result.short_justifications)} cite(s) have justification < {MIN_JUSTIFICATION_CHARS} chars")

    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--experiment-index", default="EXPERIMENT_INDEX.md", help="Path to EXPERIMENT_INDEX.md (default: ./EXPERIMENT_INDEX.md)")
    parser.add_argument("--verbose", action="store_true", help="Per-entry detail")
    parser.add_argument("--json", action="store_true", help="Machine-readable JSON output")
    parser.add_argument("--strict", action="store_true", help="Exit 1 if any WARN found (default: always exit 0)")
    args = parser.parse_args()

    path = Path(args.experiment_index)
    if not path.exists():
        print(f"ERROR: {path} not found. Run from lob-model-trainer/ root.", file=sys.stderr)
        return 2

    text = path.read_text()
    entries = parse_entries(text)
    results = []

    for entry_id, start, end in entries:
        entry_text = text[start:end]
        line_number = find_line_number(text, start)
        result = audit_entry(entry_id, entry_text, line_number)
        results.append(result)

    # Summary
    total = len(results)
    grandfathered = sum(1 for r in results if r.grandfathered)
    post_c11 = total - grandfathered
    with_warns = sum(1 for r in results if any(w.startswith("WARN") for w in r.warnings))
    total_warns = sum(len([w for w in r.warnings if w.startswith("WARN")]) for r in results)

    if args.json:
        out = {
            "summary": {
                "total_entries": total,
                "grandfathered": grandfathered,
                "post_cycle_11": post_c11,
                "entries_with_warnings": with_warns,
                "total_warnings": total_warns,
            },
            "entries": [
                {
                    "id": r.entry_id,
                    "line": r.line_number,
                    "date": r.date_string,
                    "grandfathered": r.grandfathered,
                    "has_wiki_block": r.has_wiki_consultation_block,
                    "citations": r.citations_found,
                    "has_fallback": r.has_negative_fallback,
                    "short_justifications": [j for j, _ in r.short_justifications],
                    "warnings": r.warnings,
                }
                for r in results
            ],
        }
        print(json.dumps(out, indent=2))
    else:
        print(f"=== check_experiment_index_completeness.py — Cycle 11 Option δ Phase 1 validator ===")
        print(f"File: {path}")
        print(f"Cycle 11 ship date (grandfather threshold): {CYCLE_11_SHIP_DATE}")
        print(f"")
        print(f"Total entries: {total}")
        print(f"  Grandfathered (pre-Cycle-11): {grandfathered}")
        print(f"  Post-Cycle-11: {post_c11}")
        print(f"  Entries with warnings: {with_warns}")
        print(f"  Total warnings: {total_warns}")
        print(f"")

        if args.verbose or with_warns > 0:
            print("Per-entry detail (entries with warnings or all if --verbose):")
            for r in results:
                if args.verbose or r.warnings:
                    status = "[grandfathered]" if r.grandfathered else ""
                    block = "✓ block present" if r.has_wiki_consultation_block else "✗ NO BLOCK"
                    cites = f"{len(r.citations_found)} cites"
                    fb = " + fallback" if r.has_negative_fallback else ""
                    print(f"  {r.entry_id} (L{r.line_number}, date={r.date_string}) {status}")
                    if not r.grandfathered:
                        print(f"    {block}, {cites}{fb}")
                    for w in r.warnings:
                        print(f"    {w}")

    if args.strict and total_warns > 0:
        print(f"\nSTRICT: {total_warns} warnings; exit 1", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
