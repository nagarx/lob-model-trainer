"""Cross-module parity test: trainer inline `_compute_content_hash`
byte-matches `hft_contracts.canonical_hash` (Phase 4 Batch 4c hardening).

The trainer's resolver intentionally inlines `_compute_content_hash` to
stay independent of hft-ops (cross-venv boundary). The downside is that
the inline copy could silently drift from the canonical form used by
the producer (`hft_ops.feature_sets.hashing.compute_feature_set_hash`),
which would cause every FeatureSet produced by hft-ops to fail integrity
verification at the trainer.

These tests LOCK the byte parity by independently reproducing the same
canonical form (via hft_contracts.canonical_hash, which is the single
source of truth since Batch 4c hardening) and asserting equality with
the trainer's inlined result on a diverse set of fixtures.

If this test fails:
- Someone changed the canonical form in one place but not the other.
- Regenerate expected hashes ONLY after confirming the change is
  deliberate and all 5 call sites (hft-ops dedup, hft-ops lineage,
  evaluator pipeline, hft-ops feature_sets/hashing, this trainer
  inline) agree on the new form.
"""

from __future__ import annotations

import pytest

from lobtrainer.data.feature_set_resolver import _compute_content_hash


# ---------------------------------------------------------------------------
# Fixture set (diverse inputs to catch drift in any corner)
# ---------------------------------------------------------------------------


PARITY_CASES = [
    # (feature_indices, source_feature_count, contract_version)
    ([0], 1, "1.0"),                      # minimum
    ([0, 5, 12], 98, "2.2"),              # typical NVDA 98-feature
    ([0, 5, 12, 84, 85, 86], 98, "2.2"),  # signals subset
    (list(range(98)), 98, "2.2"),         # all features
    (list(range(40)), 40, "2.2"),         # LOB-only
    ([127], 128, "2.2"),                  # last index of 128-feat
    ([0, 1, 2], 98, "2.3"),               # different contract version
    ([5, 12, 0], 98, "2.2"),              # unsorted — canonical form must normalize
    ([5, 5, 12, 0, 5], 98, "2.2"),        # duplicates — canonical form must dedupe
]


class TestCanonicalHashParity:
    """Byte-parity lock between trainer inline and hft_contracts canonical."""

    @pytest.mark.parametrize("indices,sfc,cv", PARITY_CASES)
    def test_trainer_inline_matches_hft_contracts_canonical(
        self, indices, sfc, cv,
    ):
        # Import the SSoT canonical form. hft_contracts is installed in
        # the trainer venv, so direct import works (no spec_from_file_location
        # needed — we're not cross-venv, we're cross-module).
        from hft_contracts.canonical_hash import canonical_json_blob, sha256_hex

        # Reproduce the canonical payload independently (exactly matches
        # what the trainer's _compute_content_hash does internally).
        canonical = {
            "feature_indices": sorted(set(int(i) for i in indices)),
            "source_feature_count": int(sfc),
            "contract_version": str(cv),
        }
        expected = sha256_hex(canonical_json_blob(canonical))

        # Trainer's inline implementation:
        got = _compute_content_hash(indices, sfc, cv)

        assert got == expected, (
            f"Canonical-hash parity drift between trainer inline and "
            f"hft_contracts.canonical_hash on inputs "
            f"(indices={indices}, sfc={sfc}, cv={cv}). "
            f"Trainer: {got}. Expected: {expected}. "
            f"Check lob-model-trainer/src/lobtrainer/data/feature_set_resolver.py::"
            f"_compute_content_hash against hft-contracts/src/hft_contracts/canonical_hash.py."
        )

    def test_all_cases_produce_distinct_hashes(self):
        # Sanity check: the 9 parity cases are semantically distinct, so
        # they must produce 9 distinct hashes. Protects against a
        # pathological bug where every hash returns a constant.
        hashes = {_compute_content_hash(*case) for case in PARITY_CASES}
        # Cases 2 and 7 differ only in sort/dedupe → should produce the
        # SAME hash as the canonical form normalizes them. So we expect
        # 9 - 1 (case [5,12,0] → same as [0,5,12]) - 1 (case [5,5,12,0,5]
        # → same as [0,5,12]) = 7 unique.
        assert len(hashes) == 7


class TestProducerConsumerParity:
    """Lock parity between what the producer (hft-ops) WOULD write and
    what the consumer (trainer) WOULD accept.

    hft-ops is NOT installed in the trainer venv (by design), so we
    cannot import hft_ops.feature_sets.hashing.compute_feature_set_hash
    directly. Instead we rely on hft_contracts.canonical_hash as the
    SSoT and verify both sides agree with it. The cross-venv load-level
    parity test lives in hft-ops/tests (not yet added — flag for
    Batch 4c.3 if it proves worthwhile).
    """

    def test_trainer_resolver_accepts_hft_contracts_canonical_hash(self):
        # If a FeatureSet JSON stores `content_hash` computed via
        # hft_contracts.canonical_hash, the trainer's integrity check
        # (_verify_content_hash → _compute_content_hash) must match.
        from hft_contracts.canonical_hash import canonical_json_blob, sha256_hex

        indices, sfc, cv = [0, 5, 12, 84], 98, "2.2"
        canonical = {
            "feature_indices": sorted(set(indices)),
            "source_feature_count": sfc,
            "contract_version": cv,
        }
        producer_hash = sha256_hex(canonical_json_blob(canonical))

        # This is what the trainer would compute when verifying:
        trainer_hash = _compute_content_hash(indices, sfc, cv)

        assert producer_hash == trainer_hash, (
            "Producer/consumer canonical-hash parity violation. "
            "A FeatureSet written by hft-ops would fail integrity "
            "verification at the trainer."
        )
