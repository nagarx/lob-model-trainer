"""Shared Pydantic v2 base class for every trainer config model.

**SafeBaseModel** packages the 4 hardening patterns that close 6 bug classes
at the TYPE layer. Every config class migrating from ``@dataclass`` to
Pydantic v2 under Phase A.5 Scope D inherits from this base — the patterns
are extracted HERE so individual class files stay minimal + no class can
accidentally miss a hardening.

Background (Phase A.5.3a.1 post-audit round, commit 180fff3, 2026-04-24):
Three specialized agents (code-reviewer + hft-architect + general-purpose)
validated the A.5.3a LabelsConfig migration and empirically confirmed 4
ship-blocker-class bugs in the naive Pydantic v2 defaults:

    1. ``model_copy(update={invalid})`` silently produces invalid instances.
       Pydantic v2's ``model_copy`` uses ``model_construct`` internally
       which SKIPS validation by design. The ``validate_assignment=True``
       flag does NOT fix this (only affects ``__setattr__``, not
       ``model_copy``). This re-opens bug class #3 (extra-field / invalid-
       value acceptance) through a common Pydantic idiom.

    2. ``horizons=["10","60"]`` silently coerces strings → ints under
       Pydantic v2 lax mode (default). Violates hft-rules §5 fail-fast.

    3. ``primary_horizon_idx=True`` silently accepts bool → 1 (bool is
       int-subclass in Python; Pydantic lax mode accepts).

    4. Container-mutation bypass: ``cfg.horizons.append(99)`` when horizons
       is ``List[int]`` — ``frozen=True`` blocks field ASSIGNMENT but NOT
       mutation of mutable containers. (Closed at the per-field level by
       changing to ``Tuple[...]`` with a ``@field_validator(mode="before")``
       for YAML list-input coercion — NOT packaged here since it's
       container-specific.)

**What SafeBaseModel packages** (applies to EVERY subclass):

    - ``frozen=True`` — field assignment raises ValidationError. Closes
      bug class: silent post-construction mutation (``cfg.source = X``).
    - ``extra="forbid"`` — unknown kwargs rejected at ``model_validate``
      time. Closes bug class: typo propagation / silent-accept of
      misspelled fields.
    - ``strict=True`` — NO implicit type coercion. Rejects ``string → int``
      (bug #2) and ``bool → int`` (bug #3). Closes bug class: silent
      type divergence.
      NOTE: strict does NOT reject NaN/Inf floats (IEEE 754 valid floats).
      Subclasses MUST add explicit ``math.isfinite`` checks for any float
      field where NaN would silently poison downstream logic.
    - ``model_copy`` override that dispatches to ``model_validate`` when
      ``update`` is provided — closes bug #1. No-update fast path preserved.

**What SafeBaseModel does NOT package** (subclass-specific):

    - ``@field_validator(mode="before")`` for container list → tuple
      coercion — container-specific, applied when a field uses ``Tuple[...]``
      to achieve true immutability (per A.5.3a.1 pattern on LabelsConfig.horizons).
    - ``@model_validator(mode="after")`` for cross-field invariants — domain-
      specific (e.g., SequenceConfig's ``stride ≤ window_size`` check,
      LabelsConfig's allowed-source frozenset membership).
    - ``math.isfinite`` checks for float fields — field-specific.
    - ``ClassVar[...]`` annotations on class-level constants — class-specific
      (v3-A discipline: without this, Pydantic v2 treats class-level
      frozensets etc. as model fields and leaks them into ``model_dump()``,
      breaking byte-identity of cross-module fingerprints).

**Migration checklist** (every class inheriting from SafeBaseModel):

    [ ] ``class MyConfig(SafeBaseModel):`` — drop ``@dataclass`` decorator
    [ ] NO inline ``model_config = ConfigDict(...)`` — inherited
    [ ] NO inline ``model_copy`` override — inherited
    [ ] Migrate ``__post_init__`` → ``@model_validator(mode="after")``
        (keep ValidationError-wrapping via ``raise ValueError(...)``
        in validator body — Pydantic wraps automatically)
    [ ] Annotate any class-level constants as ``ClassVar[...]`` (v3-A)
    [ ] For any float field that participates in thresholds/filters:
        add explicit ``math.isfinite(...)`` check
    [ ] For any List[...] field where post-construction mutation would
        break invariants: change to ``Tuple[..., ...]`` + add
        ``@field_validator(mode="before")`` for list→tuple coercion
        (YAML input compatibility)
    [ ] Add class to ``schema._PYDANTIC_CONFIG_CLASSES`` registry —
        auto-populates dacite's ``type_hooks`` table.

Design gate analysis (hft-rules §0-§14):

    - §1 SSoT: single place owning the 4 patterns. Prevents per-class
      drift / missed hardening.
    - §4 modularity: separates "Pydantic config conventions" (this file)
      from "domain schema" (schema.py). schema.py focuses on business
      rules; base.py owns framework discipline.
    - §5 fail-fast: strict=True + frozen=True + extra="forbid" all enforce
      the fail-fast doctrine. Every workaround fights Pydantic's lax
      defaults; making them the default via this base class flips the
      script — subclasses inherit "fail-fast by default" without any
      per-class opt-in.
    - §6 testing: ``tests/test_config.py::TestSafeBaseModel`` directly
      locks the base class behavior; each subclass test exercises the
      domain-specific validators.
    - §11 docs: this docstring cites exact commit hash (180fff3) + full
      post-audit agent-findings trace. Future contributors reading this
      file understand WHY each pattern exists.

Origin: Phase A.5.3b (2026-04-24) extraction, validated by hft-architect +
code-explorer agents. Plan: ``/Users/knight/.claude/plans/fuzzy-discovering-flask.md``
Scope D v4 (see "Phase A.5.3b" + the A.5.3a.1 hardening round summary).
"""

from __future__ import annotations

import json
from typing import Any, ClassVar, Dict, List, Optional

from pydantic import BaseModel, ConfigDict


__all__ = ["SafeBaseModel"]


class SafeBaseModel(BaseModel):
    """Shared base class for every ``lobtrainer.config.*`` Pydantic v2 model.

    Retires 4 empirically-confirmed ship-blocker-class bugs from the
    A.5.3a.1 post-audit round at the TYPE layer. Every subclass inherits
    "fail-fast by default" semantics; no per-class re-derivation needed.

    See the module docstring for the full rationale, the 4-bug empirical
    trace, and the migration checklist.

    Subclass usage::

        class MyConfig(SafeBaseModel):
            my_field: int = 0

            @model_validator(mode="after")
            def _validate_all(self) -> "MyConfig":
                if self.my_field < 0:
                    raise ValueError("my_field must be >= 0")
                return self

    The subclass DOES NOT re-declare ``model_config`` or ``model_copy`` —
    both inherited from this base. Declaring ``model_config`` on a subclass
    REPLACES (not merges) parent config per Pydantic v2 semantics, which
    would silently strip the hardening.

    **Strict mode implications**:

    Pydantic ``strict=True`` rejects:
        - ``"10"`` → ``int`` (string-to-int coercion) — ship-blocker #2
        - ``True`` → ``int`` (bool-is-int coercion) — ship-blocker #3
        - ``[1, 2]`` → ``Tuple[int, ...]`` (list-to-tuple coercion) —
          subclasses with Tuple fields that accept YAML list input MUST
          add an explicit ``@field_validator(mode="before")`` to convert
          list → tuple before the strict type check fires.

    **NaN handling caveat**:

    strict=True does NOT reject NaN/Inf floats — IEEE 754 considers them
    valid floats, and Pydantic correctly accepts them. Subclasses with
    float fields where NaN would silently break downstream logic MUST
    add explicit ``math.isfinite`` checks inside their ``@model_validator``.
    """

    # Shared configuration — DO NOT override on subclasses (replaces parent
    # config in Pydantic v2; would strip hardening patterns).
    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        strict=True,
    )

    # =========================================================================
    # Phase A.5.7a (2026-04-25) — auto-populated subclass registry
    # =========================================================================
    #
    # Replaces the hand-maintained ``_PYDANTIC_CONFIG_CLASSES`` list at
    # ``schema.py`` end-of-file (now a re-export shim). Eliminates the
    # silent-coverage-gap risk: previously, a contributor adding a new
    # ``SafeBaseModel`` subclass had to manually append it to the list,
    # otherwise parametric coverage tests (pickle / deepcopy / ClassVar
    # discipline) would silently exclude it.
    #
    # Hook choice — ``__pydantic_init_subclass__`` (NOT ``__init_subclass__``):
    # Pydantic v2 reserves ``__init_subclass__`` for its metaclass machinery
    # (per Pydantic v2 docs). Subclassing hook for user code is
    # ``__pydantic_init_subclass__``, called AFTER the class is fully
    # initialized by ``ModelMetaclass``.
    #
    # Test-fixture exclusion: the hook skips classes with leading-underscore
    # names (e.g., ``_FixtureConfig`` defined inside test methods). Every
    # production config class is PascalCase without leading underscore.
    #
    # Mutability concern: ``_registry`` is a class-level mutable list shared
    # across the SafeBaseModel hierarchy. Test code that imports schema.py
    # gets a populated registry by class-definition time; pytest's
    # ``@pytest.mark.parametrize`` evaluates registry contents at test-
    # collection time (before any test method runs), guaranteeing stable
    # parametrization over the 9 production classes.
    # =========================================================================

    _registry: ClassVar[List[type]] = []
    """Auto-populated list of concrete SafeBaseModel subclasses (production
    config classes only — test fixtures with leading-underscore names are
    excluded). Used by parametric coverage tests in tests/test_config.py
    (TestPydanticHardeningCoverageGaps) so a new class added to the
    hierarchy is automatically covered by pickle/deepcopy/ClassVar
    discipline tests with zero manual registration."""

    @classmethod
    def __pydantic_init_subclass__(cls, **kwargs: Any) -> None:
        """Auto-register concrete subclasses (Phase A.5.7a).

        Called by Pydantic's ``ModelMetaclass`` AFTER the subclass is fully
        initialized. Production config classes are PascalCase
        (LabelsConfig, ModelConfig, ...); test fixtures use
        leading-underscore names (_FixtureConfig). Only the former are
        added to the registry.

        Always calls ``super().__pydantic_init_subclass__(**kwargs)`` to
        chain into Pydantic internals (cooperative inheritance).
        """
        super().__pydantic_init_subclass__(**kwargs)
        # Skip test fixtures (leading-underscore convention). Production
        # config classes are PascalCase without leading underscore.
        if not cls.__name__.startswith("_"):
            SafeBaseModel._registry.append(cls)

    # =========================================================================
    # Phase A.5.7a (2026-04-25) — _canonical_form() SSoT
    # =========================================================================
    #
    # Single canonical-form computation feeds BOTH ``__eq__`` and ``__hash__``,
    # eliminating the order-sensitivity bug in the prior ``__hash__`` impl
    # (hash used ``repr(v)`` over sorted-key tuple; for dict-typed fields
    # like ``ModelConfig.params``, ``repr({"a":1,"b":2}) != repr({"b":2,"a":1})``
    # but ``__eq__`` via ``self.__dict__ == other.__dict__`` IS dict
    # order-insensitive — violating Python's invariant
    # ``a == b ⇒ hash(a) == hash(b)``).
    #
    # The fix: serialize via ``model_dump(mode="json")`` + ``json.dumps``
    # with ``sort_keys=True``. Pydantic's ``model_dump(mode="json")``
    # recursively converts:
    #   - Enum instances → .value strings
    #   - Tuples → JSON arrays (lists)
    #   - PrivateAttr fields → EXCLUDED (Pydantic default)
    # Combined with ``sort_keys=True``, dict insertion-order independence
    # is structural, not coincidental.
    #
    # Performance: ~50µs per ExperimentConfig serialize. Config equality
    # / hashing is NOT in any hot path; acceptable.
    #
    # PrivateAttr exclusion (Phase 4 R3 invariant) is preserved: Pydantic's
    # ``model_dump`` excludes PrivateAttr by default, so resolver caches
    # like ``DataConfig._feature_indices_resolved`` do NOT affect either
    # ``__eq__`` or ``__hash__`` — semantic identity remains driven by
    # public fields only.
    # =========================================================================

    def _canonical_form(self) -> str:
        """SSoT canonical form for both ``__eq__`` and ``__hash__``.

        Returns a JSON string with sorted keys, recursively canonicalized
        via ``model_dump(mode="json")``:

          - Enum instances → ``.value`` strings (byte-stable across
            module-load orderings)
          - Tuples → JSON arrays (matches Pydantic-strict tuple-coercion
            invariant: post-construction sequence fields are tuples, but
            JSON has no native tuple type)
          - Dicts → sorted-key JSON objects (insertion-order independent)
          - PrivateAttr → EXCLUDED (Pydantic default; matches Phase 4 R3
            invariant that resolver caches are not part of semantic identity)

        Used by ``__eq__`` for canonical-form comparison and by ``__hash__``
        for hash derivation. Both consume the SAME canonical form ⇒
        Python's ``a == b ⇒ hash(a) == hash(b)`` invariant holds by
        construction.

        Returns:
            A canonical JSON string. Same logical content always produces
            the same string regardless of dict insertion order, Enum
            module-load order, or tuple-vs-list ambiguity post-coercion.
        """
        return json.dumps(self.model_dump(mode="json"), sort_keys=True)

    def __eq__(self, other: object) -> bool:
        """Equality via canonical-form comparison (Phase A.5.7a SSoT).

        Compares the canonical JSON form (PrivateAttr-excluded, sorted-keys,
        Enum-as-string, tuple-as-list). Aligned with ``__hash__`` by
        construction — both consume ``_canonical_form()``.

        Phase 4 R3 invariant preserved: resolver-cache PrivateAttr fields
        (``DataConfig._feature_indices_resolved``, ``_feature_set_ref_resolved``)
        do NOT affect equality (Pydantic's ``model_dump`` excludes them).

        Args:
            other: Object to compare. Must be the same concrete subclass —
                cross-class comparison returns NotImplemented per Python
                convention; Python falls back to ``other.__eq__(self)``
                which returns NotImplemented in turn ⇒ False.

        Returns:
            True iff ``other`` is the same subclass AND its canonical form
            equals ``self``'s canonical form.
        """
        if type(self) is not type(other):
            return NotImplemented
        return self._canonical_form() == other._canonical_form()  # type: ignore[attr-defined]

    def __hash__(self) -> int:
        """Hash via canonical form (Phase A.5.7a SSoT).

        Closes the order-sensitivity bug in the prior implementation:
        previously hashed ``repr(v)`` over sorted-key tuple, which produced
        DIFFERENT hashes for dicts with same content but different
        insertion order — while ``__eq__`` (via ``__dict__ ==``) returned
        True. This violated Python's ``a == b ⇒ hash(a) == hash(b)``
        invariant.

        Post-A.5.7a: hash IS the canonical form's hash. Aligned with
        ``__eq__`` by SSoT construction.
        """
        return hash(self._canonical_form())

    def model_copy(  # type: ignore[override]
        self,
        *,
        update: Optional[Dict[str, Any]] = None,
        deep: bool = False,
    ) -> "SafeBaseModel":
        """Override Pydantic v2's ``model_copy`` to re-run validators on update.

        Closes ship-blocker bug #1 (Phase A.5.3a.1 post-audit): default
        Pydantic v2 ``model_copy(update={...})`` uses ``model_construct``
        internally which SKIPS ALL validation. An operator writing
        ``cfg.model_copy(update={"source": "bogus"})`` would produce an
        invalid config with no ValidationError — re-opening bug class #3
        (extra-field / invalid-value acceptance) through a common Pydantic
        idiom that operators naturally reach for.

        This override forces re-validation via ``model_validate`` when
        update is provided — closes the validator-bypass vector at the
        structural level (callers don't need to remember to use
        ``model_validate({**dump(), **overrides})`` idiom).

        Trade-off: slight perf cost on ``model_copy(update=...)`` (full
        validation vs fast construct). Acceptable for config classes that
        are rarely copy-mutated in hot paths.

        Args:
            update: Dict of field overrides. If non-empty, triggers full
                re-validation via ``model_validate``. If None or empty,
                delegates to ``super().model_copy`` (fast path — pure copy,
                no user data to validate).
            deep: Deep copy flag, passed through to super() on no-update
                path.

        Returns:
            A new instance of the concrete subclass with update applied
            and all validators re-fired.

        Raises:
            pydantic.ValidationError: when ``update`` contains invalid
                field values (same failure mode as direct construction).
        """
        if update:
            return self.__class__.model_validate(
                {**self.model_dump(), **update}
            )
        return super().model_copy(deep=deep)
