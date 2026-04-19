"""Tests for :mod:`pipeline.seam_measurement.params`.

Covers both the YAML-backed default path and the built-in fallback path
so either can be trusted when regressions touch the parameter surface.
"""

from __future__ import annotations

import importlib
import sys
from dataclasses import fields
from pathlib import Path

import pytest

from pipeline.seam_measurement import params as params_module
from pipeline.seam_measurement.params import GapFlushParams


def test_every_field_has_a_builtin_default():
    """All GapFlushParams fields must have a matching _BUILTIN_DEFAULTS key."""
    builtin = params_module._BUILTIN_DEFAULTS
    dataclass_fields = {field.name for field in fields(GapFlushParams)}
    missing_in_builtin = dataclass_fields - set(builtin)
    extra_in_builtin = set(builtin) - dataclass_fields
    assert not missing_in_builtin, f"fields without built-in defaults: {missing_in_builtin}"
    assert not extra_in_builtin, f"built-in defaults without matching field: {extra_in_builtin}"


def test_default_construction_matches_builtin_values():
    """Default GapFlushParams() should read its values from _CFG (yaml or builtins)."""
    instance = GapFlushParams()
    for name, expected in params_module._CFG.items():
        actual = getattr(instance, name)
        assert float(actual) == pytest.approx(float(expected)), name


def test_shipped_yaml_matches_builtin_defaults():
    """configs/gap_flush.yaml should not silently drift from the safety-net."""
    yaml = pytest.importorskip("yaml")
    config_path = Path(params_module._CONFIG_PATH)
    with config_path.open("r", encoding="utf-8") as handle:
        shipped = yaml.safe_load(handle)
    for key, builtin_value in params_module._BUILTIN_DEFAULTS.items():
        assert key in shipped, f"yaml missing key {key}"
        assert float(shipped[key]) == pytest.approx(float(builtin_value)), key


def test_missing_yaml_falls_back_to_builtins(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Loader should not raise when the yaml file is missing."""
    missing = tmp_path / "does_not_exist.yaml"
    result = params_module._load_yaml_defaults(missing)
    assert result == params_module._BUILTIN_DEFAULTS


def test_yaml_override_is_applied(tmp_path: Path):
    """Values present in yaml should override the built-in defaults."""
    pytest.importorskip("yaml")
    custom = tmp_path / "override.yaml"
    custom.write_text(
        "seam_step: 7\n"
        "section_half_length_px: 5.5\n",
        encoding="utf-8",
    )
    merged = params_module._load_yaml_defaults(custom)
    assert merged["seam_step"] == 7
    assert merged["section_half_length_px"] == 5.5
    # Non-overridden keys remain at their built-in values.
    assert merged["min_section_points"] == params_module._BUILTIN_DEFAULTS["min_section_points"]


def test_unknown_yaml_keys_are_ignored_not_fatal(tmp_path: Path):
    """Typos in the yaml should log-warn but not crash the loader."""
    pytest.importorskip("yaml")
    custom = tmp_path / "with_unknown.yaml"
    custom.write_text("seam_step: 3\ntotally_bogus_key: 42\n", encoding="utf-8")
    merged = params_module._load_yaml_defaults(custom)
    assert merged["seam_step"] == 3
    assert "totally_bogus_key" not in merged


def test_caller_override_still_works():
    """GapFlushParams(seam_step=...) keeps working after the yaml refactor."""
    instance = GapFlushParams(seam_step=99)
    assert instance.seam_step == 99


def _reload_params_module():
    # Ensures any monkeypatched state from earlier tests does not leak.
    return importlib.reload(sys.modules[params_module.__name__])
