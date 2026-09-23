"""
Tests for the shared path-resolution module.
"""
from pathlib import Path
from arcefn.utils.paths import (
    get_project_root,
    ensure_dir,
    _PROJECT_ROOT,
    DATA_DIR,
    EXPERIMENTS,
    SCRIPTS_DIR,
    MODEL_DIR,
    CHECKPOINT,
)


def test_get_project_root_returns_path():
    root = get_project_root()
    assert isinstance(root, Path)
    assert root.is_dir()


def test_project_root_contains_setup():
    root = get_project_root()
    assert (root / "setup.py").exists()


def test_project_root_contains_arcefn():
    root = get_project_root()
    assert (root / "arcefn").is_dir()


def test_data_dir_is_subdir_of_project():
    assert DATA_DIR == _PROJECT_ROOT / "data" / "top_tagging"


def test_experiments_dir_is_subdir_of_project():
    assert EXPERIMENTS == _PROJECT_ROOT / "experiments"


def test_scripts_dir_is_subdir_of_project():
    assert SCRIPTS_DIR == _PROJECT_ROOT / "scripts"


def test_model_dir_under_experiments():
    assert MODEL_DIR == EXPERIMENTS / "robustness_s16_m05"


def test_checkpoint_path():
    assert CHECKPOINT == MODEL_DIR / "checkpoint.pt"


def test_ensure_dir_creates(tmp_path):
    d = tmp_path / "a" / "b" / "c"
    assert not d.exists()
    result = ensure_dir(d)
    assert d.is_dir()
    assert result == d


def test_ensure_dir_idempotent(tmp_path):
    d = tmp_path / "existing"
    d.mkdir()
    result = ensure_dir(d)
    assert d.is_dir()
    assert result == d
