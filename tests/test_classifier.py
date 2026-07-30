"""Обязательные проверки из критериев приёмки:
короб -> B, лежащий цилиндр -> D, шар -> D, капсула -> D, пластина 1.5мм -> C,
короб 600мм -> C, повёрнутый короб -> B."""
from __future__ import annotations

import pytest

from core.classifier import classify
from core.types import Category, Config


@pytest.fixture
def config() -> Config:
    return Config()  # k_threshold=0.7 по умолчанию — ровно как в ТЗ (текстовое значение)


@pytest.mark.parametrize(
    "mesh_name,expected",
    [
        ("box_300x200x150", Category.B),
        ("cylinder_lying_r60_l300", Category.D),
        ("sphere_r80", Category.D),
        ("capsule_r50_h200", Category.D),
        ("plate_1_5mm", Category.C),
        ("box_oversize_600", Category.C),
        ("box_rotated", Category.B),
    ],
)
def test_classification_matches_expected_category(synthetic_meshes, config, mesh_name, expected):
    result = classify(synthetic_meshes[mesh_name], config)
    assert result.category == expected, result.reason


def test_undersize_dimension_is_category_c(config):
    import trimesh
    tiny = trimesh.creation.box(extents=[5, 5, 5])  # < 10x10x2 мм минимума
    result = classify(tiny, config)
    assert result.category == Category.C


def test_dimension_check_skips_shape_check(synthetic_meshes, config):
    """Категория C по габаритам не должна вычислять K вообще (порядок проверок из ТЗ)."""
    result = classify(synthetic_meshes["box_oversize_600"], config)
    assert result.category == Category.C
    assert result.k_value is None


def test_k_threshold_is_configurable_not_hardcoded(synthetic_meshes):
    """ТЗ противоречит само себе (текст 0.7, иллюстрация 0.8) — порог обязан быть
    параметром, а не константой в classifier.py."""
    mesh = synthetic_meshes["hexagon_prism_r100_l300"]  # K ~= 0.866

    strict = classify(mesh, Config(k_threshold=0.9))
    assert strict.category == Category.B  # 0.866 <= 0.9 -> не круглый по этому порогу

    lenient = classify(mesh, Config(k_threshold=0.5))
    assert lenient.category == Category.D  # 0.866 > 0.5 -> круглый

    default = classify(mesh, Config())  # default 0.7
    assert default.category == Category.D
