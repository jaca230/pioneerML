from __future__ import annotations

from typing import Any

from pioneerml.models.classifiers.group_classifier import GroupClassifier


def build_model(**kwargs: Any) -> GroupClassifier:
    defaults = dict(hidden=192, heads=4, num_blocks=3, dropout=0.1, num_classes=3)
    defaults.update(kwargs)
    return GroupClassifier(**defaults)
