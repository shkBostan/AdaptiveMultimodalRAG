"""
Author: s Bostan
Created on: Jan, 2026
Description: Test suite for MS COCO dataset loader and data validation.
"""

import pytest
from PIL import Image


def test_dataset_not_empty(mscoco_dataset):
    assert len(mscoco_dataset) > 0


def test_sample_schema(mscoco_dataset):
    sample = mscoco_dataset[0]
    assert set(sample.keys()) == {"id", "image", "text", "meta"}


def test_sample_types(mscoco_dataset):
    sample = mscoco_dataset[0]
    assert isinstance(sample["image"], Image.Image)
    assert isinstance(sample["text"], str)
    assert isinstance(sample["meta"], dict)


def test_out_of_range_index(mscoco_dataset):
    with pytest.raises(IndexError):
        _ = mscoco_dataset[len(mscoco_dataset)]
