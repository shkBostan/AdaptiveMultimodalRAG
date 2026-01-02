# src/data/mscoco_dataset.py

import json
from pathlib import Path
from typing import Dict, Any, List

from PIL import Image

from .base_dataset import BaseDataset


class MSCOCODataset(BaseDataset):
    """
    MSCOCO Dataset Adapter.

    This class adapts MSCOCO annotations and images into a unified
    multimodal sample format compatible with embedding and retrieval pipelines.
    """

    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        annotation_file: str = None,
        image_dir: str = None
    ):
        """
        Args:
            root_dir (str): Path to processed MSCOCO directory
            split (str): Dataset split (train / val / test)
            annotation_file (str): Optional custom annotation file
            image_dir (str): Optional custom image directory
        """

        self.root_dir = Path(root_dir)
        self.split = split

        self.annotation_file = (
            Path(annotation_file)
            if annotation_file
            else self.root_dir / "annotations" / f"captions_{split}.json"
        )

        self.image_dir = (
            Path(image_dir)
            if image_dir
            else self.root_dir / "images" / split
        )

        self._load_annotations()
        self._build_index()

    def _load_annotations(self) -> None:
        """Load COCO annotation JSON file."""
        if not self.annotation_file.exists():
            raise FileNotFoundError(
                f"Annotation file not found: {self.annotation_file}"
            )

        with open(self.annotation_file, "r", encoding="utf-8") as f:
            self.coco_data = json.load(f)

    def _build_index(self) -> None:
        """
        Build internal indices for fast lookup.

        Creates:
        - image_id -> image info
        - samples list mapping captions to images
        """

        self.image_index: Dict[int, Dict[str, Any]] = {
            img["id"]: img for img in self.coco_data["images"]
        }

        self.samples: List[Dict[str, Any]] = []

        for ann in self.coco_data["annotations"]:
            image_info = self.image_index.get(ann["image_id"])
            if image_info is None:
                continue

            self.samples.append({
                "sample_id": f"mscoco_{ann['id']}",
                "image_id": ann["image_id"],
                "caption_id": ann["id"],
                "caption": ann["caption"],
                "file_name": image_info["file_name"]
            })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Return a unified multimodal sample.

        Returns:
            {
              "id": str,
              "image": PIL.Image.Image,
              "text": str,
              "meta": dict
            }
        """

        sample = self.samples[idx]

        image_path = self.image_dir / sample["file_name"]
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        image = Image.open(image_path).convert("RGB")

        return {
            "id": sample["sample_id"],
            "image": image,
            "text": sample["caption"],
            "meta": {
                "image_id": sample["image_id"],
                "caption_id": sample["caption_id"],
                "file_name": sample["file_name"],
                "split": self.split
            }
        }
