# src/data/base_dataset.py

from abc import ABC, abstractmethod
from typing import Dict, Any


class BaseDataset(ABC):
    """
    Abstract base class for all datasets.

    A dataset is responsible for:
    - Loading raw or processed data
    - Providing unified multimodal samples
    - Being iterable and indexable

    Each sample MUST follow the unified schema:

    {
        "id": str,
        "image": PIL.Image.Image | None,
        "text": str | None,
        "meta": dict
    }
    """

    @abstractmethod
    def __len__(self) -> int:
        """Return the total number of samples."""
        pass

    @abstractmethod
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Return a single sample.

        Args:
            idx (int): Sample index

        Returns:
            dict: Unified multimodal sample
        """
        pass

    def get_sample_schema(self) -> Dict[str, str]:
        """
        Returns the expected schema of a dataset sample.
        Useful for validation and documentation.
        """
        return {
            "id": "Unique sample identifier",
            "image": "PIL.Image.Image or None",
            "text": "Textual content or None",
            "meta": "Additional metadata dictionary"
        }
