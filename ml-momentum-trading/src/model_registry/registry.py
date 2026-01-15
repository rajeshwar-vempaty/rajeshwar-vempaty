"""
Model Registry
Manages model versions, metadata, and feature schemas
"""

import os
import json
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class ModelRegistry:
    """
    Version-controlled model storage with metadata
    """

    def __init__(self, registry_path: str = "model_registry"):
        """
        Initialize model registry

        Args:
            registry_path: Path to registry directory
        """
        self.registry_path = Path(registry_path)
        self.registry_path.mkdir(parents=True, exist_ok=True)
        self.metadata_file = self.registry_path / "registry_metadata.json"

        # Load or initialize metadata
        self.metadata = self._load_metadata()

    def _load_metadata(self) -> Dict:
        """Load registry metadata"""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        return {"models": {}, "latest_version": None}

    def _save_metadata(self):
        """Save registry metadata"""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2, default=str)

    def register_model(
        self,
        model_path: str,
        model_name: str,
        version: Optional[str] = None,
        metrics: Optional[Dict] = None,
        feature_schema: Optional[List[str]] = None,
        hyperparameters: Optional[Dict] = None,
        notes: str = ""
    ) -> str:
        """
        Register a new model version

        Args:
            model_path: Path to model file
            model_name: Model name/identifier
            version: Optional version string (auto-generated if None)
            metrics: Model performance metrics
            feature_schema: List of feature column names
            hyperparameters: Model hyperparameters
            notes: Additional notes

        Returns:
            Version string
        """
        # Auto-generate version if not provided
        if version is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            version = f"v_{timestamp}"

        # Create version directory
        version_path = self.registry_path / model_name / version
        version_path.mkdir(parents=True, exist_ok=True)

        # Copy model file
        model_dest = version_path / "model.txt"  # LightGBM saves as .txt
        shutil.copy(model_path, model_dest)

        # Save metadata
        version_metadata = {
            "model_name": model_name,
            "version": version,
            "created_at": datetime.now().isoformat(),
            "model_path": str(model_dest),
            "metrics": metrics or {},
            "feature_schema": feature_schema or [],
            "hyperparameters": hyperparameters or {},
            "notes": notes
        }

        # Save version-specific metadata
        with open(version_path / "metadata.json", 'w') as f:
            json.dump(version_metadata, f, indent=2, default=str)

        # Update registry
        if model_name not in self.metadata["models"]:
            self.metadata["models"][model_name] = {}

        self.metadata["models"][model_name][version] = version_metadata
        self.metadata["latest_version"] = version

        self._save_metadata()

        logger.info(f"Registered model {model_name} version {version}")
        return version

    def get_model_path(self, model_name: str, version: Optional[str] = None) -> Path:
        """
        Get path to model file

        Args:
            model_name: Model name
            version: Version (uses latest if None)

        Returns:
            Path to model file
        """
        if version is None:
            version = self.get_latest_version(model_name)

        if model_name not in self.metadata["models"]:
            raise ValueError(f"Model {model_name} not found in registry")

        if version not in self.metadata["models"][model_name]:
            raise ValueError(f"Version {version} not found for model {model_name}")

        return Path(self.metadata["models"][model_name][version]["model_path"])

    def get_metadata(self, model_name: str, version: Optional[str] = None) -> Dict:
        """
        Get model metadata

        Args:
            model_name: Model name
            version: Version (uses latest if None)

        Returns:
            Metadata dict
        """
        if version is None:
            version = self.get_latest_version(model_name)

        if model_name not in self.metadata["models"]:
            raise ValueError(f"Model {model_name} not found in registry")

        if version not in self.metadata["models"][model_name]:
            raise ValueError(f"Version {version} not found for model {model_name}")

        return self.metadata["models"][model_name][version]

    def get_latest_version(self, model_name: str) -> str:
        """
        Get latest version for a model

        Args:
            model_name: Model name

        Returns:
            Latest version string
        """
        if model_name not in self.metadata["models"]:
            raise ValueError(f"Model {model_name} not found in registry")

        versions = list(self.metadata["models"][model_name].keys())
        if not versions:
            raise ValueError(f"No versions found for model {model_name}")

        # Return most recent by creation time
        return max(
            versions,
            key=lambda v: self.metadata["models"][model_name][v]["created_at"]
        )

    def list_models(self) -> List[str]:
        """List all registered models"""
        return list(self.metadata["models"].keys())

    def list_versions(self, model_name: str) -> List[str]:
        """
        List all versions of a model

        Args:
            model_name: Model name

        Returns:
            List of version strings
        """
        if model_name not in self.metadata["models"]:
            raise ValueError(f"Model {model_name} not found in registry")

        return list(self.metadata["models"][model_name].keys())

    def compare_versions(self, model_name: str, versions: List[str], metric: str = "val_auc") -> pd.DataFrame:
        """
        Compare metrics across versions

        Args:
            model_name: Model name
            versions: List of versions to compare
            metric: Metric to compare

        Returns:
            Comparison DataFrame
        """
        import pandas as pd

        comparison = []
        for version in versions:
            metadata = self.get_metadata(model_name, version)
            metrics = metadata.get("metrics", {})
            comparison.append({
                "version": version,
                "created_at": metadata["created_at"],
                metric: metrics.get(metric, None)
            })

        return pd.DataFrame(comparison).sort_values("created_at")
