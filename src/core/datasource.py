"""Datasource module"""

from dataclasses import dataclass
from typing import Any, Dict

from pydantic import Field

from src.core.base import BaseConfiguration


class BaseDatasourceConfiguration(BaseConfiguration):
    """Configuration specific to a datasource."""

    config_path: str = Field(description="Full path to the object's configuration class.")
    config_init_params: Dict[str, Any] = Field(
        default_factory=dict, description="Initialization parameters for the config class."
    )
    metadata_path: str = Field(description="Full path to the metadata class.")
    metadata_params: Dict[str, Any] = Field(default_factory=dict, description="Initialization parameters for metadata.")
