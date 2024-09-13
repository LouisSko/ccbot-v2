"""Preprocessor module"""

from typing import Any, Dict, List, Union

from pydantic import Field

from src.core.base import BaseConfiguration, ObjectId


class PreprocessorConfiguration(BaseConfiguration):
    """Configuration for preprocessors that are classifiers with init parameters."""

    config_type: str = Field(
        default="preprocessor",
        Literal=True,
        description="Type of the configuration (e.g., model, preprocessor). Do not change this value",
    )
    depends_on: Union[ObjectId, List[ObjectId]] = Field(
        description="ID(s) of the datasource or preprocessors this preprocessor depends on."
    )
    init_params: Dict[str, Any] = Field(
        default_factory=dict, description="Initialization parameters for the preprocessor."
    )


class MergePreprocessorConfiguration(BaseConfiguration):
    """Configuration for preprocessors that merge other preprocessors."""

    config_type: str = Field(
        default="merge_preprocessor",
        Literal=True,
        description="Type of the configuration (e.g., model, preprocessor). Do not change this value",
    )

