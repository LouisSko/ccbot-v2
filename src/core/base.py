# everything should have a base configuration which allows to reload it easily based on a json file


from abc import ABC, abstractmethod

from pydantic import BaseModel, Field, field_validator


class ObjectId(BaseModel):
    """Every object should get a referenceble unique id."""

    value: str = Field(description="ID of the object")


class BaseConfiguration(BaseModel):
    """Base configuration for all objects."""

    component_type: str = Field(description="Type of the configuration (e.g., model, processor).")
    object_id: ObjectId = Field(description="ID of the component to reference it.")
    resource_path: str = Field(description="Full path to the object's class or function.")
    config_path: str = Field(description="Import path of the Configuration")
    settings_path: str = Field(description="Full path to the settings parameter.")

    @field_validator(
        "component_type",
        mode="before",
    )
    def check_component_type(cls, value):  # pylint: disable=no-self-argument
        """check if timestamps are set correctly."""

        component_types = ["datasource", "processor", "merger", "model", "ensemble"]

        if value not in component_types:
            raise ValueError(f"{value} must be on of the following components: {component_types}")
        return value


class BasePipelineComponent(ABC):
    """Abstract class which is the parent class of all components of a pipeline."""

    @abstractmethod
    def create_configuration(self) -> BaseConfiguration:
        """Returns the configuration of the class which can later be loaded."""

        pass
