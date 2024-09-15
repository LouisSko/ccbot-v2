# everything should have a base configuration which allows to reload it easily based on a json file


from abc import ABC, abstractmethod

from pydantic import BaseModel, Field


class ObjectId(BaseModel):
    """Every object should get a referenceble unique id."""

    value: str = Field(description="ID of the object")


class BaseConfiguration(BaseModel):
    """Base configuration for all objects."""

    config_type: str = Field(description="Type of the configuration (e.g., model, processor).")
    object_id: ObjectId = Field(description="ID of the component to reference it.")
    resource_path: str = Field(description="Full path to the object's class or function.")


class BasePipelineComponent(ABC):
    """Abstract class which is the parent class of all components of a pipeline."""

    @abstractmethod
    def create_configuration(self) -> BaseConfiguration:
        """Returns the configuration of the class which can later be loaded."""

        pass
