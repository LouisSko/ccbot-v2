"""Module which creates the pipeline."""

import json
import os
from abc import ABC, abstractmethod
from importlib import import_module
from typing import Dict, List, Optional, Type, Union

import pandas as pd
from pydantic import BaseModel, ConfigDict, TypeAdapter

from src.common_packages import CustomJSONEncoder, create_logger, timestamp_decoder
from src.core.base import BaseConfiguration, BasePipelineComponent
from src.core.datasource import Data, Datasource
from src.core.engine import TradeSignal
from src.core.model import Model, Prediction, TrainingInformation

logger = create_logger(
    log_level=os.getenv("LOGGING_LEVEL", "INFO"),
    logger_name=__name__,
)

PIPELINE_JSON_FILE_NAME = "pipeline_config.json"


class PipelineSettings(BaseModel):
    """Create a Pipeline based on a list of BaseConfiguration"""

    components: Union[List[BaseConfiguration], List[BasePipelineComponent]]
    save_dir: str  # directory where the pipeline is getting saved
    timeframe: pd.Timedelta  # execution timeframe of the pipeline
    test_interval_length: Optional[pd.Timedelta] = (
        None  # length of the test set for evaluating the pipeline. If none is specified, use all data for training
    )

    model_config = ConfigDict(arbitrary_types_allowed=True)


class Pipeline(ABC):
    """Pipeline class"""

    def __init__(self, config: PipelineSettings):

        self.config: PipelineSettings = config
        self.last_training_date: pd.Timestamp = None
        self.simulation_mode: bool = None
        self.components: dict = self._initialize_components(self.config)
        self.execution_order, self.dependency_structure = self._get_execution_order()

    # @abstractmethod
    # def _generate_trade_signals(self, result: Dict[str, List[Prediction]]) -> TradeSignal:
    #     """generate trade signals based on predictions."""

    #     pass

    def trigger(self) -> List[TradeSignal]:
        """Triggers the pipeline."""

        logger.info("Trigger Pipeline End-to-End...")
        result = {component_id: None for component_id in self.execution_order}

        # go through the data pipeline step by step.
        for component_id in self.execution_order:

            component = self.components.get(component_id).get("instance")
            component_dependencies = self.dependency_structure[component_id]
            component_type = self.components[component_id]["config"].component_type

            if component_type == "datasource":
                result[component_id] = component.scrape_data_current()
            elif component_type == "processor":
                result[component_id] = component.process(result[component_dependencies])
            elif component_type == "merger":
                result[component_id] = component.merge_data([result[dep] for dep in component_dependencies])
            elif component_type == "model":
                result[component_id] = component.predict(result[component_dependencies])  # train the model

        # trade_signals = self._generate_trade_signals(result)

        final_predictions: List[Prediction] = result[self.execution_order[-1]]

        trade_signals = []

        for prediction in final_predictions:

            for time, pred in zip(prediction.time, prediction.prediction):

                if pred == 1:
                    position_side = "buy"
                    # limit_price = last_price * (1 + reduction)
                elif pred == -1:
                    position_side = "sell"
                else:
                    continue

                # TODO: add limit order and stop loss / take profit functionality.
                # -> for that I need the current close price
                # Create a TradeSignal for this prediction
                trade_signal = TradeSignal(
                    time=time,
                    symbol=prediction.symbol,
                    order_type="market",  # or limit
                    position_side=position_side,
                    # limit_price=10000 if position_side == "buy" else 11000,  # Example logic for setting limit price
                    # stop_loss_price=9500 if position_side == "buy" else 11500,  # Stop loss for risk management
                    # take_profit_price=12000 if position_side == "buy" else 9000,  # Take profit target
                )

                trade_signals.append(trade_signal)

        logger.info("Pipeline execution completed.")

        # return the final result of the pipeline
        return trade_signals

    def train(self) -> None:
        """Train all models in the pipeline."""

        logger.info("Start Training Pipeline End-to-End...")

        # get current date, substract x days as starting point for testing
        if self.config.test_interval_length:
            train_test_split_date = self.get_current_date() - self.config.test_interval_length
            logger.info("Train-test-split is at %s", train_test_split_date)
        else:
            train_test_split_date = None
            logger.info("Use all data for training.")

        result = {component_id: None for component_id in self.execution_order}

        # now go through the data pipeline step by step.
        for component_id in self.execution_order:

            component: BasePipelineComponent = self.components.get(component_id).get("instance")
            component_dependencies: List[str] = self.dependency_structure[component_id]
            component_type: str = self.components[component_id]["config"].component_type

            if component_type == "datasource":
                result[component_id] = component.scrape_data_historic()
            elif component_type == "processor":
                result[component_id] = component.process(result[component_dependencies])
            elif component_type == "merger":
                result[component_id] = component.merge_data([result[dep] for dep in component_dependencies])
            elif component_type == "model":
                if train_test_split_date:
                    result[component_id] = component.train_and_evaluate(
                        result[component_dependencies], train_test_split_date
                    )
                else:
                    result[component_id] = component.train(result[component_dependencies])
                component.save()

        self.last_training_date = self.get_last_training_date()

        logger.info("Pipeline Training completed.")

        self.export_pipeline()  # save the new training information

    def get_last_training_date(self):
        """Function to determine the last date the pipeline got trained.

        This is done based on the information of the different models.
        Alternatively the current time at which the pipeline got trained could be used,
        but this results in problems when backtesting.
        """

        logger.info("Determine last training date for all models.")

        train_end_dates = []

        for component_id in self.execution_order:

            component_type: str = self.components[component_id]["config"].component_type

            if component_type == "model":

                component: Model = self.components.get(component_id).get("instance")
                training_information: TrainingInformation = component.config.training_information

                if training_information is None:
                    raise ValueError(
                        "None of the models have an associated training_information. Most likely the models have not been trained yet. Check pipeline training."
                    )

                train_end_dates.append(training_information.train_end_date)

        # Ensure all models have the same training date
        if not train_end_dates:
            raise ValueError("No models found in the pipeline to determine the last training date.")

        unique_dates = set(train_end_dates)

        if len(unique_dates) != 1:
            raise ValueError(f"Multiple different last training dates found: {unique_dates}. Please check.")

        last_train_date = unique_dates.pop()

        logger.info("Last training date was: %s", last_train_date)

        return last_train_date

    def get_symbols(self) -> List[str]:
        """Function to get all the symbols which were used during training."""

        list_symbols = []

        # extract all symbols which were used for training
        for component_id in self.execution_order:

            component_type: str = self.components[component_id]["config"].component_type

            if component_type == "datasource":
                component: Datasource = self.components.get(component_id).get("instance")

                if component.config.symbols is not None:
                    list_symbols.extend(component.config.symbols)

        logger.info("Determined the symbols which were used for training: %s", list_symbols)

        return list_symbols

    def create_and_save_mock_data(self):
        """Function to create mock data and to store it in the config directory based on datasources.

        The scrape_start_date and scrape_end_date is used to fetch the mock data.
        """

        logger.info("Create mock data for all datasources.")

        for component_id in self.execution_order:

            component_type: str = self.components[component_id]["config"].component_type

            if component_type == "datasource":

                component: Datasource = self.components.get(component_id).get("instance")
                mock_data: Data = component.create_mock_data()
                component.save_mock_data(mock_data)

        logger.info("Mock data got created successfully for all datasources.")
        self.export_pipeline()  # update pipeline

    def activate_simulation_mode(self):
        """Activate simulation mode

        for each datasource it sets it to simulation mode
        """

        # go through the data pipeline step by step.
        for component_id in self.execution_order:

            component_type: str = self.components[component_id]["config"].component_type

            if component_type == "datasource":
                component: Datasource = self.components.get(component_id).get("instance")
                component.use_mock_data = True
                component.set_simulation_mode()

        self.simulation_mode = True

    def get_current_date(self) -> Optional[pd.Timestamp]:
        """get current date based on the datasources.

        This might be prone to errors in simulation mode when we have multiple datasources.


        Return:
            Optional[pd.Timestamp]: Eiter the timestamp of the current simulation step or None in case no more simulation data is available
        """

        if self.simulation_mode:

            current_date = []
            end_date = []

            for component_id in self.execution_order:

                component_type: str = self.components[component_id]["config"].component_type

                if component_type == "datasource":
                    current_date.append(self.components[component_id]["instance"].simulation_current_date)
                    end_date.append(self.components[component_id]["instance"].config.simulation_end_date)

            if max(current_date) > min(end_date):
                return None

            return max(current_date)

        # otherwise return the current time
        return pd.Timestamp.now(tz="utc")

    def deactivate_simulation_mode(self):
        """Deactivate simulation mode

        for each datasource it deactivates simulation mode
        """

        # go through the data pipeline step by step.
        for component_id in self.execution_order:

            component_type: str = self.components[component_id]["config"].component_type

            if component_type == "datasource":
                component: Datasource = self.components.get(component_id).get("instance")
                component.use_mock_data = False

                logger.info("Simulation mode deactivated for component: %s", component_id)

        self.simulation_mode = False

    def _get_execution_order(self) -> List[str]:
        """Determine the topological order of the components based on dependencies.

        In this sorting, datsources are not necessarily placed at the start. In general this is not a problem,
        but for calculating the the train test splits, all data needs to be fetched first.

        A topological sort is a graph traversal in which each node v is visited only after all its dependencies are visited.
        Here Kahn's algorithm is used to determine the correct order.
        Node: represent pipeline component
        Edge: represent dependency

        Returns:
            (List(str)): list of ordered pipeline components.
                e.g. ['exchange', 'target_processor', 'feature_processor', 'merger', 'LGBM_Model']
        """

        # Build the dependency graph
        dependency_graph: Dict[str, List[str]] = {}
        dependency_structure: Dict[str, List[str]] = {}
        indegree: Dict[str, int] = {}

        # initialize graph nodes
        for component_id in self.components:
            dependency_graph[component_id] = []
            dependency_structure[component_id] = None
            indegree[component_id] = 0

        for component_id, component_data in self.components.items():

            component_config = component_data["config"]
            depends_on = getattr(component_config.settings, "depends_on", None)

            # Add edges based on dependencies
            if isinstance(depends_on, list):

                dependency_structure[component_id] = []

                for dep in depends_on:
                    dep_id = dep.value
                    dependency_graph[dep_id].append(component_id)
                    dependency_structure[component_id].append(dep_id)
                    indegree[component_id] = indegree.get(component_id, 0) + 1
            elif depends_on:
                dep_id = depends_on.value
                dependency_graph[dep_id].append(component_id)
                dependency_structure[component_id] = dep_id
                indegree[component_id] = indegree.get(component_id, 0) + 1

        # Perform topological sort (Kahn's algorithm)
        execution_order = []
        zero_indegree = [node for node in indegree if indegree[node] == 0]

        # does that approach even work when I have more than one datasource?
        while zero_indegree:
            current = zero_indegree.pop(0)
            execution_order.append(current)

            for neighbor in dependency_graph[current]:
                indegree[neighbor] -= 1
                if indegree[neighbor] == 0:
                    zero_indegree.append(neighbor)

        if len(execution_order) != len(self.components):
            raise ValueError("The pipeline has circular dependencies!")

        # sort the dependency structure according to the execution order just for nicer visualization
        dependency_structure = {k: dependency_structure[k] for k in execution_order}

        logger.info("Topological Search found dependency graph %s", dependency_graph)
        logger.info("Execution order determined: %s", execution_order)
        return execution_order, dependency_structure

    def export_pipeline(self) -> None:
        """Export Pipeline component configurations to a json file based on the specified save directory"""

        logger.info("Saving Pipeline configuration to files...")
        file_path = os.path.join(self.config.save_dir, PIPELINE_JSON_FILE_NAME)

        # Create the directory if it doesn't exist
        if not os.path.exists(self.config.save_dir):
            logger.info("Specified directory does not exist. Create dir: %s", self.config.save_dir)
            os.makedirs(self.config.save_dir)

        data = []

        # add pipeline information
        pipeline_cls = f"{self.__module__}.{self.__class__.__name__}"
        data.append(
            {
                "pipeline_path": pipeline_cls,
                "timeframe": self.config.timeframe,
                "last_training_date": self.last_training_date,
                "test_interval_length": self.config.test_interval_length,
            }
        )  # TODO: might change the last_training_date in the future. Because now we depend on loading in the pipeline. Cannot do

        for component in self.config.components:

            if isinstance(component, BasePipelineComponent):
                data.append(component.create_configuration().model_dump())

            elif isinstance(component, BaseConfiguration):
                data.append(component.model_dump())

        with open(file_path, "w", encoding="utf8") as file:
            json.dump(data, file, indent=4, cls=CustomJSONEncoder)

        logger.info("Pipeline configuration saved to %s. It can be reloaded using 'load_pipeline()'", file_path)

    def _initialize_components(self, config: PipelineSettings):
        """Initialize all pipeline components."""

        components = {}
        for component in config.components:

            if isinstance(component, BasePipelineComponent):
                components[component.config.object_id.value] = {
                    "instance": component,
                    "config": component.create_configuration(),
                }

            elif isinstance(component, BaseConfiguration):
                components[component.object_id.value] = {
                    "instance": _load_component_from_configuration(component),
                    "config": component,
                }

            else:
                raise ValueError("component needs to be of type 'BasePipelineComponent' or 'BaseConfiguration'")

        logger.info("Pipeline initalized with the following components: %s", components)

        return components


def load_pipeline(save_directory: str) -> Pipeline:
    """Create a pipeline based on a stored pipeline component configuration json
    Args:
        save_directory (str): The path of the directory.
    """

    file_path = os.path.join(save_directory, PIPELINE_JSON_FILE_NAME)

    if not os.path.exists(file_path):
        raise ValueError("File path does not exist")

    with open(file_path, "r", encoding="utf8") as f:
        config = json.load(f, object_hook=timestamp_decoder)

    # first entry of the config is general information regarding the pipeline
    pipeline_info = config.pop(0)
    module_path, class_name = pipeline_info["pipeline_path"].rsplit(".", 1)
    module = import_module(module_path)
    pipeline_class: Pipeline = getattr(module, class_name)

    config = PipelineSettings(
        components=[_configuration_factory(config_dict) for config_dict in config],
        timeframe=pipeline_info["timeframe"],
        save_dir=save_directory,
        test_interval_length=pipeline_info["test_interval_length"],
    )

    loaded_pipeline: Pipeline = pipeline_class(config=config)

    # set the information whether the pipeline is trained or not.
    loaded_pipeline.last_training_date = pipeline_info["last_training_date"]

    return loaded_pipeline


def _configuration_factory(config_dict: dict) -> BaseConfiguration:
    """Factory method to create the appropriate configuration object.

    Args:
        config_dict (dict): The configuration dictionary.

    Returns:
        BaseConfiguration: The appropriate configuration object.
    """

    # create settings object
    module_path, class_name = config_dict.get("settings_path").rsplit(".", 1)
    module = import_module(module_path)
    component_settings_cls = getattr(module, class_name)
    config_dict["settings"] = TypeAdapter(component_settings_cls).validate_python(config_dict.get("settings"))

    # create config object
    module_path, class_name = config_dict.get("config_path").rsplit(".", 1)
    module = import_module(module_path)
    component_config_cls: Type[BaseConfiguration] = getattr(module, class_name)

    return TypeAdapter(component_config_cls).validate_python(config_dict)


def _load_component_from_configuration(config: BaseConfiguration) -> BasePipelineComponent:
    """Create an instance of a PipelineComponent based on the provided configuration.

    Args:
        config (ModelConfiguration): The configuration object to create the model instance.
    """

    # load the correct class based on the config, such as a datasource or a processor
    module_path, class_name = config.resource_path.rsplit(".", 1)
    module = import_module(module_path)
    pipeline_component_class: Type[BasePipelineComponent] = getattr(module, class_name)

    # create an instance of that class
    instance = pipeline_component_class(config.settings)

    return instance
