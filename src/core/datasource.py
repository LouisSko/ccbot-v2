"""Module for abstract datasource class"""

import os
from abc import abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Type, Union

import joblib
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.common_packages import create_logger
from src.core.base import BaseConfiguration, BasePipelineComponent, ObjectId

# instantiate logger
logger = create_logger(
    log_level=os.getenv("LOGGING_LEVEL", "INFO"),
    logger_name=__name__,
)


@dataclass
class Data:
    """Data loaded from a datasource."""

    object_ref: ObjectId  # Object ID of the Pipeline Component that created the data.
    data: Dict[str, pd.DataFrame]


@dataclass
class TrainingData(Data):
    """Data loaded from a datasource."""

    def __post_init__(self):
        for symbol, df in self.data.items():
            if "target" not in df.columns:
                raise ValueError(f"Target column is missing in the data for symbol '{symbol}'")


# TODO mock_data_start_date and end date should not be defined here
class DatasourceSettings(BaseModel):
    """Settings used to configure the datasource."""

    object_id: ObjectId
    symbols: Optional[List[str]] = None
    data_directory: Optional[str] = None  # path to the directory where the mock data is stored
    scrape_start_date: Optional[pd.Timestamp] = None
    scrape_end_date: Optional[pd.Timestamp] = None
    mock_data_start_date: Optional[pd.Timestamp] = None  # just for information purpose
    mock_data_end_date: Optional[pd.Timestamp] = None  # just for information purpose
    simulation_start_date: Optional[pd.Timestamp] = None
    simulation_end_date: Optional[pd.Timestamp] = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @field_validator(
        "simulation_start_date",
        "simulation_end_date",
        "scrape_start_date",
        "scrape_end_date",
        mode="before",
    )
    def check_timestamp(cls, value):  # pylint: disable=no-self-argument
        """check if timestamps are set correctly."""

        if isinstance(value, pd.Timestamp):
            if value != value.round(pd.Timedelta("1d")):
                raise ValueError(f"{value} must be defined as a full day: like '2023-08-08T00:00:00+00:00'")

        return value


class DatasourceConfiguration(BaseConfiguration):
    """Configuration specific to a datasource."""

    component_type: str = Field(
        default="datasource",
        Literal=True,
        description="Type of the configuration (e.g., model, processor). Do not change this value",
    )

    settings: Union[DatasourceSettings, Type[DatasourceSettings]]


class Datasource(BasePipelineComponent):
    """Abstract datasource class."""

    def __init__(self, config: DatasourceSettings):
        """Initialize the DataSource object with specific parameters."""

        self.config = config
        self.mock_data = None
        self.use_mock_data = False
        self.simulation_current_date = None
        self.simululation_end = False

    @abstractmethod
    def _scrape_mock_data_historic(self, symbols: Optional[List[str]]) -> Data:
        """Scrape historic data from mock data."""

        pass

    @abstractmethod
    def _scrape_real_data_historic(self, symbols: Optional[List[str]]) -> Data:
        """Scrape historic data from the real exchange."""

        pass

    @abstractmethod
    def _scrape_mock_data_current(self, symbols: Optional[List[str]]) -> Data:
        """Scrape current data from mock data."""

        pass

    @abstractmethod
    def _scrape_real_data_current(self, symbols: Optional[List[str]]) -> Data:
        """Scrape current data from the real exchange."""

        pass

    def scrape_data_historic(self, symbols: Optional[List[str]] = None) -> Data:
        """Scrape historic data."""

        symbols = symbols or self.config.symbols

        if self.use_mock_data:
            if self.mock_data:
                return self._scrape_mock_data_historic(symbols)
            raise ValueError("Mock data needs to be loaded. Call set_simulation_mode() first")

        return self._scrape_real_data_historic(symbols)

    def scrape_data_current(self, symbols: Optional[List[str]] = None) -> Data:
        """Scrape current data."""

        symbols = symbols or self.config.symbols

        if self.use_mock_data:
            if self.mock_data:
                return self._scrape_mock_data_current(symbols)
            raise ValueError("Mock data needs to be loaded. Call set_simulation_mode() first")

        return self._scrape_real_data_current(symbols)

    def set_simulation_mode(self):
        """Activate simulation mode by loading or creating mock data."""

        # Check if use_mock_data is set to True
        if not self.use_mock_data:
            raise ValueError("Simulation mode requires `use_mock_data` to be set to True.")

        # Load or create mock data if not already loaded
        if self.mock_data:
            logger.info("Mock data already loaded. Trying to reload...")

        mock_data_path = os.path.join(self.config.data_directory, self.config.object_id.value + "_mock.joblib")
        # check if the mock data file exists
        if os.path.exists(mock_data_path):
            logger.info("Found existing mock data.")
            self.mock_data = self.load_mock_data()
        else:
            logger.info("Create new mock data.")
            self.mock_data = self.create_mock_data()

        # Process symbol-specific data dates
        symbol_dates = [data.index for data in self.mock_data.data.values() if not data.empty]
        first_mock_date = min(min(dates) for dates in symbol_dates)
        last_mock_date = max(max(dates) for dates in symbol_dates)

        # Validate simulation start and end dates based on available mock data
        if first_mock_date is None or last_mock_date is None:
            raise ValueError("No valid mock data found to set simulation mode.")

        if self.config.simulation_start_date is None or self.config.simulation_start_date < first_mock_date:
            logger.info("Simulation start date is too early. Adjusting to %s.", first_mock_date)
            self.config.simulation_start_date = first_mock_date

        if self.config.simulation_end_date is None or self.config.simulation_end_date > last_mock_date:
            logger.info("Simulation end date is too late. Adjusting to %s.", last_mock_date)
            self.config.simulation_end_date = last_mock_date

        # set first simulation date
        # TODO: make it more robust because if simulation date is same as mock date we might not be able to fetch data correctly
        self.simulation_current_date = self.config.simulation_start_date

        logger.info(
            "Simulation mode activated. From %s to %s.",
            self.config.simulation_start_date,
            self.config.simulation_end_date,
        )

    def load_mock_data(self, file_path: Optional[str] = None) -> Data:
        """Load mock data from a joblib file or from a default path specified in the config.

        Args:
            file_path (Optional[str]): Path to the joblib file containing mock data. If not provided, the path specified in the config will be used.
        """

        joblib_file_path = file_path or os.path.join(
            self.config.data_directory, self.config.object_id.value + "_mock.joblib"
        )

        logger.info("Evaluation is turned on. Load historic mock dataset from: %s", joblib_file_path)

        data: Data = joblib.load(joblib_file_path)

        logger.info("Mock data loaded successfully.")

        return data

    def save_mock_data(self, data: Data, file_path: Optional[str] = None) -> None:
        """Save mock data to a joblib file.

        Args:
            file_path (str): Path to the joblib file where the data will be saved.
        """

        data_file_path = file_path or os.path.join(
            self.config.data_directory, self.config.object_id.value + "_mock.joblib"
        )

        # Get the directory from the model path
        data_dir = os.path.dirname(data_file_path)

        # Create the directory if it doesn't exist
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
            logger.info("Directory does not exist. Create a new one.")

        # Reset index to ensure the timestamp is included in the saved DataFrame
        # for df in data.data.values():
        #    df = df.reset_index()

        joblib.dump(data, data_file_path)

        logger.info("Mock data saved to %s", data_file_path)

    def create_mock_data(self, symbols: Optional[List[str]] = None) -> Data:
        """Create mock data by scraping historical data."""

        logger.info("Evaluation is turned on. Creating mock data by scraping historic real data...")

        symbols = symbols or self.config.symbols

        data = self._scrape_real_data_historic(symbols)
        
        data.data = {key: data for key, data in data.data.items() if len(data) > 0}

        self.config.mock_data_start_date = min(min(data.index) for data in data.data.values())
        self.config.mock_data_end_date = max(max(data.index) for data in data.data.values())

        logger.info(
            "Scraping process finished. Start date: %s, End date: %s. This data can now be used to test the pipeline.",
            self.config.mock_data_start_date,
            self.config.mock_data_end_date,
        )

        return data

    def create_configuration(self) -> DatasourceConfiguration:
        """Returns the configuration of the class."""

        resource_path = f"{self.__module__}.{self.__class__.__name__}"
        config_path = f"{DatasourceConfiguration.__module__}.{DatasourceConfiguration.__name__}"
        settings_path = f"{self.config.__module__}.{self.config.__class__.__name__}"

        return DatasourceConfiguration(
            object_id=self.config.object_id,
            resource_path=resource_path,
            config_path=config_path,
            settings_path=settings_path,
            settings=self.config,
        )
