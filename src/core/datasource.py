"""Module for abstract datasource class"""

import os
from abc import abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional

import joblib
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

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


class DatasourceSettings(BaseModel):
    """Settings used to configure the datasource."""

    # TODO: maybe unify scrape start date and mock start date
    object_id: ObjectId
    symbols: Optional[List[str]] = None
    use_mock_data: bool = False
    mock_data_path: Optional[str] = None
    scrape_start_date: Optional[pd.Timestamp] = None
    scrape_end_date: Optional[pd.Timestamp] = None
    mock_data_start_date: Optional[pd.Timestamp] = None
    mock_data_end_date: Optional[pd.Timestamp] = None
    simulation_start_date: Optional[pd.Timestamp] = None
    simulation_end_date: Optional[pd.Timestamp] = None

    model_config = ConfigDict(arbitrary_types_allowed=True)


class DatasourceConfiguration(BaseConfiguration):
    """Configuration specific to a datasource."""

    config_type: str = Field(
        default="datasource",
        Literal=True,
        description="Type of the configuration (e.g., model, processor). Do not change this value",
    )
    settings: DatasourceSettings


class Datasource(BasePipelineComponent):
    """Abstract datasource class."""

    def __init__(self, config: DatasourceSettings):
        """Initialize the DataSource object with specific parameters."""

        self.config = config
        self.mock_data = None
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

        if self.config.use_mock_data:
            if self.mock_data:
                return self._scrape_mock_data_historic(symbols)
            raise ValueError("Mock data needs to be loaded. Call set_simulation_mode() first")

        return self._scrape_real_data_historic(symbols)

    def scrape_data_current(self, symbols: Optional[List[str]] = None) -> Data:
        """Scrape current data."""

        symbols = symbols or self.config.symbols

        if self.config.use_mock_data:
            if self.mock_data:
                return self._scrape_mock_data_current(symbols)
            raise ValueError("Mock data needs to be loaded. Call set_simulation_mode() first")

        return self._scrape_real_data_current(symbols)

    def set_simulation_mode(self):
        """Activate simulation mode by loading or creating mock data."""

        # Check if use_mock_data is set to True
        if not self.config.use_mock_data:
            raise ValueError("Simulation mode requires `use_mock_data` to be set to True.")

        # Load or create mock data if not already loaded
        if hasattr(self, "mock_data") and self.mock_data:
            logger.info("Mock data already loaded.")
        else:
            self.mock_data = self.load_mock_data() if self.config.mock_data_path else self.create_mock_data()

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
        self.simulation_current_date = first_mock_date

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

        joblib_file_path = file_path or self.config.mock_data_path

        logger.info("Evaluation is turned on. Load historic mock dataset from: %s", joblib_file_path)

        data: Data = joblib.load(joblib_file_path)

        # Ensure that each entry is a DataFrame and set the index correctly
        for symbol, df in data.data.items():
            if not isinstance(df, pd.DataFrame):
                raise ValueError(f"The data for {symbol} is not a pandas DataFrame.")
            df["time"] = pd.to_datetime(df["time"], utc=True)
            df.set_index("time", inplace=True)
            data.data[symbol] = df

        logger.info("Mock data loaded successfully.")

        return data

    def save_mock_data(self, data: Data, file_path: Optional[str] = None) -> None:
        """Save mock data to a joblib file.

        Args:
            file_path (str): Path to the joblib file where the data will be saved.
        """

        joblib_file_path = file_path or self.config.mock_data_path

        # Reset index to ensure the timestamp is included in the saved DataFrame
        for symbol, df in data.data.items():
            data[symbol] = df.reset_index()

        joblib.dump(data, joblib_file_path)

        logger.info("Mock data saved to %s", joblib_file_path)

    def create_mock_data(self, symbols: Optional[List[str]] = None) -> Data:
        """Create mock data by scraping historical data."""

        logger.info("Evaluation is turned on. Creating mock data by scraping historic real data...")

        symbols = symbols or self.config.symbols

        data = self._scrape_real_data_historic(symbols)

        self.config.mock_data_start_date = min([min(data.index) for data in data.data.values()])
        self.config.mock_data_end_date = max([max(data.index) for data in data.data.values()])

        logger.info("Scraping process finished. This data can now be used to test the pipeline.")

        return data

    def create_configuration(self) -> DatasourceConfiguration:
        """Returns the configuration of the class."""

        resource_path = f"{self.__module__}.{self.__class__.__name__}"

        return DatasourceConfiguration(
            object_id=self.config.object_id,
            resource_path=resource_path,
            settings=self.config,
        )
