""" This module defines the logging component."""

import json
import logging
import re

import numpy as np
import pandas as pd


def create_logger(log_level: str, logger_name: str = "custom_logger"):
    """Create a logging based on logger.

    Args:
        log_level (str): Kind of logging
        logger_name (str, optional): Name of logger

    Returns:
        logger: returns logger
    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)  # Set the base logging level to the lowest (DEBUG)

    # If logger already has handlers, don't add a new one
    if logger.hasHandlers():
        logger.handlers.clear()

    # Create a console handler and set the level based on the input
    console_handler = logging.StreamHandler()
    if log_level == "DEBUG":
        console_handler.setLevel(logging.DEBUG)
    elif log_level == "INFO":
        console_handler.setLevel(logging.INFO)
    elif log_level == "WARNING":
        console_handler.setLevel(logging.WARNING)
    elif log_level == "ERROR":
        console_handler.setLevel(logging.ERROR)
    else:
        raise ValueError("Invalid log level provided")

    # Create a formatter and set it for the console handler
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s [%(name)s] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console_handler.setFormatter(formatter)

    # Add the console handler to the logger
    logger.addHandler(console_handler)

    return logger


class CustomJSONEncoder(json.JSONEncoder):
    """Class for serializing timestamps."""

    def default(self, o):
        """Override the default method to serialize timestamps.

        Args:
            obj: The object to serialize.

        Returns:

            str: The serialized object.
        """

        if isinstance(o, pd.Timestamp):
            if o.tzinfo is not None:
                # Serialize with timezone information
                return o.isoformat()
            else:
                # Serialize without timezone information
                return o.strftime("%Y-%m-%d %H:%M:%S")

        if isinstance(o, pd.Timedelta):
            # Serialize Timedelta as a string in "Xd", "Xh", "Xm", etc.
            seconds = o.total_seconds()
            if seconds % 86400 == 0:  # 86400 seconds in a day
                return f"{int(seconds // 86400)}d"
            elif seconds % 3600 == 0:
                return f"{int(seconds // 3600)}h"
            elif seconds % 60 == 0:
                return f"{int(seconds // 60)}m"
            else:
                return f"{seconds}s"

        if isinstance(o, np.float32):
            return float(o)

        return super().default(o)


def timestamp_decoder(obj: dict):
    """Convert strings back to pd.Timestamp or pd.Timedelta during JSON decoding."""

    # Regular expression to match Timedelta strings (e.g., "4h", "15min", "1d")
    timedelta_pattern = re.compile(r"^(\d+)([a-z]+)$")

    for key, value in obj.items():
        if isinstance(value, str):
            # Check if the value matches a Timedelta pattern (e.g., "4h", "15min", "1d")
            match = timedelta_pattern.match(value)
            if match:
                # Convert to pd.Timedelta if it matches the pattern
                obj[key] = pd.Timedelta(value)
            else:
                # If not a Timedelta pattern, try parsing as a Timestamp
                try:
                    obj[key] = pd.Timestamp(value, tz="utc")
                except ValueError:
                    # If it fails, leave it as a string
                    pass
    return obj
