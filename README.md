# CCBot-v2

A flexible cryptocurrency trading bot with modular pipeline architecture and comprehensive backtesting capabilities.

## Table of Contents
- [Features](#features)
- [Installation](#installation)
- [Configuration](#configuration)
- [Components](#components)
- [Usage](#usage)
- [Trading Logic](#trading-logic)

## Features
- Modular pipeline architecture for a cryptocurrency trading bot
- Support for multiple exchanges through CCXT abstraction
- Supports flexible selection of trading pairs
- Real-time and historical data processing
- Possiblity to create arbitrarily complex prediction pipelines
- Comprehensive backtesting capabilities
- Risk management and position sizing
- Mock exchange for simulation testing

## Installation

### 1. Create Virtual Environment
```bash
python3.12 -m venv .venv
```

### 2. Activate Environment
```bash
source .venv/bin/activate
```

### 3. Install Requirements
```bash
pip install -r requirements.txt
```

### 4. Set Environment Variables
```bash
cp .example_env .env
```

### 5. Set Python Path
```bash
export PYTHONPATH=$(pwd)/src:$PYTHONPATH 
```
Configure your exchange API credentials and paths in the `.env` file. The current setup uses Bitget, but other CCXT-supported exchanges can be integrated with minor code adaptations.

## Components

### Pipeline System
The pipeline generates trade signals based on processed data through a chain of modular components:

#### 1. Datasources
- Handles real-time and historical data collection
- Multiple datasources can be combined. 
- Output Object: `Data`

#### 2. Data Processors
- Calculates features and target variables
- Implements technical indicators and custom metrics
- Input Object `Data`
- Output Object: `Data`

#### 3. Data Merger
- Combines data from multiple sources
- Input Object: `List[Data]`
- Output Object: `Data`

#### 4. Models
- Makes predictions using processed data
- Supports multiple model types and frameworks
- Additional models can be defined
- Input Object: `Data`
- Output Object: `List[Prediction]`

#### 5. Ensemble Models
- Aggregates predictions from multiple models
- Implements voting and probability-based mechanisms
- Input Object: `List[List[Prediction]]`
- Output Object: `List[Prediction]`

#### 6. Trade Signal Generators
- Converts predictions into actionable trade signals
- Defines order types, stop-loss, and take-profit levels
- Input Object: `List[Prediction]`
- Output Object: `List[TradeSignal]`

### Trading Engine
- Executes trades based on generated signals
- Manages position sizing and risk parameters
- Implements trading strategy rules
- Input Object: `List[TradeSignal]`


## Usage

### Configure and Train Pipeline

1. Create a pipeline configuration.

Examples are provided in `src/interface/cli/pipeline_configs/`.
Here you are able to define various parameters, such as which  pairs should be traded etc. 

2. Run the pipeline configuration script:
```bash
python src/interface/cli/pipeline_configs/pipeline_proba.py
```

This will:
- Create and configure the pipeline
- Export settings to JSON
- Save historical data
- Train machine learning models

An example configuration can be found in `example/pipeline_config.json`, demonstrating:
- 2 different datasources
- Multiple feature and target processors
- 3 directional models
- 3 volatility models
- 2 ensemble models
- Trade signal generator

### Running the Bot

#### Live Trading
```bash
python src/interface/app_realtime.py
```

#### Backtesting (optional)

1. Generate historical data:
```bash
python src/interface/cli/exchange_create_mock_data.py
```

2. Run simulation:
```bash
python src/interface/app_simulation.py
```

I implemented a fully working mock exchange which simulates real time trading based on historic data. It only works for bitget futures. Could be extended in the future.

## Trading Logic

The following matrix defines the bot's actions based on current positions and predicted signals:


| **Current Action \ Next Action** | **no trade**     | **buy**          | **sell**         |
|----------------------------------|------------------|------------------|------------------|
| **no trade**                     | do nothing       | open             | open             |
| **buy**                          | close            | do nothing       | open reverse     |
| **sell**                         | close            | open reverse     | do nothing       |


### Action Definitions
- **Do Nothing**: Maintain current position
- **Open Buy/Sell**: Enter new long/short position
- **Close Position**: Exit current position
- **Reverse Position**: Close current position and open opposite position


## License: Research and Educational Use Only

This software is licensed for research and educational purposes only. Any use of this software for live trading in financial markets or real-world financial transactions is strictly prohibited.

The author(s) make no warranties regarding the suitability of this software for any particular purpose and shall not be liable for any financial losses or legal consequences resulting from the use or misuse of this software. 