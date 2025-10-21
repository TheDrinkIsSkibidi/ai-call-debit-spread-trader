# External Dependencies

This directory contains the three open-source repositories that are integrated into the AI Call Debit Spread Trader platform.

## Required Repositories

Please clone the following repositories into this directory:

### 1. optlib
- **Repository**: https://github.com/dbrojas/optlib.git
- **Purpose**: Option chain retrieval, pricing models, and analytical Greeks computation
- **Installation**: 
  ```bash
  git clone https://github.com/dbrojas/optlib.git
  cd optlib && pip install -e .
  ```

### 2. optionlab  
- **Repository**: https://github.com/rgaveiga/optionlab.git
- **Purpose**: Option strategy simulation, payoff diagrams, and probability-of-profit computation
- **Installation**:
  ```bash
  git clone https://github.com/rgaveiga/optionlab.git
  cd optionlab && pip install -e .
  ```

### 3. pyalgostrategypool
- **Repository**: https://github.com/algobulls/pyalgostrategypool.git  
- **Purpose**: Backtesting and strategy execution infrastructure for algorithmic trading
- **Installation**:
  ```bash
  git clone https://github.com/algobulls/pyalgostrategypool.git
  cd pyalgostrategypool && pip install -e .
  ```

## Quick Setup

Run this from the project root to set up all external dependencies:

```bash
# From project root directory
cd external/

# Clone all repositories
git clone https://github.com/dbrojas/optlib.git
git clone https://github.com/rgaveiga/optionlab.git
git clone https://github.com/algobulls/pyalgostrategypool.git

# Install as editable packages
cd optlib && pip install -e . && cd ..
cd optionlab && pip install -e . && cd ..
cd pyalgostrategypool && pip install -e . && cd ..

cd ..
```

## Integration Notes

These repositories are integrated into our platform as follows:

- **optlib**: Used in `src/data_ingestion/market_data.py` and `src/spread_constructor/spread_builder.py`
- **optionlab**: Used in `src/optimizer/strategy_optimizer.py` for strategy evaluation
- **pyalgostrategypool**: Used in `src/trade_execution/trade_manager.py` for trade execution infrastructure

## License Compliance

Please ensure you comply with the licenses of these external repositories:
- optlib: Check repository for license terms
- optionlab: Check repository for license terms  
- pyalgostrategypool: MIT License

## Attribution

We acknowledge and thank the maintainers of these excellent open-source projects:
- optlib by Davis Edwards & Daniel Rojas
- optionlab by rgaveiga
- pyalgostrategypool by AlgoBulls