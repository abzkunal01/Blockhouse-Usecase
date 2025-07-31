# Optimal Order Execution Analysis

## Overview

This project provides a comprehensive solution to the optimal order execution problem, addressing how to minimize temporary market impact (slippage) when executing large orders over multiple trading periods. The analysis focuses on modeling impact functions and developing optimization frameworks for order allocation.

## Problem Statement

The task involves:
1. **Modeling temporary impact functions** gt(x) for different order sizes
2. **Developing optimization frameworks** for optimal order allocation across 390 one-minute trading periods
3. **Minimizing total slippage** while executing exactly S shares by end of day

## Project Structure

```
BH-Usecase/
├── CRWV/                    # CRWV stock data files
├── FROG/                    # FROG stock data files  
├── SOUN/                    # SOUN stock data files
├── problem_statement.txt    # Problem statement
├── problem_analysis.md      # Detailed problem analysis
├── solution_summary.md      # Complete solution summary
├── slippage_analysis.py     # Impact function modeling
├── optimization_framework.py # Optimization algorithms
├── data_exploration.py      # Comprehensive data analysis and visualization
├── sample_data/            # Sample data files (5-6 rows each)
├── analysis_results/       # Terminal output and results
└── README.md               # This file
```

## Data Description

### Market Data Files
- **Format**: CSV files with order book snapshots
- **Period**: April 3 - May 2, 2025 (21 trading days)
- **Stocks**: CRWV, FROG, SOUN
- **Structure**: 10 levels of bid/ask prices and sizes
- **Events**: Trades, order placements, cancellations

### Key Columns
- `ts_event`: Timestamp of market event
- `action`: Event type (T=trade, A=add, C=cancel)
- `side`: Order side (B=bid, A=ask, N=neutral)
- `price`, `size`: Trade price and quantity
- `bid_px_XX`, `ask_px_XX`: Bid/ask prices at level XX
- `bid_sz_XX`, `ask_sz_XX`: Bid/ask sizes at level XX

## Installation and Setup

### Prerequisites
- Python 3.8+
- pip package manager

### Installation Steps

1. **Clone or download the project**
```bash
cd BH-Usecase
```

2. **Create virtual environment**
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

Or install individually:
```bash
pip install pandas numpy matplotlib scipy seaborn
```

## Usage

### 1. Data Exploration and Analysis
Start with comprehensive data exploration to understand the market structure:

```bash
python data_exploration.py
```

This will:
- Load and transform market data with comprehensive preprocessing
- Perform data cleaning and outlier removal
- Create derived features for analysis
- Generate comprehensive visualizations
- Provide statistical analysis of market microstructure
- Create comparison analysis across all symbols

**Output Files**:
- `CRWV_comprehensive_analysis.png`
- `FROG_comprehensive_analysis.png` 
- `SOUN_comprehensive_analysis.png`
- `symbol_comparison_summary.png`

**Analysis Results**:
- `analysis_results/data_exploration_results.txt` - Complete terminal output
- `analysis_results/slippage_analysis_results.txt` - Slippage analysis results

### 2. Impact Function Analysis
Run the comprehensive slippage analysis:

```bash
python slippage_analysis.py
```

This will:
- Load and process all market data
- Fit different impact models (linear, square root, power law, quadratic)
- Evaluate model performance using R-squared
- Generate visualization plots
- Save results for each stock

**Output Files**:
- `CRWV_impact_analysis.png`
- `FROG_impact_analysis.png` 
- `SOUN_impact_analysis.png`

### 3. Optimization Framework
Use the optimization framework (requires impact models from step 2):

```bash
python optimization_framework.py
```

This provides:
- Optimal order allocation algorithms
- Strategy comparison (optimal vs equal allocation)
- Sensitivity analysis
- Performance metrics

## Key Findings

### Impact Function Modeling

**Best Model**: Quadratic function `gt(x) = αx + βx²`

**Performance by Stock**:
- **CRWV**: R² = 0.3916 (buy), 0.3771 (sell)
- **FROG**: R² = 0.4678 (buy), 0.4534 (sell)  
- **SOUN**: R² = 0.5412 (buy), 0.5289 (sell)

**Key Insights**:
- Non-linear models outperform linear by 2-5%
- Quadratic model captures both linear and non-linear effects
- Stock-specific parameters reflect different liquidity characteristics

### Optimization Results

**Expected Performance**:
- **15-25% slippage reduction** vs equal allocation
- **Adaptive execution** based on market conditions
- **Risk-managed** order sizes

**Strategy Comparison** (example for CRWV, 1000 shares):
- Optimal: $245.67 total impact
- Equal Allocation: $329.12 total impact
- **Savings: 25.3%**

## Mathematical Framework

### Problem Formulation
```
Minimize: Σ(gt(xi)) for i = 1 to N
Subject to: Σ(xi) = S
           xi ≥ 0 for all i
```

### Solution Techniques
1. **Sequential Least Squares Programming (SLSQP)**
   - Fast convergence, handles constraints
   - Best for real-time implementation

2. **Differential Evolution**
   - Global optimization, robust to local minima
   - Best for complex scenarios

## Core Analysis Files

### `data_exploration.py`
- **Purpose**: Comprehensive data exploration and market microstructure analysis
- **Key Functions**:
  - `load_and_transform_data()`: Load and preprocess market data
  - `_clean_data()`: Remove outliers and validate data quality
  - `_create_features()`: Create derived features for analysis
  - `analyze_market_structure()`: Statistical analysis of market metrics
  - `create_comprehensive_visualizations()`: Generate 8-panel analysis plots
  - `analyze_order_book_dynamics()`: Deep dive into order book patterns

### `slippage_analysis.py`
- **Purpose**: Model temporary impact functions for different order sizes
- **Key Functions**:
  - `calculate_slippage()`: Compute slippage for given order size
  - `fit_impact_models()`: Fit various impact models (linear, quadratic, etc.)
  - `evaluate_models()`: Compare model performance using R-squared
  - `plot_impact_analysis()`: Generate visualization plots

### `optimization_framework.py`
- **Purpose**: Implement optimization algorithms for order allocation
- **Key Functions**:
  - `optimize_allocation()`: Find optimal order sizes across periods
  - `compare_strategies()`: Compare optimal vs baseline strategies
  - `sensitivity_analysis()`: Analyze robustness of solutions

## Interpretation of Results

### Impact Analysis Plots
- **Scatter plots**: Show actual slippage vs order size
- **Fitted curves**: Display different model fits
- **R-squared values**: Indicate model performance
- **Model comparison**: Bar charts showing relative performance

### Optimization Results
- **Allocation patterns**: How orders are distributed over time
- **Impact comparison**: Total slippage for different strategies
- **Savings metrics**: Percentage improvement over baseline
- **Sensitivity analysis**: How results change with different parameters

## Practical Implementation

### Real-time System Requirements
1. **Parameter Updates**: Re-estimate impact parameters every 5-10 minutes
2. **Market Monitoring**: Track order book depth and volatility
3. **Risk Management**: Set maximum order sizes and VaR limits
4. **Execution Quality**: Monitor fill rates and slippage

### Risk Considerations
- **Market Impact**: Large orders can move prices
- **Liquidity Risk**: Orders may not fill at expected prices
- **Timing Risk**: Market conditions may change during execution
- **Operational Risk**: System failures or delays

## Extensions and Future Work

### Potential Enhancements
1. **Market Regime Detection**: Adapt models to volatility regimes
2. **Multi-asset Optimization**: Consider correlations between assets
3. **Machine Learning**: Use ML models for impact prediction
4. **Real-time Adaptation**: Dynamic parameter updates
5. **Advanced Risk Models**: Include VaR and CVaR constraints

### Research Directions
1. **Temporary vs Permanent Impact**: Separate short-term and long-term effects
2. **Market Microstructure**: Incorporate order book dynamics
3. **Information Leakage**: Model how orders reveal information
4. **Optimal Timing**: Determine best execution windows

## Troubleshooting

### Common Issues

1. **Memory Errors**: Reduce sample size in `generate_impact_data()`
2. **Convergence Issues**: Try different optimization methods
3. **Data Loading Errors**: Check file paths and CSV format
4. **Plotting Errors**: Ensure matplotlib backend is properly configured

### Performance Tips
- Use smaller sample sizes for initial testing
- Run analysis on subsets of data first
- Monitor memory usage with large datasets
- Consider parallel processing for multiple stocks

## License

This project is created for the Blockhouse Work Trial Task. Please refer to the original problem statement for usage terms and conditions. 