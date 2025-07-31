# Blockhouse Work Trial Task - Problem Analysis & Solution Approach

## Problem Statement Summary

The problem involves **optimal order execution** to minimize **temporary market impact** (slippage) when executing a large order over multiple trading periods. Here's what we need to solve:

### Key Requirements:
1. **Total Order Size**: Must execute exactly S shares by end of day
2. **Trading Periods**: N = 390 (one-minute windows)
3. **Objective**: Minimize total temporary impact across all executions
4. **Data**: Order book snapshots for 3 stocks (CRWV, FROG, SOUN) over multiple days

### Mathematical Formulation:
- Let `gt(x)` be the temporary impact function at time t for order size x
- Let `x ∈ R^N` be the allocation vector (shares to buy in each period)
- Constraint: `Σ(xi) = S` (total shares must equal S)
- Objective: Minimize `Σ(gt(xi))` (total slippage cost)

## Data Understanding

### Dataset Structure:
- **3 Stocks**: CRWV, FROG, SOUN
- **Time Period**: April 3 - May 2, 2025 (21 trading days each)
- **Data Format**: Order book snapshots with 10 levels of depth
- **Columns**: Timestamp, action, side, price, size, bid/ask prices and sizes at each level

### Key Observations from Data Analysis:

#### 1. **Market Characteristics by Stock**:
- **CRWV**: Mid-price ~$55, spread ~$0.30, moderate liquidity
- **FROG**: Mid-price ~$31, spread ~$1.90, lower liquidity, higher impact
- **SOUN**: Mid-price ~$7.90, spread ~$0.02, high liquidity, low impact

#### 2. **Temporary Impact Patterns**:
- **Non-linear relationship**: Impact increases more than linearly with order size
- **Asymmetric impact**: Buy and sell impacts can differ significantly
- **Stock-specific behavior**: Each stock shows different impact characteristics

#### 3. **Order Book Depth**:
- 10 levels of depth available
- Significant variation in liquidity across levels
- Some stocks show deeper liquidity than others

## Solution Approach

### 1. Temporary Impact Modeling

#### Linear Model (Baseline):
```
gt(x) ≈ βt * x
```
**Pros**: Simple, computationally efficient
**Cons**: Oversimplified, doesn't capture non-linear effects

#### Non-linear Models to Consider:

**a) Square Root Model**:
```
gt(x) = αt * √x + βt * x
```
- Captures diminishing marginal impact
- Common in market microstructure literature

**b) Power Law Model**:
```
gt(x) = αt * x^γ
```
- Where γ > 1 for convex impact, γ < 1 for concave
- Flexible to capture different impact patterns

**c) Piecewise Linear Model**:
```
gt(x) = Σ(βi * min(x, Li))
```
- Where Li are liquidity thresholds
- Captures discrete order book levels

### 2. Optimization Framework

#### Mathematical Formulation:
```
Minimize: Σ(gt(xi)) for i = 1 to N
Subject to: Σ(xi) = S
           xi ≥ 0 for all i
```

#### Solution Techniques:

**a) Dynamic Programming**:
- Break problem into subproblems
- Solve for optimal allocation at each time step
- Consider future impact predictions

**b) Convex Optimization**:
- If gt(x) is convex, use standard convex optimization
- Can handle constraints efficiently

**c) Reinforcement Learning**:
- Learn optimal policy from historical data
- Adapt to changing market conditions

### 3. Implementation Strategy

#### Phase 1: Impact Function Estimation
1. **Data Preprocessing**: Clean and align order book data
2. **Impact Calculation**: Compute slippage for various order sizes
3. **Model Fitting**: Fit different impact models to data
4. **Validation**: Test models on out-of-sample data

#### Phase 2: Optimization Algorithm
1. **Parameter Estimation**: Estimate model parameters for each stock
2. **Constraint Handling**: Implement total volume constraint
3. **Time-varying Models**: Account for intraday patterns
4. **Risk Management**: Add constraints for maximum order size

#### Phase 3: Backtesting & Validation
1. **Historical Simulation**: Test strategy on historical data
2. **Performance Metrics**: Compare against benchmarks
3. **Robustness Testing**: Test across different market conditions

### 4. Key Considerations

#### Market Microstructure Factors:
- **Bid-ask spread**: Wider spreads increase impact
- **Order book depth**: Deeper books reduce impact
- **Market volatility**: Higher volatility increases impact
- **Time of day**: Impact varies throughout trading day

#### Practical Constraints:
- **Minimum order sizes**: Some exchanges have minimums
- **Maximum order sizes**: Risk management limits
- **Execution speed**: Need to execute within time windows
- **Market impact**: Large orders can move prices

#### Risk Management:
- **Maximum drawdown**: Limit worst-case slippage
- **VaR constraints**: Control value at risk
- **Liquidity constraints**: Ensure orders can be filled

## Next Steps

1. **Implement impact function estimation** for each stock
2. **Develop optimization algorithm** with proper constraints
3. **Create backtesting framework** to validate approach
4. **Analyze results** and refine models
5. **Document findings** and recommendations

## Expected Outcomes

- **Reduced slippage**: Compared to naive strategies (e.g., equal allocation)
- **Adaptive execution**: Strategy that adapts to market conditions
- **Robust performance**: Consistent across different market environments
- **Practical implementation**: Algorithm that can be deployed in real markets 