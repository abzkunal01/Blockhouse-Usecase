# Sample Market Data

This directory contains sample data files (5-6 rows each) from the full market datasets to demonstrate the data structure and format.

## Files

- `CRWV_sample.csv` - Sample data for CrowdStrike (CRWV)
- `FROG_sample.csv` - Sample data for Frog (FROG)  
- `SOUN_sample.csv` - Sample data for SoundHound (SOUN)

## Data Structure

Each CSV file contains order book snapshots with the following key columns:

### Core Fields
- `ts_event`: Timestamp of the market event
- `action`: Event type (T=trade, A=add order, C=cancel order)
- `side`: Order side (B=bid, A=ask, N=neutral)
- `price`: Trade price (for trades)
- `size`: Trade quantity (for trades)

### Order Book Levels (0-9)
- `bid_px_XX`: Bid price at level XX
- `ask_px_XX`: Ask price at level XX
- `bid_sz_XX`: Bid size at level XX
- `ask_sz_XX`: Ask size at level XX
- `bid_ct_XX`: Bid count at level XX
- `ask_ct_XX`: Ask count at level XX

### Example Row
```
ts_event,action,side,price,size,bid_px_00,ask_px_00,bid_sz_00,ask_sz_00,...
2025-04-14 13:30:00.746984828+00:00,T,N,0,46.77,1,46.72,46.84,138,18,...
```

## Data Period
- **Full Dataset**: April 3 - May 2, 2025 (21 trading days)
- **Sample Data**: 6 rows from April 14, 2025

## Usage
These sample files can be used to:
1. Understand the data format
2. Test data loading scripts
3. Verify column structure
4. Demonstrate the analysis pipeline

## Note
The full datasets (CRWV/, FROG/, SOUN/) are excluded from this repository to maintain privacy. The sample data provides sufficient information to understand the data structure and run the analysis scripts. 