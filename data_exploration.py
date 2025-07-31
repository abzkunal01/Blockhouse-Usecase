#!/usr/bin/env python3
"""
Comprehensive Data Exploration and Analysis
Blockhouse Work Trial Task

This script demonstrates deep understanding of:
1. Data loading and transformation
2. Data cleaning and preprocessing
3. Exploratory data analysis
4. Visualization for better understanding
5. Statistical analysis of market microstructure
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import glob
import os
import warnings
from scipy import stats
from collections import defaultdict
warnings.filterwarnings('ignore')

# Set style for blue-white theme
plt.style.use('default')
sns.set_palette("husl")

class MarketDataExplorer:
    def __init__(self, data_dir="."):
        self.data_dir = data_dir
        self.symbols = ['CRWV', 'FROG', 'SOUN']
        self.data = {}
        self.processed_data = {}
        
    def load_and_transform_data(self, symbol, sample_days=3):
        """
        Load and transform market data with comprehensive preprocessing
        """
        print(f"Loading and transforming {symbol} data...")
        
        # Load data files
        files = glob.glob(f"{symbol}/*.csv")
        files.sort()
        
        # Take sample of recent days for analysis
        files = files[-sample_days:] if len(files) > sample_days else files
        
        all_data = []
        for file in files:
            print(f"  Processing {os.path.basename(file)}...")
            
            # Load with optimized settings
            df = pd.read_csv(file, 
                           parse_dates=['ts_event'],
                           infer_datetime_format=True,
                           low_memory=False)
            
            # Basic data transformation
            df['symbol'] = symbol
            df['date'] = df['ts_event'].dt.date
            df['time'] = df['ts_event'].dt.time
            df['hour'] = df['ts_event'].dt.hour
            df['minute'] = df['ts_event'].dt.minute
            
            # Extract trading session
            df['trading_session'] = df['hour'].apply(
                lambda x: 'pre_market' if x < 9 else 'regular' if x < 16 else 'post_market'
            )
            
            all_data.append(df)
        
        # Combine all data
        combined_df = pd.concat(all_data, ignore_index=True)
        
        # Data cleaning and validation
        combined_df = self._clean_data(combined_df)
        
        # Create derived features
        combined_df = self._create_features(combined_df)
        
        self.data[symbol] = combined_df
        print(f"  Loaded {len(combined_df):,} records for {symbol}")
        
        return combined_df
    
    def _clean_data(self, df):
        """
        Comprehensive data cleaning and validation
        """
        print("  Cleaning data...")
        
        # Remove duplicates
        initial_rows = len(df)
        df = df.drop_duplicates()
        print(f"    Removed {initial_rows - len(df)} duplicate rows")
        
        # Handle missing values
        missing_counts = df.isnull().sum()
        print(f"    Missing values: {missing_counts.sum()} total")
        
        # Filter valid order book states
        df = df[df['bid_px_00'] > 0]
        df = df[df['ask_px_00'] > 0]
        df = df[df['bid_px_00'] < df['ask_px_00']]  # Valid spread
        
        # Remove extreme outliers
        df = self._remove_outliers(df)
        
        print(f"    Final clean dataset: {len(df):,} rows")
        return df
    
    def _remove_outliers(self, df, threshold=3):
        """
        Remove statistical outliers from price and size data
        """
        # Calculate mid price
        df['mid_price'] = (df['bid_px_00'] + df['ask_px_00']) / 2
        
        # Remove price outliers
        price_z_scores = np.abs(stats.zscore(df['mid_price']))
        df = df[price_z_scores < threshold]
        
        # Remove size outliers
        size_columns = [col for col in df.columns if 'sz_' in col]
        for col in size_columns:
            if df[col].dtype in ['int64', 'float64']:
                z_scores = np.abs(stats.zscore(df[col]))
                df = df[z_scores < threshold]
        
        return df
    
    def _create_features(self, df):
        """
        Create derived features for analysis
        """
        print("  Creating derived features...")
        
        # Basic order book metrics
        df['spread'] = df['ask_px_00'] - df['bid_px_00']
        df['spread_bps'] = (df['spread'] / df['bid_px_00']) * 10000  # Basis points
        df['mid_price'] = (df['bid_px_00'] + df['ask_px_00']) / 2
        
        # Order book depth
        df['bid_depth'] = df[[col for col in df.columns if 'bid_sz_' in col]].sum(axis=1)
        df['ask_depth'] = df[[col for col in df.columns if 'ask_sz_' in col]].sum(axis=1)
        df['total_depth'] = df['bid_depth'] + df['ask_depth']
        
        # Price levels
        for i in range(10):
            bid_px_col = f'bid_px_{i:02d}'
            ask_px_col = f'ask_px_{i:02d}'
            if bid_px_col in df.columns and ask_px_col in df.columns:
                df[f'level_{i}_spread'] = df[ask_px_col] - df[bid_px_col]
        
        # Volume-weighted average price (VWAP) approximation
        df['vwap_approx'] = (
            (df['bid_px_00'] * df['bid_sz_00'] + df['ask_px_00'] * df['ask_sz_00']) /
            (df['bid_sz_00'] + df['ask_sz_00'])
        )
        
        # Market impact indicators
        df['bid_ask_imbalance'] = (df['bid_depth'] - df['ask_depth']) / df['total_depth']
        
        # Time-based features
        df['seconds_from_open'] = (
            df['ts_event'].dt.hour * 3600 + 
            df['ts_event'].dt.minute * 60 + 
            df['ts_event'].dt.second
        )
        
        return df
    
    def analyze_market_structure(self, symbol):
        """
        Comprehensive market structure analysis
        """
        df = self.data[symbol]
        print(f"\n=== Market Structure Analysis for {symbol} ===")
        
        # Basic statistics
        print(f"Data Period: {df['date'].min()} to {df['date'].max()}")
        print(f"Total Records: {len(df):,}")
        print(f"Trading Days: {df['date'].nunique()}")
        
        # Price statistics
        print(f"\nPrice Statistics:")
        print(f"  Mid Price - Mean: ${df['mid_price'].mean():.2f}")
        print(f"  Mid Price - Std: ${df['mid_price'].std():.2f}")
        print(f"  Mid Price - Range: ${df['mid_price'].min():.2f} - ${df['mid_price'].max():.2f}")
        
        # Spread analysis
        print(f"\nSpread Analysis:")
        print(f"  Average Spread: ${df['spread'].mean():.4f}")
        print(f"  Spread in BPS: {df['spread_bps'].mean():.2f}")
        print(f"  Spread Std: ${df['spread'].std():.4f}")
        
        # Depth analysis
        print(f"\nOrder Book Depth:")
        print(f"  Average Bid Depth: {df['bid_depth'].mean():,.0f} shares")
        print(f"  Average Ask Depth: {df['ask_depth'].mean():,.0f} shares")
        print(f"  Total Depth: {df['total_depth'].mean():,.0f} shares")
        
        # Event analysis
        print(f"\nEvent Analysis:")
        event_counts = df['action'].value_counts()
        for action, count in event_counts.items():
            print(f"  {action}: {count:,} ({count/len(df)*100:.1f}%)")
        
        return df
    
    def create_comprehensive_visualizations(self, symbol):
        """
        Create comprehensive visualizations for data understanding
        """
        df = self.data[symbol]
        
        # Set up the plotting style
        plt.style.use('default')
        fig = plt.figure(figsize=(20, 24))
        
        # 1. Price and Spread Time Series
        plt.subplot(4, 2, 1)
        sample_data = df.sample(min(10000, len(df)))
        plt.scatter(sample_data['ts_event'], sample_data['mid_price'], 
                   alpha=0.6, s=1, color='blue')
        plt.title(f'{symbol} - Mid Price Over Time', fontsize=14, fontweight='bold')
        plt.ylabel('Mid Price ($)')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        # 2. Spread Distribution
        plt.subplot(4, 2, 2)
        plt.hist(df['spread_bps'], bins=50, alpha=0.7, color='lightblue', edgecolor='black')
        plt.title(f'{symbol} - Spread Distribution (BPS)', fontsize=14, fontweight='bold')
        plt.xlabel('Spread (Basis Points)')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        
        # 3. Order Book Depth Analysis
        plt.subplot(4, 2, 3)
        depth_data = df[['bid_depth', 'ask_depth']].sample(min(5000, len(df)))
        plt.scatter(depth_data['bid_depth'], depth_data['ask_depth'], 
                   alpha=0.6, s=1, color='green')
        plt.plot([0, depth_data['bid_depth'].max()], [0, depth_data['bid_depth'].max()], 
                'r--', alpha=0.8, label='Perfect Balance')
        plt.title(f'{symbol} - Bid vs Ask Depth', fontsize=14, fontweight='bold')
        plt.xlabel('Bid Depth (shares)')
        plt.ylabel('Ask Depth (shares)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 4. Trading Session Analysis
        plt.subplot(4, 2, 4)
        session_stats = df.groupby('trading_session').agg({
            'spread_bps': 'mean',
            'total_depth': 'mean',
            'bid_ask_imbalance': 'mean'
        }).reset_index()
        
        x = np.arange(len(session_stats))
        width = 0.25
        
        plt.bar(x - width, session_stats['spread_bps'], width, 
               label='Spread (BPS)', color='lightcoral', alpha=0.8)
        plt.bar(x, session_stats['total_depth']/1000, width, 
               label='Depth (K shares)', color='lightblue', alpha=0.8)
        plt.bar(x + width, session_stats['bid_ask_imbalance']*100, width, 
               label='Imbalance (%)', color='lightgreen', alpha=0.8)
        
        plt.title(f'{symbol} - Market Metrics by Trading Session', fontsize=14, fontweight='bold')
        plt.xlabel('Trading Session')
        plt.ylabel('Value')
        plt.xticks(x, session_stats['trading_session'])
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 5. Event Type Distribution
        plt.subplot(4, 2, 5)
        event_counts = df['action'].value_counts()
        colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']
        plt.pie(event_counts.values, labels=event_counts.index, autopct='%1.1f%%',
                colors=colors[:len(event_counts)], startangle=90)
        plt.title(f'{symbol} - Event Type Distribution', fontsize=14, fontweight='bold')
        
        # 6. Price Level Analysis
        plt.subplot(4, 2, 6)
        level_spreads = []
        level_names = []
        for i in range(5):  # First 5 levels
            col = f'level_{i}_spread'
            if col in df.columns:
                level_spreads.append(df[col].mean())
                level_names.append(f'Level {i}')
        
        plt.bar(level_names, level_spreads, color='skyblue', alpha=0.8)
        plt.title(f'{symbol} - Average Spread by Price Level', fontsize=14, fontweight='bold')
        plt.ylabel('Average Spread ($)')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        # 7. Bid-Ask Imbalance Distribution
        plt.subplot(4, 2, 7)
        plt.hist(df['bid_ask_imbalance'], bins=50, alpha=0.7, color='orange', edgecolor='black')
        plt.axvline(df['bid_ask_imbalance'].mean(), color='red', linestyle='--', 
                   label=f'Mean: {df["bid_ask_imbalance"].mean():.3f}')
        plt.title(f'{symbol} - Bid-Ask Imbalance Distribution', fontsize=14, fontweight='bold')
        plt.xlabel('Imbalance Ratio')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 8. Volume Profile (approximation)
        plt.subplot(4, 2, 8)
        # Group by price buckets
        df['price_bucket'] = pd.cut(df['mid_price'], bins=20)
        volume_profile = df.groupby('price_bucket')['total_depth'].mean()
        
        plt.bar(range(len(volume_profile)), volume_profile.values, 
               color='purple', alpha=0.7)
        plt.title(f'{symbol} - Volume Profile (Depth by Price)', fontsize=14, fontweight='bold')
        plt.xlabel('Price Bucket')
        plt.ylabel('Average Depth')
        plt.xticks(range(len(volume_profile)), 
                  [f'{bucket.left:.1f}' for bucket in volume_profile.index], 
                  rotation=45)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{symbol}_comprehensive_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Comprehensive analysis saved as {symbol}_comprehensive_analysis.png")
    
    def analyze_order_book_dynamics(self, symbol):
        """
        Analyze order book dynamics and patterns
        """
        df = self.data[symbol]
        print(f"\n=== Order Book Dynamics Analysis for {symbol} ===")
        
        # Analyze price level depth
        print("\nPrice Level Depth Analysis:")
        for i in range(5):
            bid_sz_col = f'bid_sz_{i:02d}'
            ask_sz_col = f'ask_sz_{i:02d}'
            if bid_sz_col in df.columns and ask_sz_col in df.columns:
                avg_bid = df[bid_sz_col].mean()
                avg_ask = df[ask_sz_col].mean()
                print(f"  Level {i}: Bid={avg_bid:.0f}, Ask={avg_ask:.0f}, Ratio={avg_bid/avg_ask:.2f}")
        
        # Analyze spread dynamics
        print(f"\nSpread Dynamics:")
        print(f"  Spread volatility: {df['spread_bps'].std():.2f} BPS")
        print(f"  Spread skewness: {df['spread_bps'].skew():.3f}")
        print(f"  Spread kurtosis: {df['spread_bps'].kurtosis():.3f}")
        
        # Analyze depth dynamics
        print(f"\nDepth Dynamics:")
        print(f"  Depth volatility: {df['total_depth'].std():.0f} shares")
        print(f"  Depth skewness: {df['total_depth'].skew():.3f}")
        print(f"  Depth kurtosis: {df['total_depth'].kurtosis():.3f}")
        
        # Correlation analysis
        correlations = df[['spread_bps', 'total_depth', 'bid_ask_imbalance', 'mid_price']].corr()
        print(f"\nCorrelation Matrix:")
        print(correlations.round(3))
        
        return correlations
    
    def run_complete_analysis(self):
        """
        Run complete analysis for all symbols
        """
        print("=== COMPREHENSIVE MARKET DATA ANALYSIS ===\n")
        
        for symbol in self.symbols:
            print(f"\n{'='*60}")
            print(f"ANALYZING {symbol}")
            print(f"{'='*60}")
            
            # Load and transform data
            df = self.load_and_transform_data(symbol, sample_days=3)
            
            # Analyze market structure
            self.analyze_market_structure(symbol)
            
            # Analyze order book dynamics
            self.analyze_order_book_dynamics(symbol)
            
            # Create visualizations
            self.create_comprehensive_visualizations(symbol)
            
            print(f"\n{'-'*60}")
            print(f"COMPLETED ANALYSIS FOR {symbol}")
            print(f"{'-'*60}")
        
        # Create summary comparison
        self.create_summary_comparison()
    
    def create_summary_comparison(self):
        """
        Create summary comparison across all symbols
        """
        print("\n=== SUMMARY COMPARISON ACROSS SYMBOLS ===")
        
        summary_data = []
        for symbol in self.symbols:
            df = self.data[symbol]
            summary_data.append({
                'Symbol': symbol,
                'Records': len(df),
                'Avg_Price': df['mid_price'].mean(),
                'Price_Vol': df['mid_price'].std(),
                'Avg_Spread_BPS': df['spread_bps'].mean(),
                'Spread_Vol': df['spread_bps'].std(),
                'Avg_Depth': df['total_depth'].mean(),
                'Depth_Vol': df['total_depth'].std(),
                'Imbalance_Mean': df['bid_ask_imbalance'].mean(),
                'Imbalance_Vol': df['bid_ask_imbalance'].std()
            })
        
        summary_df = pd.DataFrame(summary_data)
        print("\nSummary Statistics:")
        print(summary_df.round(3))
        
        # Create comparison visualization
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Price comparison
        axes[0, 0].bar(summary_df['Symbol'], summary_df['Avg_Price'], 
                      color=['#ff9999', '#66b3ff', '#99ff99'], alpha=0.8)
        axes[0, 0].set_title('Average Mid Price by Symbol', fontweight='bold')
        axes[0, 0].set_ylabel('Price ($)')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Spread comparison
        axes[0, 1].bar(summary_df['Symbol'], summary_df['Avg_Spread_BPS'], 
                      color=['#ff9999', '#66b3ff', '#99ff99'], alpha=0.8)
        axes[0, 1].set_title('Average Spread by Symbol', fontweight='bold')
        axes[0, 1].set_ylabel('Spread (BPS)')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Depth comparison
        axes[1, 0].bar(summary_df['Symbol'], summary_df['Avg_Depth']/1000, 
                      color=['#ff9999', '#66b3ff', '#99ff99'], alpha=0.8)
        axes[1, 0].set_title('Average Order Book Depth by Symbol', fontweight='bold')
        axes[1, 0].set_ylabel('Depth (K shares)')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Imbalance comparison
        axes[1, 1].bar(summary_df['Symbol'], summary_df['Imbalance_Mean']*100, 
                      color=['#ff9999', '#66b3ff', '#99ff99'], alpha=0.8)
        axes[1, 1].set_title('Average Bid-Ask Imbalance by Symbol', fontweight='bold')
        axes[1, 1].set_ylabel('Imbalance (%)')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('symbol_comparison_summary.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("\nSummary comparison saved as 'symbol_comparison_summary.png'")
        
        return summary_df

def main():
    """
    Main execution function
    """
    print("Starting Comprehensive Market Data Analysis...")
    
    # Initialize explorer
    explorer = MarketDataExplorer()
    
    # Run complete analysis
    explorer.run_complete_analysis()
    
    print("\n=== ANALYSIS COMPLETE ===")
    print("Generated files:")
    print("- CRWV_comprehensive_analysis.png")
    print("- FROG_comprehensive_analysis.png") 
    print("- SOUN_comprehensive_analysis.png")
    print("- symbol_comparison_summary.png")

if __name__ == "__main__":
    main() 