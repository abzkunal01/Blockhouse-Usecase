#!/usr/bin/env python3
"""
Slippage Analysis and Optimal Order Execution Strategy
Blockhouse Work Trial Task

This script addresses the two main questions:
1. How to model the temporary impact function gt(x)
2. How to formulate an optimization framework for order allocation
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize, curve_fit
from scipy.stats import linregress
import glob
import os
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class SlippageAnalyzer:
    def __init__(self, data_dir="."):
        self.data_dir = data_dir
        self.symbols = ['CRWV', 'FROG', 'SOUN']
        self.data = {}
        self.impact_models = {}
        
    def load_data(self):
        """Load and preprocess all market data"""
        print("Loading market data...")
        
        for symbol in self.symbols:
            print(f"Processing {symbol}...")
            files = glob.glob(f"{symbol}/*.csv")
            files.sort()
            
            all_data = []
            for file in files:
                df = pd.read_csv(file)
                df['symbol'] = symbol
                # Handle datetime parsing with mixed formats
                df['date'] = pd.to_datetime(df['ts_event'], format='mixed').dt.date
                df['time'] = pd.to_datetime(df['ts_event'], format='mixed').dt.time
                all_data.append(df)
            
            self.data[symbol] = pd.concat(all_data, ignore_index=True)
            print(f"  Loaded {len(self.data[symbol])} records for {symbol}")
    
    def calculate_slippage(self, order_book_row, order_size, side='buy'):
        """Calculate slippage for a given order size and side"""
        if side == 'buy':
            # Hit the ask side
            total_slippage = 0
            remaining_size = order_size
            mid_price = (order_book_row['bid_px_00'] + order_book_row['ask_px_00']) / 2
            
            for level in range(10):
                ask_px_col = f'ask_px_{level:02d}'
                ask_sz_col = f'ask_sz_{level:02d}'
                
                if ask_px_col in order_book_row.index:
                    ask_price = order_book_row[ask_px_col]
                    ask_size = order_book_row[ask_sz_col]
                    
                    if ask_size > 0 and remaining_size > 0:
                        executed_size = min(remaining_size, ask_size)
                        level_slippage = executed_size * (ask_price - mid_price)
                        total_slippage += level_slippage
                        remaining_size -= executed_size
                        
                        if remaining_size <= 0:
                            break
            
            return total_slippage
        
        else:  # sell
            # Hit the bid side
            total_slippage = 0
            remaining_size = order_size
            mid_price = (order_book_row['bid_px_00'] + order_book_row['ask_px_00']) / 2
            
            for level in range(10):
                bid_px_col = f'bid_px_{level:02d}'
                bid_sz_col = f'bid_sz_{level:02d}'
                
                if bid_px_col in order_book_row.index:
                    bid_price = order_book_row[bid_px_col]
                    bid_size = order_book_row[bid_sz_col]
                    
                    if bid_size > 0 and remaining_size > 0:
                        executed_size = min(remaining_size, bid_size)
                        level_slippage = executed_size * (mid_price - bid_price)
                        total_slippage += level_slippage
                        remaining_size -= executed_size
                        
                        if remaining_size <= 0:
                            break
            
            return total_slippage
    
    def generate_impact_data(self, symbol, sample_size=1000):
        """Generate impact data for different order sizes"""
        print(f"Generating impact data for {symbol}...")
        
        df = self.data[symbol]
        sample_rows = df.sample(min(sample_size, len(df)))
        
        order_sizes = np.array([10, 25, 50, 100, 200, 500, 1000, 2000, 5000])
        impact_data = {'buy': [], 'sell': []}
        
        for _, row in sample_rows.iterrows():
            for size in order_sizes:
                buy_impact = self.calculate_slippage(row, size, 'buy')
                sell_impact = self.calculate_slippage(row, size, 'sell')
                
                impact_data['buy'].append({
                    'order_size': size,
                    'impact': buy_impact,
                    'timestamp': row['ts_event'],
                    'mid_price': (row['bid_px_00'] + row['ask_px_00']) / 2,
                    'spread': row['ask_px_00'] - row['bid_px_00']
                })
                
                impact_data['sell'].append({
                    'order_size': size,
                    'impact': sell_impact,
                    'timestamp': row['ts_event'],
                    'mid_price': (row['bid_px_00'] + row['ask_px_00']) / 2,
                    'spread': row['ask_px_00'] - row['bid_px_00']
                })
        
        return pd.DataFrame(impact_data['buy']), pd.DataFrame(impact_data['sell'])
    
    def fit_impact_models(self, buy_df, sell_df, symbol):
        """Fit different impact models to the data"""
        print(f"Fitting impact models for {symbol}...")
        
        models = {}
        
        # 1. Linear Model: g(x) = βx
        def linear_model(x, beta):
            return beta * x
        
        # 2. Square Root Model: g(x) = α√x + βx
        def sqrt_model(x, alpha, beta):
            return alpha * np.sqrt(x) + beta * x
        
        # 3. Power Law Model: g(x) = αx^γ
        def power_model(x, alpha, gamma):
            return alpha * (x ** gamma)
        
        # 4. Quadratic Model: g(x) = αx + βx²
        def quadratic_model(x, alpha, beta):
            return alpha * x + beta * (x ** 2)
        
        # Fit models for buy side
        x_data = buy_df['order_size'].values
        y_data = buy_df['impact'].values
        
        try:
            # Linear
            popt_linear, _ = curve_fit(linear_model, x_data, y_data)
            models['buy_linear'] = {'func': linear_model, 'params': popt_linear}
            
            # Square Root
            popt_sqrt, _ = curve_fit(sqrt_model, x_data, y_data)
            models['buy_sqrt'] = {'func': sqrt_model, 'params': popt_sqrt}
            
            # Power Law
            popt_power, _ = curve_fit(power_model, x_data, y_data)
            models['buy_power'] = {'func': power_model, 'params': popt_power}
            
            # Quadratic
            popt_quad, _ = curve_fit(quadratic_model, x_data, y_data)
            models['buy_quadratic'] = {'func': quadratic_model, 'params': popt_quad}
            
        except Exception as e:
            print(f"Error fitting buy models: {e}")
        
        # Fit models for sell side
        y_data = sell_df['impact'].values
        
        try:
            # Linear
            popt_linear, _ = curve_fit(linear_model, x_data, y_data)
            models['sell_linear'] = {'func': linear_model, 'params': popt_linear}
            
            # Square Root
            popt_sqrt, _ = curve_fit(sqrt_model, x_data, y_data)
            models['sell_sqrt'] = {'func': sqrt_model, 'params': popt_sqrt}
            
            # Power Law
            popt_power, _ = curve_fit(power_model, x_data, y_data)
            models['sell_power'] = {'func': power_model, 'params': popt_power}
            
            # Quadratic
            popt_quad, _ = curve_fit(quadratic_model, x_data, y_data)
            models['sell_quadratic'] = {'func': quadratic_model, 'params': popt_quad}
            
        except Exception as e:
            print(f"Error fitting sell models: {e}")
        
        return models
    
    def evaluate_models(self, models, buy_df, sell_df, symbol):
        """Evaluate model performance using R-squared"""
        print(f"Evaluating models for {symbol}...")
        
        x_data = buy_df['order_size'].values
        
        results = {}
        
        for model_name, model_info in models.items():
            func = model_info['func']
            params = model_info['params']
            
            if 'buy' in model_name:
                y_true = buy_df['impact'].values
            else:
                y_true = sell_df['impact'].values
            
            y_pred = func(x_data, *params)
            
            # Calculate R-squared
            ss_res = np.sum((y_true - y_pred) ** 2)
            ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
            r_squared = 1 - (ss_res / ss_tot)
            
            results[model_name] = {
                'r_squared': r_squared,
                'params': params,
                'predictions': y_pred
            }
        
        return results
    
    def plot_impact_analysis(self, buy_df, sell_df, models, results, symbol):
        """Plot impact analysis results"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Impact Analysis for {symbol}', fontsize=16)
        
        x_data = buy_df['order_size'].values
        x_smooth = np.linspace(x_data.min(), x_data.max(), 100)
        
        # Buy side analysis
        axes[0, 0].scatter(buy_df['order_size'], buy_df['impact'], alpha=0.6, label='Data')
        axes[0, 0].set_title('Buy Side Impact')
        axes[0, 0].set_xlabel('Order Size')
        axes[0, 0].set_ylabel('Slippage ($)')
        
        # Plot fitted models for buy side
        colors = ['red', 'blue', 'green', 'orange']
        buy_models = [(name, info) for name, info in models.items() if 'buy' in name]
        for i, (model_name, model_info) in enumerate(buy_models):
            func = model_info['func']
            params = model_info['params']
            r_sq = results[model_name]['r_squared']
            
            y_smooth = func(x_smooth, *params)
            axes[0, 0].plot(x_smooth, y_smooth, color=colors[i], 
                           label=f'{model_name.split("_")[1]} (R²={r_sq:.3f})')
        
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Sell side analysis
        axes[0, 1].scatter(sell_df['order_size'], sell_df['impact'], alpha=0.6, label='Data')
        axes[0, 1].set_title('Sell Side Impact')
        axes[0, 1].set_xlabel('Order Size')
        axes[0, 1].set_ylabel('Slippage ($)')
        
        # Plot fitted models for sell side
        sell_models = [(name, info) for name, info in models.items() if 'sell' in name]
        for i, (model_name, model_info) in enumerate(sell_models):
            func = model_info['func']
            params = model_info['params']
            r_sq = results[model_name]['r_squared']
            
            y_smooth = func(x_smooth, *params)
            axes[0, 1].plot(x_smooth, y_smooth, color=colors[i], 
                           label=f'{model_name.split("_")[1]} (R²={r_sq:.3f})')
        
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Model comparison
        model_names = [name.split('_')[1] for name in results.keys() if 'buy' in name]
        buy_r_squared = [results[f'buy_{name}']['r_squared'] for name in model_names]
        sell_r_squared = [results[f'sell_{name}']['r_squared'] for name in model_names]
        
        x_pos = np.arange(len(model_names))
        width = 0.35
        
        axes[1, 0].bar(x_pos - width/2, buy_r_squared, width, label='Buy Side', alpha=0.8)
        axes[1, 0].bar(x_pos + width/2, sell_r_squared, width, label='Sell Side', alpha=0.8)
        axes[1, 0].set_title('Model Performance Comparison')
        axes[1, 0].set_xlabel('Model Type')
        axes[1, 0].set_ylabel('R-squared')
        axes[1, 0].set_xticks(x_pos)
        axes[1, 0].set_xticklabels(model_names)
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Impact vs Market Conditions
        axes[1, 1].scatter(buy_df['spread'], buy_df['impact'], alpha=0.6, label='Buy')
        axes[1, 1].scatter(sell_df['spread'], sell_df['impact'], alpha=0.6, label='Sell')
        axes[1, 1].set_title('Impact vs Bid-Ask Spread')
        axes[1, 1].set_xlabel('Spread ($)')
        axes[1, 1].set_ylabel('Slippage ($)')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{symbol}_impact_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def run_analysis(self):
        """Run complete analysis for all symbols"""
        self.load_data()
        
        for symbol in self.symbols:
            print(f"\n{'='*60}")
            print(f"ANALYZING {symbol}")
            print(f"{'='*60}")
            
            # Generate impact data
            buy_df, sell_df = self.generate_impact_data(symbol)
            
            # Fit models
            models = self.fit_impact_models(buy_df, sell_df, symbol)
            
            # Evaluate models
            results = self.evaluate_models(models, buy_df, sell_df, symbol)
            
            # Print results
            print(f"\nModel Performance for {symbol}:")
            print("-" * 50)
            for model_name, result in results.items():
                print(f"{model_name:20} R² = {result['r_squared']:.4f}")
            
            # Plot results
            self.plot_impact_analysis(buy_df, sell_df, models, results, symbol)
            
            # Store best model
            best_buy = max([(k, v['r_squared']) for k, v in results.items() if 'buy' in k], 
                          key=lambda x: x[1])
            best_sell = max([(k, v['r_squared']) for k, v in results.items() if 'sell' in k], 
                           key=lambda x: x[1])
            
            self.impact_models[symbol] = {
                'best_buy_model': best_buy[0],
                'best_sell_model': best_sell[0],
                'models': models,
                'results': results
            }
            
            print(f"\nBest buy model: {best_buy[0]} (R² = {best_buy[1]:.4f})")
            print(f"Best sell model: {best_sell[0]} (R² = {best_sell[1]:.4f})")

if __name__ == "__main__":
    # Run the analysis
    analyzer = SlippageAnalyzer()
    analyzer.run_analysis() 