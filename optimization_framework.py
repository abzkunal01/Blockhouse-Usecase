#!/usr/bin/env python3
"""
Optimal Order Execution Framework
Blockhouse Work Trial Task - Question 2

This script implements the mathematical framework for optimal order allocation
to minimize total temporary impact (slippage) across multiple trading periods.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, differential_evolution
from scipy.stats import linregress
import warnings
warnings.filterwarnings('ignore')

class OptimalExecutionFramework:
    def __init__(self, impact_models):
        """
        Initialize the optimization framework
        
        Args:
            impact_models: Dictionary containing fitted impact models for each symbol
        """
        self.impact_models = impact_models
        self.N = 390  # Number of trading periods (one-minute windows)
        
    def impact_function(self, x, model_type, params, side='buy'):
        """
        Calculate temporary impact for order size x using specified model
        
        Args:
            x: Order size
            model_type: Type of model ('linear', 'sqrt', 'power', 'quadratic')
            params: Model parameters
            side: 'buy' or 'sell'
        
        Returns:
            Temporary impact (slippage cost)
        """
        if model_type == 'linear':
            return params[0] * x
        elif model_type == 'sqrt':
            return params[0] * np.sqrt(x) + params[1] * x
        elif model_type == 'power':
            return params[0] * (x ** params[1])
        elif model_type == 'quadratic':
            return params[0] * x + params[1] * (x ** 2)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def objective_function(self, x, impact_params, side='buy'):
        """
        Objective function: total temporary impact across all periods
        
        Args:
            x: Allocation vector (order sizes for each period)
            impact_params: Dictionary with model type and parameters
            side: 'buy' or 'sell'
        
        Returns:
            Total slippage cost
        """
        total_impact = 0
        model_type = impact_params['model_type']
        params = impact_params['params']
        
        for xi in x:
            if xi > 0:  # Only calculate impact for non-zero orders
                impact = self.impact_function(xi, model_type, params, side)
                total_impact += impact
        
        return total_impact
    
    def constraint_function(self, x, total_shares):
        """
        Constraint function: total shares must equal target
        
        Args:
            x: Allocation vector
            total_shares: Target total shares to execute
        
        Returns:
            Constraint violation (should be 0)
        """
        return np.sum(x) - total_shares
    
    def optimize_allocation(self, symbol, total_shares, side='buy', method='SLSQP'):
        """
        Optimize order allocation to minimize total impact
        
        Args:
            symbol: Stock symbol
            total_shares: Total shares to execute
            side: 'buy' or 'sell'
            method: Optimization method ('SLSQP', 'differential_evolution')
        
        Returns:
            Dictionary with optimization results
        """
        print(f"Optimizing {side} allocation for {symbol} ({total_shares} shares)")
        
        # Get impact model parameters
        if side == 'buy':
            model_key = self.impact_models[symbol]['best_buy_model']
        else:
            model_key = self.impact_models[symbol]['best_sell_model']
        
        model_type = model_key.split('_')[1]  # Extract model type
        params = self.impact_models[symbol]['results'][model_key]['params']
        
        impact_params = {
            'model_type': model_type,
            'params': params
        }
        
        # Initial guess: equal allocation
        x0 = np.full(self.N, total_shares / self.N)
        
        # Bounds: non-negative orders
        bounds = [(0, total_shares)] * self.N
        
        # Constraints
        constraints = {
            'type': 'eq',
            'fun': lambda x: self.constraint_function(x, total_shares)
        }
        
        if method == 'SLSQP':
            # Sequential Least Squares Programming
            result = minimize(
                fun=lambda x: self.objective_function(x, impact_params, side),
                x0=x0,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 1000}
            )
        elif method == 'differential_evolution':
            # Differential evolution (global optimization)
            result = differential_evolution(
                func=lambda x: self.objective_function(x, impact_params, side),
                bounds=bounds,
                constraints=constraints,
                maxiter=1000,
                popsize=15
            )
        else:
            raise ValueError(f"Unknown optimization method: {method}")
        
        # Calculate results
        optimal_allocation = result.x
        total_impact = result.fun
        success = result.success
        
        # Calculate statistics
        non_zero_orders = optimal_allocation[optimal_allocation > 0]
        avg_order_size = np.mean(non_zero_orders) if len(non_zero_orders) > 0 else 0
        max_order_size = np.max(optimal_allocation)
        min_order_size = np.min(non_zero_orders) if len(non_zero_orders) > 0 else 0
        
        results = {
            'symbol': symbol,
            'side': side,
            'total_shares': total_shares,
            'optimal_allocation': optimal_allocation,
            'total_impact': total_impact,
            'success': success,
            'avg_order_size': avg_order_size,
            'max_order_size': max_order_size,
            'min_order_size': min_order_size,
            'num_periods_used': len(non_zero_orders),
            'model_type': model_type,
            'model_params': params
        }
        
        return results
    
    def compare_strategies(self, symbol, total_shares, side='buy'):
        """
        Compare different execution strategies
        
        Args:
            symbol: Stock symbol
            total_shares: Total shares to execute
            side: 'buy' or 'sell'
        
        Returns:
            Dictionary with comparison results
        """
        print(f"\nComparing execution strategies for {symbol} ({side} {total_shares} shares)")
        
        strategies = {}
        
        # 1. Optimal strategy
        optimal_result = self.optimize_allocation(symbol, total_shares, side)
        strategies['Optimal'] = optimal_result
        
        # 2. Equal allocation (naive strategy)
        equal_allocation = np.full(self.N, total_shares / self.N)
        if side == 'buy':
            model_key = self.impact_models[symbol]['best_buy_model']
        else:
            model_key = self.impact_models[symbol]['best_sell_model']
        
        model_type = model_key.split('_')[1]
        params = self.impact_models[symbol]['results'][model_key]['params']
        
        equal_impact = self.objective_function(
            equal_allocation, 
            {'model_type': model_type, 'params': params}, 
            side
        )
        
        strategies['Equal Allocation'] = {
            'symbol': symbol,
            'side': side,
            'total_shares': total_shares,
            'optimal_allocation': equal_allocation,
            'total_impact': equal_impact,
            'success': True,
            'avg_order_size': total_shares / self.N,
            'max_order_size': total_shares / self.N,
            'min_order_size': total_shares / self.N,
            'num_periods_used': self.N,
            'model_type': model_type,
            'model_params': params
        }
        
        # 3. Front-loaded strategy (execute more early)
        front_loaded = np.zeros(self.N)
        front_loaded[:50] = total_shares / 50  # Execute in first 50 periods
        front_impact = self.objective_function(
            front_loaded, 
            {'model_type': model_type, 'params': params}, 
            side
        )
        
        strategies['Front-loaded'] = {
            'symbol': symbol,
            'side': side,
            'total_shares': total_shares,
            'optimal_allocation': front_loaded,
            'total_impact': front_impact,
            'success': True,
            'avg_order_size': total_shares / 50,
            'max_order_size': total_shares / 50,
            'min_order_size': 0,
            'num_periods_available': 50,
            'model_type': model_type,
            'model_params': params
        }
        
        # 4. Back-loaded strategy (execute more late)
        back_loaded = np.zeros(self.N)
        back_loaded[-50:] = total_shares / 50  # Execute in last 50 periods
        back_impact = self.objective_function(
            back_loaded, 
            {'model_type': model_type, 'params': params}, 
            side
        )
        
        strategies['Back-loaded'] = {
            'symbol': symbol,
            'side': side,
            'total_shares': total_shares,
            'optimal_allocation': back_loaded,
            'total_impact': back_impact,
            'success': True,
            'avg_order_size': total_shares / 50,
            'max_order_size': total_shares / 50,
            'min_order_size': 0,
            'num_periods_available': 50,
            'model_type': model_type,
            'model_params': params
        }
        
        return strategies
    
    def plot_allocation_comparison(self, strategies, symbol, side):
        """Plot comparison of different allocation strategies"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Execution Strategy Comparison - {symbol} ({side})', fontsize=16)
        
        strategy_names = list(strategies.keys())
        impacts = [strategies[name]['total_impact'] for name in strategy_names]
        
        # Impact comparison
        axes[0, 0].bar(strategy_names, impacts, color=['red', 'blue', 'green', 'orange'])
        axes[0, 0].set_title('Total Impact Comparison')
        axes[0, 0].set_ylabel('Total Slippage ($)')
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].grid(True, alpha=0.3)
        
        # Add value labels on bars
        for i, v in enumerate(impacts):
            axes[0, 0].text(i, v + max(impacts) * 0.01, f'${v:.2f}', 
                           ha='center', va='bottom', fontweight='bold')
        
        # Allocation patterns
        colors = ['red', 'blue', 'green', 'orange']
        for i, (name, strategy) in enumerate(strategies.items()):
            allocation = strategy['optimal_allocation']
            axes[0, 1].plot(allocation, label=name, color=colors[i], alpha=0.8)
        
        axes[0, 1].set_title('Allocation Patterns Over Time')
        axes[0, 1].set_xlabel('Trading Period')
        axes[0, 1].set_ylabel('Order Size')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Order size statistics
        avg_sizes = [strategies[name]['avg_order_size'] for name in strategy_names]
        max_sizes = [strategies[name]['max_order_size'] for name in strategy_names]
        
        x_pos = np.arange(len(strategy_names))
        width = 0.35
        
        axes[1, 0].bar(x_pos - width/2, avg_sizes, width, label='Average Order Size', alpha=0.8)
        axes[1, 0].bar(x_pos + width/2, max_sizes, width, label='Maximum Order Size', alpha=0.8)
        axes[1, 0].set_title('Order Size Statistics')
        axes[1, 0].set_xlabel('Strategy')
        axes[1, 0].set_ylabel('Order Size')
        axes[1, 0].set_xticks(x_pos)
        axes[1, 0].set_xticklabels(strategy_names, rotation=45)
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Impact savings
        baseline_impact = strategies['Equal Allocation']['total_impact']
        savings = [(baseline_impact - impact) / baseline_impact * 100 for impact in impacts]
        
        axes[1, 1].bar(strategy_names, savings, color=['red', 'blue', 'green', 'orange'])
        axes[1, 1].set_title('Impact Savings vs Equal Allocation')
        axes[1, 1].set_ylabel('Savings (%)')
        axes[1, 1].tick_params(axis='x', rotation=45)
        axes[1, 1].grid(True, alpha=0.3)
        
        # Add value labels
        for i, v in enumerate(savings):
            axes[1, 1].text(i, v + max(savings) * 0.01, f'{v:.1f}%', 
                           ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f'{symbol}_{side}_strategy_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def sensitivity_analysis(self, symbol, total_shares, side='buy'):
        """
        Perform sensitivity analysis on optimal allocation
        
        Args:
            symbol: Stock symbol
            total_shares: Base total shares
            side: 'buy' or 'sell'
        """
        print(f"\nPerforming sensitivity analysis for {symbol} ({side})")
        
        # Test different total share amounts
        share_amounts = [total_shares * 0.5, total_shares * 0.75, total_shares, 
                        total_shares * 1.25, total_shares * 1.5]
        
        results = []
        for shares in share_amounts:
            result = self.optimize_allocation(symbol, shares, side)
            results.append({
                'total_shares': shares,
                'total_impact': result['total_impact'],
                'avg_order_size': result['avg_order_size'],
                'max_order_size': result['max_order_size'],
                'num_periods_used': result['num_periods_used']
            })
        
        # Plot sensitivity results
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Sensitivity Analysis - {symbol} ({side})', fontsize=16)
        
        share_values = [r['total_shares'] for r in results]
        impact_values = [r['total_impact'] for r in results]
        avg_sizes = [r['avg_order_size'] for r in results]
        max_sizes = [r['max_order_size'] for r in results]
        
        # Impact vs total shares
        axes[0, 0].plot(share_values, impact_values, 'o-', linewidth=2, markersize=8)
        axes[0, 0].set_title('Total Impact vs Total Shares')
        axes[0, 0].set_xlabel('Total Shares')
        axes[0, 0].set_ylabel('Total Impact ($)')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Average order size vs total shares
        axes[0, 1].plot(share_values, avg_sizes, 'o-', linewidth=2, markersize=8)
        axes[0, 1].set_title('Average Order Size vs Total Shares')
        axes[0, 1].set_xlabel('Total Shares')
        axes[0, 1].set_ylabel('Average Order Size')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Maximum order size vs total shares
        axes[1, 0].plot(share_values, max_sizes, 'o-', linewidth=2, markersize=8)
        axes[1, 0].set_title('Maximum Order Size vs Total Shares')
        axes[1, 0].set_xlabel('Total Shares')
        axes[1, 0].set_ylabel('Maximum Order Size')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Impact per share
        impact_per_share = [impact / shares for impact, shares in zip(impact_values, share_values)]
        axes[1, 1].plot(share_values, impact_per_share, 'o-', linewidth=2, markersize=8)
        axes[1, 1].set_title('Impact per Share vs Total Shares')
        axes[1, 1].set_xlabel('Total Shares')
        axes[1, 1].set_ylabel('Impact per Share ($)')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{symbol}_{side}_sensitivity_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return results

def main():
    """Main function to demonstrate the optimization framework"""
    print("Optimal Order Execution Framework")
    print("=" * 50)
    
    # This would typically be loaded from the slippage analysis
    # For demonstration, we'll create a simple example
    print("Note: This framework requires impact models from slippage_analysis.py")
    print("Please run slippage_analysis.py first to generate the required models.")
    
    # Example usage (when impact models are available):
    """
    # Load impact models from slippage analysis
    analyzer = SlippageAnalyzer()
    analyzer.run_analysis()
    
    # Create optimization framework
    framework = OptimalExecutionFramework(analyzer.impact_models)
    
    # Test with different scenarios
    test_cases = [
        ('CRWV', 1000, 'buy'),
        ('FROG', 500, 'sell'),
        ('SOUN', 2000, 'buy')
    ]
    
    for symbol, shares, side in test_cases:
        # Compare strategies
        strategies = framework.compare_strategies(symbol, shares, side)
        framework.plot_allocation_comparison(strategies, symbol, side)
        
        # Sensitivity analysis
        framework.sensitivity_analysis(symbol, shares, side)
    """

if __name__ == "__main__":
    main() 