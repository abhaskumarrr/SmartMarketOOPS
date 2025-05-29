"""
Backtesting Utilities

This module provides utility functions for backtesting, including visualization
and data handling.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from typing import Dict, List, Any, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def plot_backtest_results(
    results: Dict[str, Any],
    output_dir: Optional[str] = None,
    plot_trades: bool = True,
    plot_equity: bool = True,
    plot_drawdown: bool = True
) -> Dict[str, plt.Figure]:
    """
    Plot comprehensive backtest results.
    
    Args:
        results: Dictionary of backtest results
        output_dir: Directory to save plots (if None, plots are displayed)
        plot_trades: Whether to plot individual trades
        plot_equity: Whether to plot equity curve
        plot_drawdown: Whether to plot drawdown chart
        
    Returns:
        Dictionary of created figures
    """
    figures = {}
    
    # Check if we have the necessary data
    if 'equity_curve' not in results or not results['equity_curve']:
        logger.warning("No equity curve data available for plotting")
        return figures
    
    # Plot equity curve
    if plot_equity:
        fig_equity = plot_equity_curve(results['equity_curve'], results['symbol'])
        figures['equity'] = fig_equity
        
        if output_dir:
            equity_path = os.path.join(output_dir, f"{results['symbol']}_equity.png")
            fig_equity.savefig(equity_path, dpi=300, bbox_inches='tight')
            logger.info(f"Equity curve plot saved to {equity_path}")
    
    # Plot drawdown
    if plot_drawdown:
        fig_drawdown = plot_drawdown_chart(results['equity_curve'])
        figures['drawdown'] = fig_drawdown
        
        if output_dir:
            drawdown_path = os.path.join(output_dir, f"{results['symbol']}_drawdown.png")
            fig_drawdown.savefig(drawdown_path, dpi=300, bbox_inches='tight')
            logger.info(f"Drawdown plot saved to {drawdown_path}")
    
    # Plot trades if available
    if plot_trades and 'trades' in results and results['trades']:
        fig_trades = plot_trades_chart(results)
        figures['trades'] = fig_trades
        
        if output_dir:
            trades_path = os.path.join(output_dir, f"{results['symbol']}_trades.png")
            fig_trades.savefig(trades_path, dpi=300, bbox_inches='tight')
            logger.info(f"Trades plot saved to {trades_path}")
    
    # Display plots if not saving to files
    if not output_dir:
        plt.show()
    
    return figures


def plot_equity_curve(equity_curve: List[Dict[str, Any]], symbol: str = '') -> plt.Figure:
    """
    Plot equity curve from backtest results.
    
    Args:
        equity_curve: List of equity points
        symbol: Trading symbol for title
        
    Returns:
        Matplotlib figure
    """
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Convert to DataFrame for plotting
    equity_df = pd.DataFrame(equity_curve)
    
    # Convert timestamps to datetime if needed
    if 'timestamp' in equity_df.columns and not pd.api.types.is_datetime64_any_dtype(equity_df['timestamp']):
        equity_df['timestamp'] = pd.to_datetime(equity_df['timestamp'])
    
    # Filter out the first row with None timestamp
    if 'timestamp' in equity_df.columns:
        equity_df = equity_df[equity_df['timestamp'].notna()]
    
    # Plot equity curve
    if 'timestamp' in equity_df.columns:
        # Plot with timestamps on x-axis
        ax.plot(equity_df['timestamp'], equity_df['equity'], 'b-', linewidth=1.5, label='Equity')
        
        # Add cash if available
        if 'cash' in equity_df.columns:
            ax.plot(equity_df['timestamp'], equity_df['cash'], 'g--', linewidth=1, alpha=0.7, label='Cash')
        
        # Format x-axis for dates
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.xticks(rotation=45)
    else:
        # Plot without timestamps (use indices)
        ax.plot(equity_df['equity'], 'b-', linewidth=1.5, label='Equity')
        
        # Add cash if available
        if 'cash' in equity_df.columns:
            ax.plot(equity_df['cash'], 'g--', linewidth=1, alpha=0.7, label='Cash')
    
    # Set labels and title
    ax.set_title(f'Equity Curve - {symbol}' if symbol else 'Equity Curve')
    ax.set_xlabel('Date/Time')
    ax.set_ylabel('Equity')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')
    
    # Add initial and final equity values
    if len(equity_df) > 1:
        initial_equity = equity_df['equity'].iloc[0]
        final_equity = equity_df['equity'].iloc[-1]
        total_return = (final_equity / initial_equity - 1) * 100
        
        # Add text with key metrics
        text = (
            f"Initial: ${initial_equity:.2f}\n"
            f"Final: ${final_equity:.2f}\n"
            f"Return: {total_return:.2f}%"
        )
        
        # Add text box
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(0.02, 0.95, text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=props)
    
    # Adjust layout
    plt.tight_layout()
    
    return fig


def plot_drawdown_chart(equity_curve: List[Dict[str, Any]]) -> plt.Figure:
    """
    Plot drawdown chart from equity curve.
    
    Args:
        equity_curve: List of equity points
        
    Returns:
        Matplotlib figure
    """
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Convert to DataFrame for plotting
    equity_df = pd.DataFrame(equity_curve)
    
    # Convert timestamps to datetime if needed
    if 'timestamp' in equity_df.columns and not pd.api.types.is_datetime64_any_dtype(equity_df['timestamp']):
        equity_df['timestamp'] = pd.to_datetime(equity_df['timestamp'])
    
    # Filter out the first row with None timestamp
    if 'timestamp' in equity_df.columns:
        equity_df = equity_df[equity_df['timestamp'].notna()]
    
    # Calculate running maximum
    equity_values = equity_df['equity'].values
    running_max = np.maximum.accumulate(equity_values)
    
    # Calculate drawdown in percentage terms
    drawdown = (running_max - equity_values) / running_max * 100
    
    # Add drawdown to DataFrame
    equity_df['drawdown'] = drawdown
    
    # Plot drawdown
    if 'timestamp' in equity_df.columns:
        # Plot with timestamps on x-axis
        ax.fill_between(equity_df['timestamp'], 0, equity_df['drawdown'], color='red', alpha=0.3)
        ax.plot(equity_df['timestamp'], equity_df['drawdown'], 'r-', linewidth=1)
        
        # Format x-axis for dates
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.xticks(rotation=45)
    else:
        # Plot without timestamps (use indices)
        ax.fill_between(range(len(equity_df)), 0, equity_df['drawdown'], color='red', alpha=0.3)
        ax.plot(equity_df['drawdown'], 'r-', linewidth=1)
    
    # Invert y-axis for better visualization (drawdown is negative)
    ax.invert_yaxis()
    
    # Set labels and title
    ax.set_title('Drawdown Chart')
    ax.set_xlabel('Date/Time')
    ax.set_ylabel('Drawdown (%)')
    ax.grid(True, alpha=0.3)
    
    # Add max drawdown value
    max_dd = np.max(drawdown)
    max_dd_idx = np.argmax(drawdown)
    
    # Add text with max drawdown
    props = dict(boxstyle='round', facecolor='white', alpha=0.8)
    ax.text(0.02, 0.95, f"Max Drawdown: {max_dd:.2f}%", transform=ax.transAxes,
            fontsize=10, verticalalignment='top', bbox=props)
    
    # Adjust layout
    plt.tight_layout()
    
    return fig


def plot_trades_chart(results: Dict[str, Any]) -> plt.Figure:
    """
    Plot trades and equity curve with entry/exit points.
    
    Args:
        results: Dictionary of backtest results
        
    Returns:
        Matplotlib figure
    """
    # Extract data
    equity_curve = results['equity_curve']
    trades = results['trades']
    symbol = results.get('symbol', '')
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Convert equity curve to DataFrame
    equity_df = pd.DataFrame(equity_curve)
    
    # Convert timestamps to datetime if needed
    if 'timestamp' in equity_df.columns and not pd.api.types.is_datetime64_any_dtype(equity_df['timestamp']):
        equity_df['timestamp'] = pd.to_datetime(equity_df['timestamp'])
    
    # Filter out the first row with None timestamp
    if 'timestamp' in equity_df.columns:
        equity_df = equity_df[equity_df['timestamp'].notna()]
    
    # Plot equity curve
    if 'timestamp' in equity_df.columns:
        ax.plot(equity_df['timestamp'], equity_df['equity'], 'b-', linewidth=1.5, label='Equity')
    else:
        ax.plot(equity_df['equity'], 'b-', linewidth=1.5, label='Equity')
    
    # Process trades data
    trades_df = pd.DataFrame(trades)
    
    # Convert timestamps to datetime if needed
    if 'entry_time' in trades_df.columns and not pd.api.types.is_datetime64_any_dtype(trades_df['entry_time']):
        trades_df['entry_time'] = pd.to_datetime(trades_df['entry_time'])
    
    if 'exit_time' in trades_df.columns and not pd.api.types.is_datetime64_any_dtype(trades_df['exit_time']):
        trades_df['exit_time'] = pd.to_datetime(trades_df['exit_time'])
    
    # Prepare for plotting trades
    if 'entry_time' in trades_df.columns and 'exit_time' in trades_df.columns and 'timestamp' in equity_df.columns:
        # Plot entry and exit points on the equity curve
        
        # Get equity values at entry and exit times
        entries = []
        exits = []
        
        for _, trade in trades_df.iterrows():
            # Find closest equity points to entry and exit times
            entry_idx = (equity_df['timestamp'] - trade['entry_time']).abs().idxmin()
            exit_idx = (equity_df['timestamp'] - trade['exit_time']).abs().idxmin()
            
            entries.append((trade['entry_time'], equity_df.loc[entry_idx, 'equity'], trade['direction']))
            exits.append((trade['exit_time'], equity_df.loc[exit_idx, 'equity'], trade['pnl'] > 0))
        
        # Plot entry points
        for entry_time, entry_equity, direction in entries:
            color = 'g' if direction == 'long' else 'r'
            marker = '^' if direction == 'long' else 'v'
            ax.plot(entry_time, entry_equity, marker=marker, color=color, markersize=8, alpha=0.7)
        
        # Plot exit points
        for exit_time, exit_equity, is_profit in exits:
            color = 'g' if is_profit else 'r'
            marker = 'o'
            ax.plot(exit_time, exit_equity, marker=marker, color=color, markersize=6, alpha=0.7)
    
    # Set labels and title
    ax.set_title(f'Trades Overview - {symbol}' if symbol else 'Trades Overview')
    ax.set_xlabel('Date/Time')
    ax.set_ylabel('Equity')
    ax.grid(True, alpha=0.3)
    
    # Add custom legend
    import matplotlib.lines as mlines
    
    equity_line = mlines.Line2D([], [], color='b', linewidth=1.5, label='Equity')
    long_entry = mlines.Line2D([], [], color='g', marker='^', linestyle='None', markersize=8, label='Long Entry')
    short_entry = mlines.Line2D([], [], color='r', marker='v', linestyle='None', markersize=8, label='Short Entry')
    profit_exit = mlines.Line2D([], [], color='g', marker='o', linestyle='None', markersize=6, label='Profit Exit')
    loss_exit = mlines.Line2D([], [], color='r', marker='o', linestyle='None', markersize=6, label='Loss Exit')
    
    ax.legend(handles=[equity_line, long_entry, short_entry, profit_exit, loss_exit], loc='best')
    
    # Adjust layout
    plt.tight_layout()
    
    # Format x-axis for dates
    if 'timestamp' in equity_df.columns:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.xticks(rotation=45)
    
    # Add text with trade statistics
    if len(trades_df) > 0:
        win_trades = trades_df[trades_df['pnl'] > 0]
        loss_trades = trades_df[trades_df['pnl'] <= 0]
        
        win_rate = len(win_trades) / len(trades_df) if len(trades_df) > 0 else 0
        
        # Calculate profit factor
        profit_factor = (
            win_trades['pnl'].sum() / abs(loss_trades['pnl'].sum())
            if len(loss_trades) > 0 and loss_trades['pnl'].sum() != 0
            else float('inf') if len(win_trades) > 0
            else 0
        )
        
        # Create text box content
        text = (
            f"Trades: {len(trades_df)}\n"
            f"Win Rate: {win_rate:.2%}\n"
            f"Profit Factor: {profit_factor:.2f}"
        )
        
        # Add text box
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(0.02, 0.95, text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=props)
    
    return fig


def plot_monthly_returns_heatmap(results: Dict[str, Any]) -> plt.Figure:
    """
    Plot monthly returns as a heatmap.
    
    Args:
        results: Dictionary of backtest results
        
    Returns:
        Matplotlib figure
    """
    # Extract equity curve
    equity_curve = results['equity_curve']
    
    # Convert to DataFrame
    equity_df = pd.DataFrame(equity_curve)
    
    # Convert timestamps to datetime if needed
    if 'timestamp' in equity_df.columns and not pd.api.types.is_datetime64_any_dtype(equity_df['timestamp']):
        equity_df['timestamp'] = pd.to_datetime(equity_df['timestamp'])
    
    # Filter out the first row with None timestamp
    if 'timestamp' in equity_df.columns:
        equity_df = equity_df[equity_df['timestamp'].notna()]
    
    # Check if we have timestamp data
    if 'timestamp' not in equity_df.columns:
        logger.warning("No timestamp data available for monthly returns heatmap")
        return None
    
    # Set timestamp as index
    equity_df.set_index('timestamp', inplace=True)
    
    # Extract equity values
    equity_values = equity_df['equity']
    
    # Resample to month-end
    monthly_equity = equity_values.resample('M').last()
    
    # Calculate monthly returns
    monthly_returns = monthly_equity.pct_change().dropna() * 100  # as percentage
    
    # Create a pivot table with year and month
    returns_table = pd.DataFrame({
        'year': monthly_returns.index.year,
        'month': monthly_returns.index.month,
        'return': monthly_returns.values
    })
    
    # Pivot the table
    pivot_table = returns_table.pivot_table(
        index='year', 
        columns='month', 
        values='return',
        aggfunc='first'
    )
    
    # Map month numbers to names
    month_names = {
        1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
        7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'
    }
    pivot_table.columns = [month_names[m] for m in pivot_table.columns]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create heatmap
    cmap = plt.cm.RdYlGn  # Red (negative) to Green (positive)
    
    # Find min and max returns for consistent color scale
    vmin = min(-5, pivot_table.min().min())  # At least -5%
    vmax = max(5, pivot_table.max().max())   # At least 5%
    
    # Plot heatmap
    im = ax.imshow(pivot_table, cmap=cmap, vmin=vmin, vmax=vmax)
    
    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('Return (%)', rotation=-90, va="bottom")
    
    # Add axis labels
    ax.set_xticks(np.arange(len(pivot_table.columns)))
    ax.set_yticks(np.arange(len(pivot_table.index)))
    ax.set_xticklabels(pivot_table.columns)
    ax.set_yticklabels(pivot_table.index)
    
    # Add text annotations
    for i in range(len(pivot_table.index)):
        for j in range(len(pivot_table.columns)):
            value = pivot_table.iloc[i, j]
            if not np.isnan(value):
                # Choose text color based on background
                text_color = 'white' if abs(value) > (vmax - vmin) / 2 else 'black'
                ax.text(j, i, f"{value:.1f}%", ha="center", va="center", color=text_color)
            else:
                ax.text(j, i, "N/A", ha="center", va="center", color="gray")
    
    # Set title and labels
    ax.set_title('Monthly Returns (%)')
    
    # Add yearly stats
    yearly_returns = pivot_table.mean(axis=1)
    yearly_win_rates = (pivot_table > 0).mean(axis=1) * 100
    
    # Add yearly stats as extra column
    ax.text(len(pivot_table.columns) + 0.5, -0.5, "Year Avg", ha="center", va="center", fontweight='bold')
    
    for i, year in enumerate(pivot_table.index):
        year_return = yearly_returns[year]
        year_win_rate = yearly_win_rates[year]
        
        if not np.isnan(year_return):
            ax.text(len(pivot_table.columns) + 0.5, i, f"{year_return:.1f}%\n({year_win_rate:.0f}%)", 
                    ha="center", va="center")
    
    # Adjust layout
    plt.tight_layout()
    
    return fig


def load_ohlcv_data(file_path: str, date_column: str = 'timestamp') -> pd.DataFrame:
    """
    Load OHLCV data from file.
    
    Args:
        file_path: Path to the CSV file
        date_column: Name of the date/timestamp column
        
    Returns:
        DataFrame with OHLCV data
    """
    # Load data
    df = pd.read_csv(file_path)
    
    # Check required columns
    required_columns = ['open', 'high', 'low', 'close', 'volume']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        raise ValueError(f"Required columns missing: {missing_columns}")
    
    # Convert date column to datetime
    if date_column in df.columns:
        df[date_column] = pd.to_datetime(df[date_column])
    else:
        raise ValueError(f"Date column '{date_column}' not found in data")
    
    # Sort by date
    df.sort_values(by=date_column, inplace=True)
    
    return df


def resample_ohlcv_data(df: pd.DataFrame, timeframe: str, date_column: str = 'timestamp') -> pd.DataFrame:
    """
    Resample OHLCV data to a different timeframe.
    
    Args:
        df: DataFrame with OHLCV data
        timeframe: Target timeframe (e.g., '1H', '4H', '1D')
        date_column: Name of the date/timestamp column
        
    Returns:
        Resampled DataFrame
    """
    # Check required columns
    required_columns = ['open', 'high', 'low', 'close', 'volume']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        raise ValueError(f"Required columns missing: {missing_columns}")
    
    # Ensure date column is datetime
    if not pd.api.types.is_datetime64_any_dtype(df[date_column]):
        df[date_column] = pd.to_datetime(df[date_column])
    
    # Set date as index
    df_indexed = df.set_index(date_column)
    
    # Define resampling functions
    ohlc_dict = {
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }
    
    # Resample data
    resampled = df_indexed.resample(timeframe).agg(ohlc_dict)
    
    # Reset index
    resampled.reset_index(inplace=True)
    
    return resampled 