#!/usr/bin/env python3
"""
Create a visual comparison chart between Spray-and-Pray vs Institutional Grade trading
"""

import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

def create_comparison_chart():
    # Data for comparison
    metrics = ['Signals/Day', 'Win Rate (%)', 'Risk/Trade (%)', 'Confluence (%)', 'Monthly Return (%)']
    spray_and_pray = [46, 26, 50, 35, 941]  # Unsustainable
    institutional = [2, 100, 1.5, 85, 4.06]  # Sustainable
    
    # Create figure with multiple subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('🏛️ INSTITUTIONAL vs SPRAY-AND-PRAY TRADING COMPARISON', fontsize=16, fontweight='bold')
    
    # Chart 1: Signal Volume Comparison
    methods = ['Spray-and-Pray', 'Institutional']
    signals = [46, 2]
    colors = ['red', 'green']
    
    bars1 = ax1.bar(methods, signals, color=colors, alpha=0.7)
    ax1.set_title('Daily Signal Generation', fontweight='bold')
    ax1.set_ylabel('Signals per Day')
    ax1.set_ylim(0, 50)
    
    # Add value labels on bars
    for bar, value in zip(bars1, signals):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{value}', ha='center', va='bottom', fontweight='bold')
    
    # Chart 2: Win Rate Comparison
    win_rates = [26, 100]
    bars2 = ax2.bar(methods, win_rates, color=colors, alpha=0.7)
    ax2.set_title('Win Rate Comparison', fontweight='bold')
    ax2.set_ylabel('Win Rate (%)')
    ax2.set_ylim(0, 110)
    
    for bar, value in zip(bars2, win_rates):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f'{value}%', ha='center', va='bottom', fontweight='bold')
    
    # Chart 3: Risk Management
    risk_per_trade = [50, 1.5]
    bars3 = ax3.bar(methods, risk_per_trade, color=colors, alpha=0.7)
    ax3.set_title('Risk Per Trade', fontweight='bold')
    ax3.set_ylabel('Risk per Trade (%)')
    ax3.set_ylim(0, 55)
    
    for bar, value in zip(bars3, risk_per_trade):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{value}%', ha='center', va='bottom', fontweight='bold')
    
    # Chart 4: Quality vs Quantity Philosophy
    # Create a scatter plot showing the relationship
    x_data = [46, 2]  # Signals per day
    y_data = [26, 100]  # Win rate
    size_data = [50*20, 1.5*20]  # Risk per trade (scaled for visibility)
    
    scatter = ax4.scatter(x_data, y_data, s=size_data, c=colors, alpha=0.7)
    ax4.set_title('Quality vs Quantity Analysis', fontweight='bold')
    ax4.set_xlabel('Signals per Day')
    ax4.set_ylabel('Win Rate (%)')
    ax4.set_xlim(0, 50)
    ax4.set_ylim(0, 110)
    
    # Add labels for each point
    ax4.annotate('Spray-and-Pray\n(High Volume, Low Quality)', 
                xy=(46, 26), xytext=(35, 40),
                arrowprops=dict(arrowstyle='->', color='red'),
                fontsize=10, ha='center')
    
    ax4.annotate('Institutional\n(Low Volume, High Quality)', 
                xy=(2, 100), xytext=(15, 85),
                arrowprops=dict(arrowstyle='->', color='green'),
                fontsize=10, ha='center')
    
    plt.tight_layout()
    
    # Save the chart
    chart_filename = f"trading_philosophy_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    plt.savefig(chart_filename, dpi=300, bbox_inches='tight')
    print(f"📊 Comparison chart saved as: {chart_filename}")
    
    plt.show()
    
    # Create summary statistics table
    print("\n🏛️ INSTITUTIONAL vs SPRAY-AND-PRAY SUMMARY")
    print("=" * 80)
    print(f"{'Metric':<20} {'Spray-and-Pray':<15} {'Institutional':<15} {'Winner':<15}")
    print("-" * 80)
    
    comparisons = [
        ("Signals/Day", "46", "2", "Institutional*"),
        ("Win Rate", "26%", "100%", "Institutional"),
        ("Risk/Trade", "50%", "1.5%", "Institutional"),
        ("Confluence Req", "35%", "85%", "Institutional"),
        ("Monthly Return", "941%**", "4.06%", "Institutional*"),
        ("Sustainability", "No", "Yes", "Institutional"),
        ("Max Drawdown", "High", "0%", "Institutional"),
        ("Sharpe Ratio", "Negative", "3.66", "Institutional"),
    ]
    
    for metric, spray, inst, winner in comparisons:
        print(f"{metric:<20} {spray:<15} {inst:<15} {winner:<15}")
    
    print("\n* Quality over quantity approach")
    print("** Unsustainable due to extreme leverage and risk")
    print("\n🎯 CONCLUSION: Institutional approach wins on ALL meaningful metrics")


if __name__ == "__main__":
    create_comparison_chart()