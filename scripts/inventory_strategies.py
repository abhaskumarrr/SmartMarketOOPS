import os
import json
import re

def is_strategy_file(filename):
    """
    Heuristics to identify if a file is a standalone strategy backtest.
    - Ends with _system.py, _strategy.py, _backtester.py, _optimizer.py
    - Does not start with __
    - Is a .py file
    """
    if not filename.endswith('.py') or filename.startswith('__'):
        return False
    
    strategy_patterns = [
        '_system.py',
        '_strategy.py',
        '_backtester.py',
        '_optimizer.py',
        '_smc.py'
    ]
    
    return any(p in filename for p in strategy_patterns)

def inventory_strategies(start_path, output_file):
    """
    Scans the given path for strategy files and creates a JSON manifest.
    """
    strategy_manifest = []
    
    for root, _, files in os.walk(start_path):
        for file in files:
            if is_strategy_file(file):
                # Convert file path to module import path
                relative_path = os.path.relpath(os.path.join(root, file), 'ml/src')
                module_name = relative_path.replace(os.path.sep, '.').replace('.py', '')
                
                strategy_manifest.append({
                    "name": file.replace('.py', ''),
                    "file_path": os.path.join(root, file),
                    "module": f"src.{module_name}"
                })

    with open(output_file, 'w') as f:
        json.dump(strategy_manifest, f, indent=2)
        
    print(f"✅ Successfully created strategy manifest with {len(strategy_manifest)} entries.")
    print(f"Manifest file saved to: {output_file}")
    return strategy_manifest

if __name__ == "__main__":
    backtesting_dir = 'ml/src/backtesting'
    manifest_file = 'strategy_manifest.json'
    
    if not os.path.exists(backtesting_dir):
        print(f"Error: Directory not found at {backtesting_dir}")
    else:
        inventory_strategies(backtesting_dir, manifest_file) 