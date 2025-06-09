#!/usr/bin/env python3
"""
Comprehensive system validation script for SmartMarketOOPS
Validates all components, configurations, and dependencies
"""

import os
import sys
import subprocess
import json
import logging
from pathlib import Path
from typing import Dict, List, Any
import importlib.util

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SystemValidator:
    """Comprehensive system validation"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.validation_results = {
            'configuration': {},
            'dependencies': {},
            'data': {},
            'models': {},
            'services': {},
            'overall_status': 'UNKNOWN'
        }
    
    def validate_configuration(self) -> Dict[str, Any]:
        """Validate all configuration files"""
        logger.info("🔍 Validating configuration files...")
        
        results = {}
        
        # Check .env file
        env_file = self.project_root / '.env'
        if env_file.exists():
            results['env_file'] = {
                'exists': True,
                'size': env_file.stat().st_size,
                'status': 'VALID'
            }
            
            # Check critical environment variables
            with open(env_file) as f:
                env_content = f.read()
                
            critical_vars = [
                'NODE_ENV', 'PORT', 'DATABASE_URL', 'DELTA_EXCHANGE_API_KEY',
                'DELTA_EXCHANGE_TESTNET', 'ML_PORT', 'FRONTEND_PORT'
            ]
            
            missing_vars = []
            for var in critical_vars:
                if var not in env_content:
                    missing_vars.append(var)
            
            results['env_file']['missing_variables'] = missing_vars
            if missing_vars:
                results['env_file']['status'] = 'WARNING'
        else:
            results['env_file'] = {'exists': False, 'status': 'CRITICAL'}
        
        # Check package.json files
        package_files = [
            self.project_root / 'package.json',
            self.project_root / 'backend' / 'package.json',
            self.project_root / 'frontend' / 'package.json'
        ]
        
        for package_file in package_files:
            name = package_file.parent.name if package_file.parent != self.project_root else 'root'
            if package_file.exists():
                try:
                    with open(package_file) as f:
                        package_data = json.load(f)
                    
                    results[f'package_{name}'] = {
                        'exists': True,
                        'valid_json': True,
                        'has_scripts': 'scripts' in package_data,
                        'has_dependencies': 'dependencies' in package_data,
                        'status': 'VALID'
                    }
                except json.JSONDecodeError:
                    results[f'package_{name}'] = {
                        'exists': True,
                        'valid_json': False,
                        'status': 'CRITICAL'
                    }
            else:
                results[f'package_{name}'] = {'exists': False, 'status': 'CRITICAL'}
        
        # Check requirements.txt
        requirements_file = self.project_root / 'requirements.txt'
        if requirements_file.exists():
            with open(requirements_file) as f:
                lines = f.readlines()
            
            results['requirements'] = {
                'exists': True,
                'line_count': len(lines),
                'status': 'VALID'
            }
        else:
            results['requirements'] = {'exists': False, 'status': 'CRITICAL'}
        
        return results
    
    def validate_dependencies(self) -> Dict[str, Any]:
        """Validate system dependencies"""
        logger.info("🔍 Validating system dependencies...")
        
        results = {}
        
        # Check Python
        try:
            python_version = sys.version_info
            results['python'] = {
                'available': True,
                'version': f"{python_version.major}.{python_version.minor}.{python_version.micro}",
                'status': 'VALID' if python_version >= (3, 8) else 'WARNING'
            }
        except Exception as e:
            results['python'] = {'available': False, 'error': str(e), 'status': 'CRITICAL'}
        
        # Check Node.js
        try:
            node_result = subprocess.run(['node', '--version'], capture_output=True, text=True)
            if node_result.returncode == 0:
                results['nodejs'] = {
                    'available': True,
                    'version': node_result.stdout.strip(),
                    'status': 'VALID'
                }
            else:
                results['nodejs'] = {'available': False, 'status': 'WARNING'}
        except FileNotFoundError:
            results['nodejs'] = {'available': False, 'status': 'WARNING'}
        
        # Check npm
        try:
            npm_result = subprocess.run(['npm', '--version'], capture_output=True, text=True)
            if npm_result.returncode == 0:
                results['npm'] = {
                    'available': True,
                    'version': npm_result.stdout.strip(),
                    'status': 'VALID'
                }
            else:
                results['npm'] = {'available': False, 'status': 'WARNING'}
        except FileNotFoundError:
            results['npm'] = {'available': False, 'status': 'WARNING'}
        
        # Check Docker
        try:
            docker_result = subprocess.run(['docker', '--version'], capture_output=True, text=True)
            if docker_result.returncode == 0:
                results['docker'] = {
                    'available': True,
                    'version': docker_result.stdout.strip(),
                    'status': 'VALID'
                }
            else:
                results['docker'] = {'available': False, 'status': 'INFO'}
        except FileNotFoundError:
            results['docker'] = {'available': False, 'status': 'INFO'}
        
        # Check Python packages
        critical_packages = ['pandas', 'numpy', 'fastapi', 'ccxt', 'torch']
        for package in critical_packages:
            try:
                spec = importlib.util.find_spec(package)
                if spec is not None:
                    results[f'python_{package}'] = {'available': True, 'status': 'VALID'}
                else:
                    results[f'python_{package}'] = {'available': False, 'status': 'WARNING'}
            except Exception:
                results[f'python_{package}'] = {'available': False, 'status': 'WARNING'}
        
        return results
    
    def validate_data(self) -> Dict[str, Any]:
        """Validate data files and directories"""
        logger.info("🔍 Validating data files...")
        
        results = {}
        
        # Check data directories
        data_dirs = ['data', 'sample_data', 'models']
        for dir_name in data_dirs:
            dir_path = self.project_root / dir_name
            if dir_path.exists():
                file_count = len(list(dir_path.rglob('*')))
                results[f'{dir_name}_dir'] = {
                    'exists': True,
                    'file_count': file_count,
                    'status': 'VALID' if file_count > 0 else 'WARNING'
                }
            else:
                results[f'{dir_name}_dir'] = {'exists': False, 'status': 'WARNING'}
        
        # Check for trained models
        models_dir = self.project_root / 'models'
        if models_dir.exists():
            model_files = list(models_dir.glob('*.pt'))
            results['trained_models'] = {
                'count': len(model_files),
                'files': [f.name for f in model_files],
                'status': 'VALID' if len(model_files) > 0 else 'WARNING'
            }
        
        # Check sample data
        sample_data_dir = self.project_root / 'sample_data'
        if sample_data_dir.exists():
            csv_files = list(sample_data_dir.glob('*.csv'))
            results['sample_data'] = {
                'csv_count': len(csv_files),
                'files': [f.name for f in csv_files],
                'status': 'VALID' if len(csv_files) > 0 else 'WARNING'
            }
        
        return results
    
    def validate_services(self) -> Dict[str, Any]:
        """Validate service configurations"""
        logger.info("🔍 Validating service configurations...")
        
        results = {}
        
        # Check main.py
        main_py = self.project_root / 'main.py'
        results['main_py'] = {
            'exists': main_py.exists(),
            'status': 'VALID' if main_py.exists() else 'CRITICAL'
        }
        
        # Check backend structure
        backend_dir = self.project_root / 'backend'
        if backend_dir.exists():
            backend_files = ['src/server.ts', 'package.json']
            backend_status = all((backend_dir / f).exists() for f in backend_files)
            results['backend'] = {
                'directory_exists': True,
                'required_files': backend_status,
                'status': 'VALID' if backend_status else 'WARNING'
            }
        else:
            results['backend'] = {'directory_exists': False, 'status': 'CRITICAL'}
        
        # Check frontend structure
        frontend_dir = self.project_root / 'frontend'
        if frontend_dir.exists():
            frontend_files = ['package.json', 'src']
            frontend_status = all((frontend_dir / f).exists() for f in frontend_files)
            results['frontend'] = {
                'directory_exists': True,
                'required_files': frontend_status,
                'status': 'VALID' if frontend_status else 'WARNING'
            }
        else:
            results['frontend'] = {'directory_exists': False, 'status': 'CRITICAL'}
        
        return results
    
    def run_validation(self) -> Dict[str, Any]:
        """Run complete system validation"""
        logger.info("🚀 Starting comprehensive system validation...")
        
        # Run all validations
        self.validation_results['configuration'] = self.validate_configuration()
        self.validation_results['dependencies'] = self.validate_dependencies()
        self.validation_results['data'] = self.validate_data()
        self.validation_results['services'] = self.validate_services()
        
        # Determine overall status
        all_statuses = []
        for category in self.validation_results.values():
            if isinstance(category, dict):
                for item in category.values():
                    if isinstance(item, dict) and 'status' in item:
                        all_statuses.append(item['status'])
        
        if 'CRITICAL' in all_statuses:
            self.validation_results['overall_status'] = 'CRITICAL'
        elif 'WARNING' in all_statuses:
            self.validation_results['overall_status'] = 'WARNING'
        else:
            self.validation_results['overall_status'] = 'VALID'
        
        return self.validation_results
    
    def print_results(self):
        """Print validation results in a readable format"""
        print("\n" + "="*80)
        print("🔍 SMARTMARKETOOPS SYSTEM VALIDATION REPORT")
        print("="*80)
        
        status_colors = {
            'VALID': '✅',
            'WARNING': '⚠️',
            'CRITICAL': '❌',
            'INFO': 'ℹ️'
        }
        
        for category, items in self.validation_results.items():
            if category == 'overall_status':
                continue
                
            print(f"\n📋 {category.upper()}:")
            print("-" * 40)
            
            if isinstance(items, dict):
                for item_name, item_data in items.items():
                    if isinstance(item_data, dict) and 'status' in item_data:
                        status_icon = status_colors.get(item_data['status'], '❓')
                        print(f"  {status_icon} {item_name}: {item_data['status']}")
                        
                        # Show additional details for critical/warning items
                        if item_data['status'] in ['CRITICAL', 'WARNING']:
                            for key, value in item_data.items():
                                if key != 'status':
                                    print(f"      {key}: {value}")
        
        # Overall status
        overall_icon = status_colors.get(self.validation_results['overall_status'], '❓')
        print(f"\n🎯 OVERALL STATUS: {overall_icon} {self.validation_results['overall_status']}")
        
        if self.validation_results['overall_status'] == 'VALID':
            print("🎉 System is ready for operation!")
        elif self.validation_results['overall_status'] == 'WARNING':
            print("⚠️ System has some issues but should work. Check warnings above.")
        else:
            print("❌ System has critical issues that need to be fixed before operation.")

if __name__ == "__main__":
    validator = SystemValidator()
    results = validator.run_validation()
    validator.print_results()
    
    # Save results to file
    results_file = Path(__file__).parent.parent / 'validation_results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n📄 Detailed results saved to: {results_file}")
    
    # Exit with appropriate code
    if results['overall_status'] == 'CRITICAL':
        sys.exit(1)
    else:
        sys.exit(0)
