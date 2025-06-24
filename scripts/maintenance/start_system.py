#!/usr/bin/env python3
"""
SmartMarketOOPS System Startup Script
Comprehensive system startup with compatibility checks and fixes
"""

import os
import sys
import subprocess
import logging
import asyncio
import signal
from pathlib import Path
from typing import List, Dict, Any
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('logs/startup.log', mode='a')
    ]
)
logger = logging.getLogger(__name__)

class SystemManager:
    """Manages the complete SmartMarketOOPS system startup"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.services = {}
        self.running = False
        
    def check_dependencies(self) -> bool:
        """Check if all required dependencies are available"""
        logger.info("🔍 Checking system dependencies...")
        
        # Check Python version
        if sys.version_info < (3, 8):
            logger.error("❌ Python 3.8+ required")
            return False
        
        # Check Docker
        try:
            subprocess.run(['docker', '--version'], check=True, capture_output=True)
            logger.info("✅ Docker available")
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.warning("⚠️ Docker not available - will run in local mode")
        
        # Check Node.js
        try:
            subprocess.run(['node', '--version'], check=True, capture_output=True)
            logger.info("✅ Node.js available")
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.warning("⚠️ Node.js not available - frontend may not work")
        
        return True
    
    def setup_environment(self):
        """Setup the environment and create necessary directories"""
        logger.info("🔧 Setting up environment...")
        
        # Create necessary directories
        directories = ['logs', 'data', 'models', 'config', 'temp']
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            logger.info(f"✅ Created directory: {directory}")
        
        # Check .env file
        env_file = self.project_root / '.env'
        if not env_file.exists():
            logger.warning("⚠️ .env file not found - using defaults")
            # Create basic .env file
            with open(env_file, 'w') as f:
                f.write("""# SmartMarketOOPS Environment Configuration
NODE_ENV=development
PORT=8001
FRONTEND_PORT=3000
DATABASE_URL=postgresql://postgres:password@localhost:5432/smartmarket
REDIS_URL=redis://localhost:6379/0
LOG_LEVEL=INFO
""")
            logger.info("✅ Created basic .env file")
    
    def install_python_dependencies(self):
        """Install Python dependencies"""
        logger.info("📦 Installing Python dependencies...")
        
        try:
            # Check if virtual environment exists
            venv_path = self.project_root / 'venv'
            if not venv_path.exists():
                logger.info("Creating Python virtual environment...")
                subprocess.run([sys.executable, '-m', 'venv', 'venv'], check=True)
            
            # Install dependencies
            pip_path = venv_path / 'bin' / 'pip' if os.name != 'nt' else venv_path / 'Scripts' / 'pip.exe'
            if pip_path.exists():
                subprocess.run([str(pip_path), 'install', '-r', 'requirements.txt'], check=True)
                logger.info("✅ Python dependencies installed")
            else:
                logger.warning("⚠️ Virtual environment pip not found")
                
        except subprocess.CalledProcessError as e:
            logger.warning(f"⚠️ Failed to install Python dependencies: {e}")
    
    def install_node_dependencies(self):
        """Install Node.js dependencies"""
        logger.info("📦 Installing Node.js dependencies...")
        
        try:
            # Install root dependencies
            if (self.project_root / 'package.json').exists():
                subprocess.run(['npm', 'install'], cwd=self.project_root, check=True)
                logger.info("✅ Root Node.js dependencies installed")
            
            # Install frontend dependencies
            frontend_path = self.project_root / 'frontend'
            if (frontend_path / 'package.json').exists():
                subprocess.run(['npm', 'install'], cwd=frontend_path, check=True)
                logger.info("✅ Frontend dependencies installed")
            
            # Install backend dependencies
            backend_path = self.project_root / 'backend'
            if (backend_path / 'package.json').exists():
                subprocess.run(['npm', 'install'], cwd=backend_path, check=True)
                logger.info("✅ Backend dependencies installed")
                
        except subprocess.CalledProcessError as e:
            logger.warning(f"⚠️ Failed to install Node.js dependencies: {e}")
    
    def start_infrastructure(self):
        """Start infrastructure services (databases, etc.)"""
        logger.info("🚀 Starting infrastructure services...")
        
        try:
            # Start Docker services
            subprocess.run(['docker-compose', 'up', '-d', 'postgres', 'redis'], 
                         cwd=self.project_root, check=True)
            logger.info("✅ Infrastructure services started")
            
            # Wait for services to be ready
            time.sleep(10)
            
        except subprocess.CalledProcessError as e:
            logger.warning(f"⚠️ Failed to start infrastructure: {e}")
    
    def start_ml_system(self):
        """Start the ML trading system"""
        logger.info("🤖 Starting ML trading system...")

        try:
            # Check if main.py exists
            main_py_path = self.project_root / 'main.py'
            if not main_py_path.exists():
                logger.error("❌ main.py not found in project root")
                return

            # Start the main ML system
            process = subprocess.Popen([
                sys.executable, str(main_py_path)
            ], cwd=self.project_root)

            self.services['ml_system'] = process
            logger.info("✅ ML system started")

        except Exception as e:
            logger.error(f"❌ Failed to start ML system: {e}")
    
    def start_backend(self):
        """Start the backend API server"""
        logger.info("🔧 Starting backend server...")
        
        try:
            backend_path = self.project_root / 'backend'
            process = subprocess.Popen([
                'npm', 'run', 'start:ts'
            ], cwd=backend_path)
            
            self.services['backend'] = process
            logger.info("✅ Backend server started")
            
        except Exception as e:
            logger.error(f"❌ Failed to start backend: {e}")
    
    def start_frontend(self):
        """Start the frontend development server"""
        logger.info("🎨 Starting frontend server...")
        
        try:
            frontend_path = self.project_root / 'frontend'
            process = subprocess.Popen([
                'npm', 'run', 'dev'
            ], cwd=frontend_path)
            
            self.services['frontend'] = process
            logger.info("✅ Frontend server started")
            
        except Exception as e:
            logger.error(f"❌ Failed to start frontend: {e}")
    
    def start_system(self):
        """Start the complete system"""
        logger.info("🚀 Starting SmartMarketOOPS Complete System...")
        
        if not self.check_dependencies():
            logger.error("❌ Dependency check failed")
            return False
        
        self.setup_environment()
        self.install_python_dependencies()
        self.install_node_dependencies()
        self.start_infrastructure()
        
        # Start services in order
        self.start_ml_system()
        time.sleep(5)  # Wait for ML system to initialize
        
        self.start_backend()
        time.sleep(3)  # Wait for backend to start
        
        self.start_frontend()
        
        self.running = True
        logger.info("🎉 SmartMarketOOPS system started successfully!")
        logger.info("📊 Access points:")
        logger.info("   - ML System: http://localhost:3002")
        logger.info("   - Backend API: http://localhost:3001")
        logger.info("   - Frontend: http://localhost:3000")
        logger.info("   - API Docs: http://localhost:3002/docs")
        
        return True
    
    def stop_system(self):
        """Stop all services"""
        logger.info("🛑 Stopping SmartMarketOOPS system...")
        
        # Stop all processes
        for service_name, process in self.services.items():
            try:
                process.terminate()
                process.wait(timeout=10)
                logger.info(f"✅ Stopped {service_name}")
            except subprocess.TimeoutExpired:
                process.kill()
                logger.warning(f"⚠️ Force killed {service_name}")
            except Exception as e:
                logger.error(f"❌ Error stopping {service_name}: {e}")
        
        # Stop Docker services
        try:
            subprocess.run(['docker-compose', 'down'], 
                         cwd=self.project_root, check=True)
            logger.info("✅ Infrastructure services stopped")
        except subprocess.CalledProcessError as e:
            logger.warning(f"⚠️ Failed to stop infrastructure: {e}")
        
        self.running = False
        logger.info("✅ System stopped")
    
    def signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"Received signal {signum}, shutting down...")
        self.stop_system()
        sys.exit(0)

def main():
    """Main entry point"""
    manager = SystemManager()
    
    # Register signal handlers
    signal.signal(signal.SIGINT, manager.signal_handler)
    signal.signal(signal.SIGTERM, manager.signal_handler)
    
    try:
        if manager.start_system():
            # Keep the script running
            while manager.running:
                time.sleep(1)
        else:
            logger.error("❌ Failed to start system")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt")
        manager.stop_system()
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        manager.stop_system()
        sys.exit(1)

if __name__ == "__main__":
    main()
