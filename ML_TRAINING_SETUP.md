# ML Training Setup Guide for M2 MacBook Air 8GB

## Overview

This guide provides technical implementation details for running SmartMarketOOPS ML training on M2 MacBook Air 8GB systems, focusing on memory-efficient configurations and optimal performance for achieving the 85.3% win rate target.

## Table of Contents

1. [Memory-Efficient Training Configurations](#memory-efficient-training-configurations)
2. [Local vs Cloud Task Distribution](#local-vs-cloud-task-distribution)
3. [Training Workflow Scripts](#training-workflow-scripts)
4. [Performance Optimization Settings](#performance-optimization-settings)

---

## Memory-Efficient Training Configurations

### Python ML Configuration for 8GB Systems

```python
# ml/src/config/local_8gb_config.py
import torch
import psutil
import os
from dataclasses import dataclass
from typing import Optional, Dict, Any

@dataclass
class Local8GBConfig:
    """Optimized configuration for M2 MacBook Air 8GB"""

    # System constraints
    max_memory_usage: int = 6 * 1024 * 1024 * 1024  # 6GB max (leave 2GB for system)
    swap_threshold: int = 4 * 1024 * 1024 * 1024     # 4GB swap limit

    # Training parameters optimized for memory
    batch_size: int = 32
    max_sequence_length: int = 60
    num_workers: int = 2
    prefetch_factor: int = 2
    pin_memory: bool = False  # Disable on unified memory systems

    # Model architecture (reduced complexity)
    lstm_hidden_size: int = 128
    lstm_num_layers: int = 2
    dropout_rate: float = 0.3

    # Training optimization
    gradient_accumulation_steps: int = 4
    mixed_precision: bool = True
    checkpoint_frequency: int = 100
    early_stopping_patience: int = 10

    # External storage paths
    data_path: str = "/Volumes/ML_Storage/ML_Data"
    model_path: str = "/Volumes/ML_Storage/ML_Models"
    cache_path: str = "/Volumes/ML_Storage/Cache_Storage"

    # Memory monitoring
    enable_memory_monitoring: bool = True
    memory_check_frequency: int = 50  # Check every 50 batches

    def __post_init__(self):
        """Validate configuration and adjust based on system"""
        # Detect available memory
        available_memory = psutil.virtual_memory().available

        if available_memory < self.max_memory_usage:
            self.max_memory_usage = int(available_memory * 0.8)
            print(f"⚠️ Adjusted max memory usage to {self.max_memory_usage / (1024**3):.1f}GB")

        # Ensure external storage is available
        if not os.path.exists(self.data_path):
            raise ValueError(f"External storage not found: {self.data_path}")

        # Configure PyTorch for MPS
        if torch.backends.mps.is_available():
            os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'
            print("✅ MPS (Metal Performance Shaders) configured")

class MemoryEfficientDataLoader:
    """Custom DataLoader optimized for 8GB systems"""

    def __init__(self, config: Local8GBConfig):
        self.config = config
        self.cache_dir = config.cache_path
        os.makedirs(self.cache_dir, exist_ok=True)

    def create_dataloader(self, dataset, shuffle=True):
        """Create memory-efficient DataLoader"""
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=shuffle,
            num_workers=self.config.num_workers,
            prefetch_factor=self.config.prefetch_factor,
            pin_memory=self.config.pin_memory,
            persistent_workers=True if self.config.num_workers > 0 else False,
            drop_last=True  # Ensure consistent batch sizes
        )

    def preprocess_and_cache(self, raw_data_path: str, symbol: str):
        """Preprocess data and cache to external storage"""
        cache_file = os.path.join(self.cache_dir, f"{symbol}_processed.pkl")

        if os.path.exists(cache_file):
            print(f"📁 Loading cached data for {symbol}")
            return torch.load(cache_file)

        print(f"🔄 Preprocessing data for {symbol}...")
        # Implement preprocessing logic here
        # processed_data = preprocess_raw_data(raw_data_path)

        # Cache processed data
        # torch.save(processed_data, cache_file)
        # return processed_data
        pass

class MemoryEfficientModel(torch.nn.Module):
    """LSTM model optimized for 8GB memory constraints"""

    def __init__(self, config: Local8GBConfig, input_size: int, num_classes: int):
        super().__init__()
        self.config = config

        # Reduced LSTM layers for memory efficiency
        self.lstm = torch.nn.LSTM(
            input_size=input_size,
            hidden_size=config.lstm_hidden_size,
            num_layers=config.lstm_num_layers,
            dropout=config.dropout_rate,
            batch_first=True
        )

        # Classification head
        self.classifier = torch.nn.Sequential(
            torch.nn.Dropout(config.dropout_rate),
            torch.nn.Linear(config.lstm_hidden_size, config.lstm_hidden_size // 2),
            torch.nn.ReLU(),
            torch.nn.Dropout(config.dropout_rate),
            torch.nn.Linear(config.lstm_hidden_size // 2, num_classes)
        )

    def forward(self, x):
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)

        # Use last output for classification
        last_output = lstm_out[:, -1, :]

        # Classification
        output = self.classifier(last_output)

        return output

class MemoryMonitor:
    """Real-time memory monitoring during training"""

    def __init__(self, config: Local8GBConfig):
        self.config = config
        self.memory_history = []

    def check_memory_status(self) -> Dict[str, Any]:
        """Check current memory status"""
        memory = psutil.virtual_memory()
        swap = psutil.swap_memory()

        status = {
            'memory_percent': memory.percent,
            'memory_available_gb': memory.available / (1024**3),
            'swap_used_gb': swap.used / (1024**3),
            'swap_percent': swap.percent,
            'memory_pressure': memory.percent > 85 or swap.used > self.config.swap_threshold
        }

        self.memory_history.append(status)
        return status

    def should_trigger_cleanup(self) -> bool:
        """Determine if memory cleanup is needed"""
        status = self.check_memory_status()
        return status['memory_pressure']

    def get_memory_summary(self) -> str:
        """Get formatted memory summary"""
        if not self.memory_history:
            return "No memory data available"

        latest = self.memory_history[-1]
        return (f"Memory: {latest['memory_percent']:.1f}% used, "
                f"{latest['memory_available_gb']:.1f}GB available, "
                f"Swap: {latest['swap_used_gb']:.1f}GB used")
```

### Training Script with Memory Management

```python
# ml/src/training/memory_efficient_trainer.py
import torch
import torch.nn.functional as F
import gc
import time
from typing import Tuple, Optional
from .local_8gb_config import Local8GBConfig, MemoryEfficientModel, MemoryMonitor

class MemoryEfficientTrainer:
    """Trainer optimized for M2 MacBook Air 8GB"""

    def __init__(self, config: Local8GBConfig):
        self.config = config
        self.device = self._setup_device()
        self.memory_monitor = MemoryMonitor(config)
        self.scaler = torch.cuda.amp.GradScaler() if config.mixed_precision else None

    def _setup_device(self) -> torch.device:
        """Setup optimal device for M2 MacBook Air"""
        if torch.backends.mps.is_available():
            print("✅ Using MPS (Metal Performance Shaders)")
            return torch.device("mps")
        else:
            print("⚠️ MPS not available, using CPU")
            return torch.device("cpu")

    def train_model(self, model: MemoryEfficientModel, train_loader, val_loader,
                   epochs: int = 50) -> MemoryEfficientModel:
        """Train model with aggressive memory management"""

        print(f"🚀 Starting training on {self.device}")
        print(f"📊 Initial memory: {self.memory_monitor.get_memory_summary()}")

        # Move model to device
        model = model.to(self.device)

        # Setup optimizer with memory-efficient settings
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=0.001,
            weight_decay=0.01,
            eps=1e-8
        )

        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=0.01,
            epochs=epochs,
            steps_per_epoch=len(train_loader)
        )

        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(epochs):
            # Pre-epoch memory check
            if self.memory_monitor.should_trigger_cleanup():
                print("⚠️ Memory pressure detected, running cleanup...")
                self._aggressive_memory_cleanup()

            # Training phase
            train_loss = self._train_epoch(model, train_loader, optimizer, scheduler, epoch)

            # Validation phase
            val_loss = self._validate_epoch(model, val_loader, epoch)

            # Post-epoch cleanup
            self._cleanup_memory()

            # Early stopping logic
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                self._save_checkpoint(model, optimizer, epoch, val_loss)
            else:
                patience_counter += 1

            if patience_counter >= self.config.early_stopping_patience:
                print(f"🛑 Early stopping at epoch {epoch}")
                break

            # Memory status report
            memory_summary = self.memory_monitor.get_memory_summary()
            print(f"📊 Epoch {epoch}: Train Loss: {train_loss:.4f}, "
                  f"Val Loss: {val_loss:.4f}, {memory_summary}")

        print("✅ Training completed successfully")
        return model

    def _train_epoch(self, model, train_loader, optimizer, scheduler, epoch) -> float:
        """Memory-efficient training epoch"""
        model.train()
        total_loss = 0
        num_batches = len(train_loader)

        for batch_idx, (data, targets) in enumerate(train_loader):
            # Move data to device
            data, targets = data.to(self.device), targets.to(self.device)

            # Forward pass with optional mixed precision
            if self.config.mixed_precision and self.device.type == 'cuda':
                with torch.autocast(device_type='cuda'):
                    outputs = model(data)
                    loss = F.cross_entropy(outputs, targets)
                    loss = loss / self.config.gradient_accumulation_steps
            else:
                outputs = model(data)
                loss = F.cross_entropy(outputs, targets)
                loss = loss / self.config.gradient_accumulation_steps

            # Backward pass
            if self.config.mixed_precision and self.device.type == 'cuda':
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            # Gradient accumulation
            if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                if self.config.mixed_precision and self.device.type == 'cuda':
                    self.scaler.step(optimizer)
                    self.scaler.update()
                else:
                    optimizer.step()

                optimizer.zero_grad()
                scheduler.step()

            total_loss += loss.item() * self.config.gradient_accumulation_steps

            # Periodic memory management
            if batch_idx % self.config.memory_check_frequency == 0:
                self._cleanup_memory()

                if self.memory_monitor.should_trigger_cleanup():
                    print(f"⚠️ Memory pressure at batch {batch_idx}")
                    self._aggressive_memory_cleanup()

            # Clear batch data immediately
            del data, targets, outputs

        return total_loss / num_batches

    def _validate_epoch(self, model, val_loader, epoch) -> float:
        """Memory-efficient validation epoch"""
        model.eval()
        total_loss = 0
        num_batches = len(val_loader)

        with torch.no_grad():
            for batch_idx, (data, targets) in enumerate(val_loader):
                data, targets = data.to(self.device), targets.to(self.device)

                outputs = model(data)
                loss = F.cross_entropy(outputs, targets)
                total_loss += loss.item()

                # Clear batch data
                del data, targets, outputs

                # Periodic cleanup during validation
                if batch_idx % 20 == 0:
                    self._cleanup_memory()

        return total_loss / num_batches

    def _cleanup_memory(self):
        """Regular memory cleanup"""
        gc.collect()
        if self.device.type == 'mps':
            torch.mps.empty_cache()
        elif self.device.type == 'cuda':
            torch.cuda.empty_cache()

    def _aggressive_memory_cleanup(self):
        """Aggressive memory cleanup during pressure"""
        # Multiple garbage collection passes
        for _ in range(3):
            gc.collect()

        # Clear device cache
        if self.device.type == 'mps':
            torch.mps.empty_cache()
        elif self.device.type == 'cuda':
            torch.cuda.empty_cache()

        # Brief pause for system recovery
        time.sleep(0.1)

    def _save_checkpoint(self, model, optimizer, epoch, val_loss):
        """Save model checkpoint to external storage"""
        checkpoint_path = os.path.join(
            self.config.model_path,
            f"checkpoint_epoch_{epoch}_loss_{val_loss:.4f}.pt"
        )

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
            'config': self.config
        }, checkpoint_path)

        print(f"💾 Checkpoint saved: {checkpoint_path}")
```

---

## Local vs Cloud Task Distribution

### Task Classification Matrix

| Task Category | Local (M2 MacBook Air) | Cloud Platform | Reasoning |
|---------------|------------------------|----------------|-----------|
| **ML Model Training** | ✅ Short sessions (<2h) | ❌ Latency sensitive | Direct hardware access, faster iteration |
| **ML Inference** | ✅ Real-time predictions | ❌ Latency critical | Sub-second response needed |
| **Data Preprocessing** | ✅ Small-medium datasets | ⚠️ Large datasets only | Memory constraints for large data |
| **Model Evaluation** | ✅ Quick validation | ❌ Keep with training | Immediate feedback needed |
| **Database Operations** | ❌ Memory intensive | ✅ PostgreSQL + Redis | Free up local RAM |
| **Docker Builds** | ❌ Storage + memory heavy | ✅ CI/CD pipelines | Avoid local resource usage |
| **Integration Testing** | ❌ Multi-service testing | ✅ Full stack testing | Requires multiple services |
| **Frontend Development** | ✅ Fast local iteration | ❌ Keep local | Hot reload performance |
| **Backend API Development** | ✅ Core logic development | ⚠️ Hybrid approach | Balance local/cloud |
| **Performance Testing** | ❌ Resource intensive | ✅ Load testing | Requires significant resources |

### Cloud Platform Selection Guide

```bash
#!/bin/bash
# cloud_platform_selector.sh

select_optimal_platform() {
    local task_type="$1"

    case "$task_type" in
        "docker_development")
            echo "🐳 Recommended: GitHub Codespaces"
            echo "   - 4-core, 8GB RAM, 32GB storage"
            echo "   - Docker-in-Docker support"
            echo "   - 60 hours/month free"
            ;;
        "database_development")
            echo "🗄️ Recommended: Gitpod"
            echo "   - PostgreSQL + Redis support"
            echo "   - 50 hours/month free"
            echo "   - Persistent workspaces"
            ;;
        "integration_testing")
            echo "🧪 Recommended: GitHub Actions"
            echo "   - 2000 minutes/month free"
            echo "   - Matrix testing support"
            echo "   - Artifact storage"
            ;;
        "performance_testing")
            echo "⚡ Recommended: Railway + GitHub Actions"
            echo "   - Railway: Production-like environment"
            echo "   - GitHub Actions: Automated testing"
            ;;
        *)
            echo "❓ Unknown task type: $task_type"
            echo "Available options: docker_development, database_development, integration_testing, performance_testing"
            ;;
    esac
}

# Usage examples
select_optimal_platform "docker_development"
select_optimal_platform "database_development"
```

### Hybrid Development Configuration

```yaml
# .github/workflows/hybrid_development.yml
name: Hybrid Development Support

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  cloud_tasks:
    runs-on: ubuntu-latest
    services:
      postgres:
        image: postgres:14
        env:
          POSTGRES_PASSWORD: postgres
          POSTGRES_DB: smartmarketoops_test
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5

      redis:
        image: redis:7
        options: >-
          --health-cmd "redis-cli ping"
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5

    steps:
    - uses: actions/checkout@v4

    - name: Setup Node.js
      uses: actions/setup-node@v4
      with:
        node-version: '20'
        cache: 'npm'

    - name: Setup Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
        cache: 'pip'

    - name: Install dependencies
      run: |
        npm install
        pip install -r ml/requirements.txt

    - name: Run integration tests
      run: |
        npm run test:integration
        npm run test:e2e
      env:
        DATABASE_URL: postgresql://postgres:postgres@localhost:5432/smartmarketoops_test
        REDIS_URL: redis://localhost:6379

    - name: Build production
      run: |
        npm run build:all

    - name: Performance testing
      run: |
        npm run test:performance
```

---

## Training Workflow Scripts

### Complete Training Workflow

```bash
#!/bin/bash
# ml_training_workflow.sh

# Configuration
EXTERNAL_SSD="/Volumes/ML_Storage"
TRAINING_LOG="$EXTERNAL_SSD/Cache_Storage/training_session.log"
PYTHON_ENV="smartmarketoops"

log_message() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a "$TRAINING_LOG"
}

check_prerequisites() {
    log_message "🔍 Checking prerequisites..."

    # Check external SSD
    if [ ! -d "$EXTERNAL_SSD" ]; then
        log_message "❌ External SSD not mounted at $EXTERNAL_SSD"
        exit 1
    fi

    # Check Python environment
    if ! conda env list | grep -q "$PYTHON_ENV"; then
        log_message "❌ Python environment '$PYTHON_ENV' not found"
        exit 1
    fi

    # Check available memory
    available_memory=$(vm_stat | grep "Pages free" | awk '{print $3}' | tr -d '.')
    if [ "$available_memory" -lt 100000 ]; then
        log_message "⚠️ Low memory available: $available_memory pages"
    fi

    log_message "✅ Prerequisites check passed"
}

prepare_training_environment() {
    log_message "🔧 Preparing training environment..."

    # Activate Python environment
    source ~/miniconda3/etc/profile.d/conda.sh
    conda activate "$PYTHON_ENV"

    # Set environment variables
    export ML_DATA_PATH="$EXTERNAL_SSD/ML_Data"
    export ML_MODEL_PATH="$EXTERNAL_SSD/ML_Models"
    export ML_CACHE_PATH="$EXTERNAL_SSD/Cache_Storage"
    export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
    export OMP_NUM_THREADS=2
    export MKL_NUM_THREADS=2

    # Prepare system
    ./scripts/prepare_ml_training.sh prepare

    log_message "✅ Training environment prepared"
}

run_training_session() {
    local symbol="$1"
    local model_type="$2"
    local epochs="${3:-50}"

    log_message "🚀 Starting training session: $symbol, $model_type, $epochs epochs"

    # Start monitoring
    ./scripts/monitor_training.sh start

    # Run training with memory-efficient configuration
    python ml/src/train.py \
        --symbol "$symbol" \
        --model-type "$model_type" \
        --config local_8gb \
        --epochs "$epochs" \
        --data-path "$ML_DATA_PATH" \
        --model-path "$ML_MODEL_PATH" \
        --cache-path "$ML_CACHE_PATH" \
        --enable-monitoring \
        2>&1 | tee -a "$TRAINING_LOG"

    local training_exit_code=$?

    # Stop monitoring
    ./scripts/monitor_training.sh stop

    if [ $training_exit_code -eq 0 ]; then
        log_message "✅ Training completed successfully"
    else
        log_message "❌ Training failed with exit code: $training_exit_code"
    fi

    return $training_exit_code
}

cleanup_training_environment() {
    log_message "🧹 Cleaning up training environment..."

    # Run cleanup script
    ./scripts/prepare_ml_training.sh cleanup

    # Deactivate Python environment
    conda deactivate

    # Generate training report
    python scripts/training_dashboard.py

    log_message "✅ Cleanup completed"
}

# Main workflow
main() {
    local command="$1"
    local symbol="$2"
    local model_type="$3"
    local epochs="$4"

    case "$command" in
        "train")
            if [ -z "$symbol" ] || [ -z "$model_type" ]; then
                echo "Usage: $0 train <symbol> <model_type> [epochs]"
                echo "Example: $0 train BTCUSD lstm 50"
                exit 1
            fi

            check_prerequisites
            prepare_training_environment
            run_training_session "$symbol" "$model_type" "$epochs"
            cleanup_training_environment
            ;;
        "quick_train")
            # Quick training session (minimal setup)
            log_message "⚡ Quick training session"
            source ~/miniconda3/etc/profile.d/conda.sh
            conda activate "$PYTHON_ENV"
            python ml/src/train.py --symbol "$symbol" --model-type "$model_type" --config local_8gb_quick
            conda deactivate
            ;;
        "validate")
            check_prerequisites
            log_message "✅ System ready for training"
            ;;
        *)
            echo "Usage: $0 {train|quick_train|validate} [args...]"
            echo "Commands:"
            echo "  train <symbol> <model_type> [epochs] - Full training session"
            echo "  quick_train <symbol> <model_type>    - Quick training (minimal setup)"
            echo "  validate                             - Check system readiness"
            exit 1
            ;;
    esac
}

main "$@"
```

### Local Development Server Script

```bash
#!/bin/bash
# local_dev_server.sh

# Configuration
FRONTEND_PORT=3000
BACKEND_PORT=3001
ML_SERVICE_PORT=3002

start_local_development() {
    echo "🚀 Starting local development environment..."

    # Create PID file directory
    mkdir -p .local_dev

    # Start frontend (Next.js)
    echo "📱 Starting frontend development server..."
    cd frontend
    NODE_OPTIONS="--max-old-space-size=1024" npm run dev &
    echo $! > ../.local_dev/frontend.pid
    cd ..

    # Start backend (Express.js) with memory limits
    echo "🔧 Starting backend API server..."
    cd backend
    NODE_OPTIONS="--max-old-space-size=1024" npm run dev:local &
    echo $! > ../.local_dev/backend.pid
    cd ..

    # Start ML service (Python FastAPI) - only if needed
    if [ "$1" = "--with-ml" ]; then
        echo "🤖 Starting ML service..."
        cd ml
        source ~/miniconda3/etc/profile.d/conda.sh
        conda activate smartmarketoops
        python -m src.api.app &
        echo $! > ../.local_dev/ml_service.pid
        cd ..
    fi

    # Wait for services to start
    sleep 5

    echo "✅ Local development environment ready"
    echo "🌐 Frontend: http://localhost:$FRONTEND_PORT"
    echo "🔧 Backend: http://localhost:$BACKEND_PORT"
    if [ "$1" = "--with-ml" ]; then
        echo "🤖 ML Service: http://localhost:$ML_SERVICE_PORT"
    fi

    # Monitor memory usage
    echo "📊 Memory usage:"
    ps -o pid,rss,comm -p $(cat .local_dev/*.pid 2>/dev/null) 2>/dev/null
}

stop_local_development() {
    echo "🛑 Stopping local development environment..."

    if [ -d .local_dev ]; then
        for pidfile in .local_dev/*.pid; do
            if [ -f "$pidfile" ]; then
                pid=$(cat "$pidfile")
                kill "$pid" 2>/dev/null && echo "Stopped process $pid"
                rm "$pidfile"
            fi
        done
        rmdir .local_dev 2>/dev/null
    fi

    echo "✅ Local development environment stopped"
}

restart_service() {
    local service="$1"

    case "$service" in
        "frontend")
            if [ -f .local_dev/frontend.pid ]; then
                kill $(cat .local_dev/frontend.pid) 2>/dev/null
                cd frontend
                NODE_OPTIONS="--max-old-space-size=1024" npm run dev &
                echo $! > ../.local_dev/frontend.pid
                cd ..
                echo "🔄 Frontend restarted"
            fi
            ;;
        "backend")
            if [ -f .local_dev/backend.pid ]; then
                kill $(cat .local_dev/backend.pid) 2>/dev/null
                cd backend
                NODE_OPTIONS="--max-old-space-size=1024" npm run dev:local &
                echo $! > ../.local_dev/backend.pid
                cd ..
                echo "🔄 Backend restarted"
            fi
            ;;
        "ml")
            if [ -f .local_dev/ml_service.pid ]; then
                kill $(cat .local_dev/ml_service.pid) 2>/dev/null
                cd ml
                source ~/miniconda3/etc/profile.d/conda.sh
                conda activate smartmarketoops
                python -m src.api.app &
                echo $! > ../.local_dev/ml_service.pid
                cd ..
                echo "🔄 ML service restarted"
            fi
            ;;
        *)
            echo "Usage: $0 restart {frontend|backend|ml}"
            exit 1
            ;;
    esac
}

show_status() {
    echo "📊 Development environment status:"

    if [ -d .local_dev ]; then
        for pidfile in .local_dev/*.pid; do
            if [ -f "$pidfile" ]; then
                service=$(basename "$pidfile" .pid)
                pid=$(cat "$pidfile")
                if ps -p "$pid" > /dev/null 2>&1; then
                    memory=$(ps -o rss= -p "$pid" 2>/dev/null | awk '{print $1/1024}')
                    echo "✅ $service (PID: $pid, Memory: ${memory}MB)"
                else
                    echo "❌ $service (PID: $pid, Not running)"
                fi
            fi
        done
    else
        echo "❌ No development environment running"
    fi
}

# Main command handler
case "$1" in
    "start")
        start_local_development "$2"
        ;;
    "stop")
        stop_local_development
        ;;
    "restart")
        restart_service "$2"
        ;;
    "status")
        show_status
        ;;
    *)
        echo "Usage: $0 {start|stop|restart|status} [options]"
        echo "Options:"
        echo "  start [--with-ml]  - Start development environment"
        echo "  stop               - Stop all services"
        echo "  restart <service>  - Restart specific service"
        echo "  status             - Show service status"
        exit 1
        ;;
esac
```

---

## Performance Optimization Settings

### PyTorch Optimization for M2 MacBook Air

```python
# ml/src/config/pytorch_optimization.py
import torch
import os

def configure_pytorch_for_m2_8gb():
    """Configure PyTorch for optimal performance on M2 MacBook Air 8GB"""

    # MPS (Metal Performance Shaders) configuration
    if torch.backends.mps.is_available():
        print("✅ Configuring MPS for M2 MacBook Air")

        # Prevent memory fragmentation
        os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'

        # Enable MPS fallback for unsupported operations
        os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

        # Set memory allocation strategy
        torch.mps.set_per_process_memory_fraction(0.7)  # Use 70% of available memory

    # CPU optimization for fallback operations
    torch.set_num_threads(2)  # Limit CPU threads for memory efficiency
    torch.set_num_interop_threads(2)

    # Disable CUDA if accidentally enabled
    os.environ['CUDA_VISIBLE_DEVICES'] = ''

    # Memory optimization
    torch.backends.cudnn.benchmark = False  # Reduce memory fragmentation
    torch.backends.cudnn.deterministic = True  # Ensure reproducibility

    print("✅ PyTorch optimized for M2 MacBook Air 8GB")

def get_optimal_device():
    """Get the optimal device for training"""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

def optimize_model_for_memory(model):
    """Apply memory optimizations to model"""
    # Enable gradient checkpointing for large models
    if hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()

    # Use memory-efficient attention if available
    if hasattr(model, 'enable_memory_efficient_attention'):
        model.enable_memory_efficient_attention()

    return model
```

### Training Configuration Templates

```python
# ml/src/config/training_configs.py
from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class QuickTrainingConfig:
    """Quick training for rapid iteration (minimal memory usage)"""
    batch_size: int = 16
    max_epochs: int = 20
    lstm_hidden_size: int = 64
    lstm_num_layers: int = 1
    learning_rate: float = 0.01
    gradient_accumulation_steps: int = 2
    early_stopping_patience: int = 5

@dataclass
class StandardTrainingConfig:
    """Standard training for balanced performance/memory"""
    batch_size: int = 32
    max_epochs: int = 50
    lstm_hidden_size: int = 128
    lstm_num_layers: int = 2
    learning_rate: float = 0.001
    gradient_accumulation_steps: int = 4
    early_stopping_patience: int = 10

@dataclass
class IntensiveTrainingConfig:
    """Intensive training for maximum performance (high memory usage)"""
    batch_size: int = 64
    max_epochs: int = 100
    lstm_hidden_size: int = 256
    lstm_num_layers: int = 3
    learning_rate: float = 0.0005
    gradient_accumulation_steps: int = 8
    early_stopping_patience: int = 15

def get_config_for_available_memory(available_memory_gb: float) -> Dict[str, Any]:
    """Select optimal configuration based on available memory"""
    if available_memory_gb < 3.0:
        print("⚠️ Low memory detected, using quick training config")
        return QuickTrainingConfig().__dict__
    elif available_memory_gb < 5.0:
        print("✅ Standard memory available, using standard config")
        return StandardTrainingConfig().__dict__
    else:
        print("🚀 High memory available, using intensive config")
        return IntensiveTrainingConfig().__dict__
```

### Memory-Efficient Data Processing

```python
# ml/src/data/memory_efficient_processor.py
import pandas as pd
import numpy as np
import gc
from typing import Iterator, Tuple
import psutil

class MemoryEfficientDataProcessor:
    """Process trading data with minimal memory footprint"""

    def __init__(self, chunk_size: int = 1000, cache_path: str = None):
        self.chunk_size = chunk_size
        self.cache_path = cache_path

    def process_data_in_chunks(self, data_path: str) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Process data in memory-efficient chunks"""

        # Read data in chunks to avoid loading entire dataset
        chunk_iter = pd.read_csv(data_path, chunksize=self.chunk_size)

        for chunk_idx, chunk in enumerate(chunk_iter):
            # Process chunk
            processed_chunk = self._process_chunk(chunk)

            # Create features and targets
            X, y = self._create_features_targets(processed_chunk)

            # Memory cleanup
            del chunk, processed_chunk
            gc.collect()

            # Monitor memory usage
            memory_percent = psutil.virtual_memory().percent
            if memory_percent > 85:
                print(f"⚠️ High memory usage ({memory_percent}%) at chunk {chunk_idx}")
                gc.collect()

            yield X, y

    def _process_chunk(self, chunk: pd.DataFrame) -> pd.DataFrame:
        """Process individual data chunk"""
        # Technical indicators calculation
        chunk['sma_20'] = chunk['close'].rolling(window=20).mean()
        chunk['ema_12'] = chunk['close'].ewm(span=12).mean()
        chunk['rsi'] = self._calculate_rsi(chunk['close'])

        # Drop NaN values
        chunk = chunk.dropna()

        return chunk

    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate RSI with memory efficiency"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def _create_features_targets(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Create features and targets from processed data"""
        # Feature columns
        feature_cols = ['open', 'high', 'low', 'close', 'volume', 'sma_20', 'ema_12', 'rsi']

        # Create sequences
        sequence_length = 60
        X, y = [], []

        for i in range(sequence_length, len(data)):
            X.append(data[feature_cols].iloc[i-sequence_length:i].values)
            # Binary classification: 1 if price goes up, 0 if down
            y.append(1 if data['close'].iloc[i] > data['close'].iloc[i-1] else 0)

        return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)

class MemoryMonitoredDataLoader:
    """DataLoader with built-in memory monitoring"""

    def __init__(self, dataset, batch_size: int = 32, memory_threshold: float = 85.0):
        self.dataset = dataset
        self.batch_size = batch_size
        self.memory_threshold = memory_threshold

    def __iter__(self):
        indices = list(range(len(self.dataset)))

        for i in range(0, len(indices), self.batch_size):
            # Check memory before loading batch
            memory_percent = psutil.virtual_memory().percent

            if memory_percent > self.memory_threshold:
                print(f"⚠️ Memory threshold exceeded ({memory_percent}%), running cleanup")
                gc.collect()

                # Wait for memory to stabilize
                import time
                time.sleep(0.1)

            # Load batch
            batch_indices = indices[i:i + self.batch_size]
            batch_data = [self.dataset[idx] for idx in batch_indices]

            yield batch_data
```

### Performance Monitoring and Optimization

```python
# ml/src/utils/performance_monitor.py
import time
import psutil
import torch
from typing import Dict, List
import matplotlib.pyplot as plt

class TrainingPerformanceMonitor:
    """Monitor training performance and resource usage"""

    def __init__(self):
        self.metrics = {
            'epoch_times': [],
            'memory_usage': [],
            'swap_usage': [],
            'cpu_usage': [],
            'gpu_memory': [],
            'loss_values': []
        }
        self.start_time = None

    def start_epoch(self):
        """Start monitoring an epoch"""
        self.start_time = time.time()

    def end_epoch(self, loss: float):
        """End epoch monitoring and record metrics"""
        if self.start_time is None:
            return

        # Calculate epoch time
        epoch_time = time.time() - self.start_time
        self.metrics['epoch_times'].append(epoch_time)

        # Record system metrics
        memory = psutil.virtual_memory()
        swap = psutil.swap_memory()
        cpu = psutil.cpu_percent()

        self.metrics['memory_usage'].append(memory.percent)
        self.metrics['swap_usage'].append(swap.percent)
        self.metrics['cpu_usage'].append(cpu)
        self.metrics['loss_values'].append(loss)

        # Record GPU memory if available
        if torch.backends.mps.is_available():
            # MPS doesn't have direct memory query, use approximation
            self.metrics['gpu_memory'].append(0)  # Placeholder

        self.start_time = None

    def get_performance_summary(self) -> Dict:
        """Get performance summary"""
        if not self.metrics['epoch_times']:
            return {}

        return {
            'avg_epoch_time': np.mean(self.metrics['epoch_times']),
            'total_training_time': sum(self.metrics['epoch_times']),
            'peak_memory_usage': max(self.metrics['memory_usage']),
            'avg_memory_usage': np.mean(self.metrics['memory_usage']),
            'peak_swap_usage': max(self.metrics['swap_usage']),
            'avg_cpu_usage': np.mean(self.metrics['cpu_usage']),
            'final_loss': self.metrics['loss_values'][-1] if self.metrics['loss_values'] else 0
        }

    def plot_training_metrics(self, save_path: str = None):
        """Plot training metrics"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))

        # Epoch times
        axes[0, 0].plot(self.metrics['epoch_times'])
        axes[0, 0].set_title('Epoch Training Time')
        axes[0, 0].set_ylabel('Time (seconds)')

        # Memory usage
        axes[0, 1].plot(self.metrics['memory_usage'], label='Memory')
        axes[0, 1].plot(self.metrics['swap_usage'], label='Swap')
        axes[0, 1].set_title('Memory Usage')
        axes[0, 1].set_ylabel('Usage (%)')
        axes[0, 1].legend()

        # CPU usage
        axes[1, 0].plot(self.metrics['cpu_usage'])
        axes[1, 0].set_title('CPU Usage')
        axes[1, 0].set_ylabel('Usage (%)')

        # Loss values
        axes[1, 1].plot(self.metrics['loss_values'])
        axes[1, 1].set_title('Training Loss')
        axes[1, 1].set_ylabel('Loss')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.show()

def optimize_training_for_8gb():
    """Apply all optimizations for 8GB system"""

    # Configure PyTorch
    configure_pytorch_for_m2_8gb()

    # Set environment variables
    os.environ['OMP_NUM_THREADS'] = '2'
    os.environ['MKL_NUM_THREADS'] = '2'
    os.environ['NUMEXPR_NUM_THREADS'] = '2'

    # Garbage collection optimization
    import gc
    gc.set_threshold(700, 10, 10)  # More aggressive garbage collection

    print("✅ All optimizations applied for 8GB system")
```

### Quick Reference Commands

```bash
# Training with different configurations
python ml/src/train.py --config quick      # Minimal memory usage
python ml/src/train.py --config standard   # Balanced performance
python ml/src/train.py --config intensive  # Maximum performance

# Memory monitoring during training
python ml/src/train.py --config standard --monitor-memory

# Training with external storage
python ml/src/train.py --config standard \
    --data-path /Volumes/ML_Storage/ML_Data \
    --model-path /Volumes/ML_Storage/ML_Models \
    --cache-path /Volumes/ML_Storage/Cache_Storage
```

This ML training setup ensures optimal performance on M2 MacBook Air 8GB while maintaining the ability to achieve the 85.3% win rate target through efficient resource utilization and smart memory management.
