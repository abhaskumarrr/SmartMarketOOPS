# M2 MacBook Air 8GB Optimization Guide for SmartMarketOOPS

## Overview

This guide provides comprehensive optimization strategies for developing SmartMarketOOPS on a base M2 MacBook Air (8GB RAM, 256GB storage) while maintaining the $0/month infrastructure cost goal and achieving the 85.3% ML win rate target.

## Table of Contents

1. [External SSD Configuration](#external-ssd-configuration)
2. [Cloud Platform Offloading](#cloud-platform-offloading)
3. [Memory Management Strategies](#memory-management-strategies)
4. [System Preparation Scripts](#system-preparation-scripts)
5. [Performance Monitoring](#performance-monitoring)

---

## External SSD Configuration

### Hardware Requirements

**Recommended External SSD:**
- **Capacity**: 500GB+ (1TB preferred)
- **Interface**: USB-C 3.1 or Thunderbolt 3/4
- **Speed**: 500MB/s+ read/write
- **Format**: APFS (optimized for macOS)

### SSD Partitioning Setup

```bash
#!/bin/bash
# setup_external_ssd.sh

setup_ml_storage() {
    echo "🔧 Setting up external SSD for ML development..."

    # Identify external drive (replace disk2 with your drive)
    EXTERNAL_DISK="/dev/disk2"

    # Format with APFS
    diskutil eraseDisk APFS "ML_Storage" $EXTERNAL_DISK

    # Create optimized partitions
    diskutil addVolume disk2 APFS "ML_Data" 200GB
    diskutil addVolume disk2 APFS "ML_Models" 100GB
    diskutil addVolume disk2 APFS "Swap_Extension" 50GB
    diskutil addVolume disk2 APFS "Docker_Storage" 100GB
    diskutil addVolume disk2 APFS "Cache_Storage" 50GB

    echo "✅ External SSD partitioned successfully"
}

# Create directory structure
create_ml_directories() {
    BASE_PATH="/Volumes/ML_Storage"

    # ML Data directories
    mkdir -p "$BASE_PATH/ML_Data/raw_data"
    mkdir -p "$BASE_PATH/ML_Data/processed_data"
    mkdir -p "$BASE_PATH/ML_Data/cache"
    mkdir -p "$BASE_PATH/ML_Data/temp"

    # ML Models directories
    mkdir -p "$BASE_PATH/ML_Models/checkpoints"
    mkdir -p "$BASE_PATH/ML_Models/trained_models"
    mkdir -p "$BASE_PATH/ML_Models/experiments"
    mkdir -p "$BASE_PATH/ML_Models/backups"

    # Set proper permissions
    chmod -R 755 "$BASE_PATH"

    echo "✅ ML directory structure created"
}

setup_ml_storage
create_ml_directories
```

### Swap File Configuration

```bash
#!/bin/bash
# setup_swap_extension.sh

setup_additional_swap() {
    SWAP_PATH="/Volumes/ML_Storage/Swap_Extension"
    SWAP_SIZE="8192"  # 8GB

    echo "💾 Setting up additional swap space..."

    # Create swap file
    sudo dd if=/dev/zero of="$SWAP_PATH/swapfile" bs=1m count=$SWAP_SIZE

    # Set permissions
    sudo chmod 600 "$SWAP_PATH/swapfile"

    # Initialize swap
    sudo mkswap "$SWAP_PATH/swapfile"

    echo "✅ Additional swap file created (${SWAP_SIZE}MB)"
}

enable_ml_swap() {
    SWAP_FILE="/Volumes/ML_Storage/Swap_Extension/swapfile"

    if [ -f "$SWAP_FILE" ]; then
        sudo swapon "$SWAP_FILE"
        echo "✅ ML training swap enabled"
        echo "📊 Current swap status:"
        swapon -s
    else
        echo "❌ Swap file not found. Run setup first."
    fi
}

disable_ml_swap() {
    SWAP_FILE="/Volumes/ML_Storage/Swap_Extension/swapfile"

    if [ -f "$SWAP_FILE" ]; then
        sudo swapoff "$SWAP_FILE"
        echo "✅ ML training swap disabled"
    fi
}

# Usage: ./setup_swap_extension.sh {setup|enable|disable}
case "$1" in
    setup)
        setup_additional_swap
        ;;
    enable)
        enable_ml_swap
        ;;
    disable)
        disable_ml_swap
        ;;
    *)
        echo "Usage: $0 {setup|enable|disable}"
        exit 1
        ;;
esac
```

---

## Cloud Platform Offloading

### GitHub Codespaces Configuration

Create `.devcontainer/devcontainer.json`:

```json
{
  "name": "SmartMarketOOPS Cloud Development",
  "image": "mcr.microsoft.com/devcontainers/python:3.11",
  "features": {
    "ghcr.io/devcontainers/features/node:1": {
      "version": "20"
    },
    "ghcr.io/devcontainers/features/docker-in-docker:2": {},
    "ghcr.io/devcontainers/features/postgresql:1": {
      "version": "14"
    }
  },
  "customizations": {
    "vscode": {
      "extensions": [
        "ms-python.python",
        "ms-toolsai.jupyter",
        "bradlc.vscode-tailwindcss",
        "ms-vscode.vscode-typescript-next"
      ],
      "settings": {
        "python.defaultInterpreterPath": "/usr/local/bin/python",
        "typescript.preferences.includePackageJsonAutoImports": "off"
      }
    }
  },
  "forwardPorts": [3000, 3001, 3002, 5432, 6379],
  "postCreateCommand": "npm run setup:cloud",
  "remoteUser": "vscode",
  "mounts": [
    "source=smartmarketoops-node-modules,target=${containerWorkspaceFolder}/node_modules,type=volume"
  ]
}
```

### Gitpod Configuration

Create `.gitpod.yml`:

```yaml
image:
  file: .gitpod.Dockerfile

tasks:
  - name: Environment Setup
    init: |
      npm install -g pnpm
      pip install -r ml/requirements.txt
      pnpm install:all
    command: |
      echo "🚀 SmartMarketOOPS development environment ready"

  - name: Database Setup
    command: |
      pg_ctl start -D /workspace/.pgsql/data
      createdb smartmarketoops
      npm run db:migrate

  - name: Services
    command: |
      npm run dev:cloud

ports:
  - port: 3000
    onOpen: open-preview
    description: "Frontend Dashboard"
  - port: 3001
    onOpen: ignore
    description: "Backend API"
  - port: 3002
    onOpen: ignore
    description: "ML Service"
  - port: 5432
    onOpen: ignore
    description: "PostgreSQL"

vscode:
  extensions:
    - ms-python.python
    - ms-toolsai.jupyter
    - bradlc.vscode-tailwindcss
```

### Cloud vs Local Task Distribution

```bash
#!/bin/bash
# cloud_offload_strategy.sh

# Tasks to run in cloud environments
CLOUD_TASKS=(
    "docker-compose up -d"           # Full stack with containers
    "npm run test:integration"       # Integration testing
    "npm run test:e2e"              # End-to-end testing
    "npm run build:production"       # Production builds
    "npm run db:seed"               # Database seeding
    "npm run performance:test"       # Performance testing
)

# Tasks to keep local
LOCAL_TASKS=(
    "npm run dev:frontend"          # Frontend development
    "python ml/src/train.py"        # ML training (short sessions)
    "npm run dev:backend"           # Backend API development
    "npm run test:unit"             # Unit testing
)

print_task_distribution() {
    echo "🌥️ CLOUD TASKS (GitHub Codespaces/Gitpod):"
    for task in "${CLOUD_TASKS[@]}"; do
        echo "  ✅ $task"
    done

    echo ""
    echo "💻 LOCAL TASKS (M2 MacBook Air):"
    for task in "${LOCAL_TASKS[@]}"; do
        echo "  ✅ $task"
    done
}

print_task_distribution
```

---

## Memory Management Strategies

### System Memory Optimization

```bash
#!/bin/bash
# memory_optimization.sh

optimize_system_memory() {
    echo "🧠 Optimizing system memory for ML training..."

    # 1. Adjust virtual memory settings
    sudo sysctl vm.swappiness=60          # More aggressive swapping
    sudo sysctl vm.vfs_cache_pressure=50  # Balance cache vs swap
    sudo sysctl vm.overcommit_memory=1    # Allow memory overcommit

    # 2. Optimize memory pressure handling
    sudo sysctl vm.memory_pressure_disable_zone_reclaim=1

    # 3. Clear system caches
    sudo purge

    # 4. Disable unnecessary services during training
    sudo launchctl unload -w /System/Library/LaunchDaemons/com.apple.metadata.mds.plist 2>/dev/null

    echo "✅ System memory optimized for ML training"
}

restore_normal_memory() {
    echo "🔄 Restoring normal memory settings..."

    # Restore conservative settings
    sudo sysctl vm.swappiness=10
    sudo sysctl vm.vfs_cache_pressure=100
    sudo sysctl vm.overcommit_memory=0

    # Re-enable services
    sudo launchctl load -w /System/Library/LaunchDaemons/com.apple.metadata.mds.plist 2>/dev/null

    echo "✅ Normal memory settings restored"
}

# Usage: ./memory_optimization.sh {optimize|restore}
case "$1" in
    optimize)
        optimize_system_memory
        ;;
    restore)
        restore_normal_memory
        ;;
    *)
        echo "Usage: $0 {optimize|restore}"
        exit 1
        ;;
esac
```

### Application Memory Management

```bash
#!/bin/bash
# app_memory_management.sh

close_memory_intensive_apps() {
    echo "🔄 Closing memory-intensive applications..."

    # Close browsers
    osascript -e 'quit app "Safari"' 2>/dev/null
    osascript -e 'quit app "Google Chrome"' 2>/dev/null
    osascript -e 'quit app "Firefox"' 2>/dev/null

    # Close communication apps
    osascript -e 'quit app "Slack"' 2>/dev/null
    osascript -e 'quit app "Discord"' 2>/dev/null
    osascript -e 'quit app "Zoom"' 2>/dev/null

    # Close other memory-heavy apps
    osascript -e 'quit app "Docker Desktop"' 2>/dev/null
    osascript -e 'quit app "Spotify"' 2>/dev/null

    echo "✅ Memory-intensive applications closed"
}

set_development_priorities() {
    echo "⚡ Setting process priorities for development..."

    # Higher priority for development tools
    sudo renice -10 $(pgrep "Code") 2>/dev/null      # VS Code
    sudo renice -10 $(pgrep "node") 2>/dev/null      # Node.js processes
    sudo renice -10 $(pgrep "python") 2>/dev/null    # Python processes

    # Lower priority for background processes
    sudo renice 10 $(pgrep "Spotlight") 2>/dev/null
    sudo renice 10 $(pgrep "mds") 2>/dev/null

    echo "✅ Process priorities optimized"
}

close_memory_intensive_apps
set_development_priorities
```

---

## System Preparation Scripts

### Complete ML Training Preparation

```bash
#!/bin/bash
# prepare_ml_training.sh

prepare_complete_ml_environment() {
    echo "🚀 Preparing complete ML training environment..."

    # 1. Check external SSD
    if [ ! -d "/Volumes/ML_Storage" ]; then
        echo "❌ External SSD not mounted. Please connect ML_Storage drive."
        exit 1
    fi

    # 2. Enable additional swap
    ./setup_swap_extension.sh enable

    # 3. Optimize memory settings
    ./memory_optimization.sh optimize

    # 4. Close unnecessary applications
    ./app_memory_management.sh

    # 5. Set environment variables
    export ML_DATA_PATH="/Volumes/ML_Storage/ML_Data"
    export ML_MODEL_PATH="/Volumes/ML_Storage/ML_Models"
    export ML_CACHE_PATH="/Volumes/ML_Storage/Cache_Storage"
    export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0

    # 6. Configure Python for memory efficiency
    export PYTHONHASHSEED=0
    export OMP_NUM_THREADS=2
    export MKL_NUM_THREADS=2

    # 7. Start memory monitoring
    ./monitor_training.sh start &

    echo "✅ ML training environment fully prepared"
    echo "📊 Memory status:"
    vm_stat | head -5
}

cleanup_after_training() {
    echo "🧹 Cleaning up after ML training..."

    # 1. Stop memory monitoring
    ./monitor_training.sh stop

    # 2. Restore normal memory settings
    ./memory_optimization.sh restore

    # 3. Disable additional swap
    ./setup_swap_extension.sh disable

    # 4. Clear Python cache
    find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null

    # 5. Force garbage collection
    python3 -c "import gc; gc.collect()"

    # 6. Clear system caches
    sudo purge

    echo "✅ Cleanup completed"
}

# Usage: ./prepare_ml_training.sh {prepare|cleanup}
case "$1" in
    prepare)
        prepare_complete_ml_environment
        ;;
    cleanup)
        cleanup_after_training
        ;;
    *)
        echo "Usage: $0 {prepare|cleanup}"
        echo "  prepare - Set up complete ML training environment"
        echo "  cleanup - Clean up after training session"
        exit 1
        ;;
esac
```

---

## Performance Monitoring

### Real-Time Memory Monitoring

```bash
#!/bin/bash
# monitor_training.sh

MONITOR_PID_FILE="/tmp/ml_monitor.pid"
LOG_FILE="/Volumes/ML_Storage/Cache_Storage/training_monitor.log"

start_monitoring() {
    echo "👀 Starting ML training monitoring..."

    # Create log file
    touch "$LOG_FILE"

    # Start monitoring in background
    {
        while true; do
            timestamp=$(date '+%Y-%m-%d %H:%M:%S')

            # Get memory statistics
            memory_info=$(vm_stat)
            memory_pressure=$(echo "$memory_info" | grep "Pages free" | awk '{print $3}' | tr -d '.')

            # Get swap usage
            swap_info=$(swapon -s 2>/dev/null | tail -1)
            swap_used=$(echo "$swap_info" | awk '{print $4}' 2>/dev/null || echo "0")

            # Get CPU usage
            cpu_usage=$(top -l 1 | grep "CPU usage" | awk '{print $3}' | tr -d '%')

            # Get temperature (if available)
            temp_info=$(sudo powermetrics -n 1 -s thermal 2>/dev/null | grep "CPU die temperature" | awk '{print $4}' || echo "N/A")

            # Log information
            echo "$timestamp,Memory_Free:$memory_pressure,Swap_Used:${swap_used}KB,CPU:${cpu_usage}%,Temp:${temp_info}" >> "$LOG_FILE"

            # Check for critical conditions
            if [ "$memory_pressure" -lt 50000 ]; then
                echo "🚨 CRITICAL: Very low memory available" | tee -a "$LOG_FILE"
            fi

            if [ "$swap_used" -gt 4000000 ]; then
                echo "⚠️ WARNING: High swap usage (${swap_used}KB)" | tee -a "$LOG_FILE"
            fi

            sleep 10
        done
    } &

    # Save monitoring PID
    echo $! > "$MONITOR_PID_FILE"
    echo "✅ Monitoring started (PID: $(cat $MONITOR_PID_FILE))"
}

stop_monitoring() {
    if [ -f "$MONITOR_PID_FILE" ]; then
        monitor_pid=$(cat "$MONITOR_PID_FILE")
        kill "$monitor_pid" 2>/dev/null
        rm "$MONITOR_PID_FILE"
        echo "✅ Monitoring stopped"
    else
        echo "❌ No monitoring process found"
    fi
}

show_stats() {
    if [ -f "$LOG_FILE" ]; then
        echo "📊 Recent training statistics:"
        tail -20 "$LOG_FILE"
    else
        echo "❌ No monitoring log found"
    fi
}

# Usage: ./monitor_training.sh {start|stop|stats}
case "$1" in
    start)
        start_monitoring
        ;;
    stop)
        stop_monitoring
        ;;
    stats)
        show_stats
        ;;
    *)
        echo "Usage: $0 {start|stop|stats}"
        exit 1
        ;;
esac
```

### Training Performance Dashboard

```python
# scripts/training_dashboard.py
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import psutil
import time

class TrainingDashboard:
    def __init__(self, log_file="/Volumes/ML_Storage/Cache_Storage/training_monitor.log"):
        self.log_file = log_file

    def parse_log_data(self):
        """Parse monitoring log data"""
        try:
            data = []
            with open(self.log_file, 'r') as f:
                for line in f:
                    parts = line.strip().split(',')
                    if len(parts) >= 4:
                        timestamp = datetime.strptime(parts[0], '%Y-%m-%d %H:%M:%S')
                        memory_free = int(parts[1].split(':')[1])
                        swap_used = int(parts[2].split(':')[1].replace('KB', ''))
                        cpu_usage = float(parts[3].split(':')[1].replace('%', ''))

                        data.append({
                            'timestamp': timestamp,
                            'memory_free': memory_free,
                            'swap_used': swap_used / 1024,  # Convert to MB
                            'cpu_usage': cpu_usage
                        })

            return pd.DataFrame(data)
        except Exception as e:
            print(f"Error parsing log data: {e}")
            return pd.DataFrame()

    def create_dashboard(self):
        """Create training performance dashboard"""
        df = self.parse_log_data()

        if df.empty:
            print("No data available for dashboard")
            return

        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('SmartMarketOOPS ML Training Performance Dashboard', fontsize=16)

        # Memory usage over time
        axes[0, 0].plot(df['timestamp'], df['memory_free'], 'b-', linewidth=2)
        axes[0, 0].set_title('Memory Free Over Time')
        axes[0, 0].set_ylabel('Memory Free (pages)')
        axes[0, 0].grid(True, alpha=0.3)

        # Swap usage over time
        axes[0, 1].plot(df['timestamp'], df['swap_used'], 'r-', linewidth=2)
        axes[0, 1].set_title('Swap Usage Over Time')
        axes[0, 1].set_ylabel('Swap Used (MB)')
        axes[0, 1].grid(True, alpha=0.3)

        # CPU usage over time
        axes[1, 0].plot(df['timestamp'], df['cpu_usage'], 'g-', linewidth=2)
        axes[1, 0].set_title('CPU Usage Over Time')
        axes[1, 0].set_ylabel('CPU Usage (%)')
        axes[1, 0].grid(True, alpha=0.3)

        # Resource correlation
        axes[1, 1].scatter(df['cpu_usage'], df['swap_used'], alpha=0.6)
        axes[1, 1].set_title('CPU vs Swap Usage Correlation')
        axes[1, 1].set_xlabel('CPU Usage (%)')
        axes[1, 1].set_ylabel('Swap Used (MB)')
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('/Volumes/ML_Storage/Cache_Storage/training_dashboard.png', dpi=300, bbox_inches='tight')
        plt.show()

        # Print summary statistics
        print("\n📊 Training Session Summary:")
        print(f"Duration: {df['timestamp'].max() - df['timestamp'].min()}")
        print(f"Average Memory Free: {df['memory_free'].mean():.0f} pages")
        print(f"Peak Swap Usage: {df['swap_used'].max():.1f} MB")
        print(f"Average CPU Usage: {df['cpu_usage'].mean():.1f}%")
        print(f"Peak CPU Usage: {df['cpu_usage'].max():.1f}%")

if __name__ == "__main__":
    dashboard = TrainingDashboard()
    dashboard.create_dashboard()
```

---

## Quick Reference Commands

### Daily Development Workflow

```bash
# Morning setup
./prepare_ml_training.sh prepare

# Start ML training
python ml/src/train.py --config local_8gb

# Monitor progress
./monitor_training.sh stats

# Evening cleanup
./prepare_ml_training.sh cleanup
```

### Emergency Memory Recovery

```bash
# If system becomes unresponsive
sudo purge                    # Clear caches
./memory_optimization.sh restore
./setup_swap_extension.sh disable
killall Python node          # Stop development processes
```

### Performance Verification

```bash
# Check system readiness
vm_stat | head -5
swapon -s
df -h /Volumes/ML_Storage
```

This optimization guide ensures your M2 MacBook Air 8GB can effectively handle SmartMarketOOPS ML training while maintaining system stability and achieving the target 85.3% win rate.
```
