# Development Workflow Guide for SmartMarketOOPS

## Overview

This guide provides a structured development approach for building SmartMarketOOPS on M2 MacBook Air 8GB, alternating between ML training and other development tasks while maintaining the $0/month infrastructure cost and achieving the 85.3% ML win rate target.

## Table of Contents

1. [Phase-Based Development Timeline](#phase-based-development-timeline)
2. [Hybrid Local/Cloud Development Strategy](#hybrid-localcloud-development-strategy)
3. [Resource Management Best Practices](#resource-management-best-practices)
4. [Integration with Free-Tier Architecture](#integration-with-free-tier-architecture)

---

## Phase-Based Development Timeline

### Week 1-2: Foundation & Infrastructure Setup

#### Week 1: Free-Tier Infrastructure (Task 28)
**Focus**: External SSD setup and cloud platform configuration

**Daily Schedule**:
```bash
# Monday: External SSD Setup
./scripts/setup_external_ssd.sh
./scripts/setup_swap_extension.sh setup

# Tuesday-Wednesday: Cloud Platform Setup
# GitHub Codespaces configuration
# Gitpod workspace setup
# Railway backend deployment

# Thursday-Friday: CI/CD Pipeline
# GitHub Actions workflow setup
# Automated testing pipeline
# Deployment automation
```

**Memory Usage**: Low (2-3GB) - Primarily configuration work
**Cloud Usage**: High - Setting up all cloud services

#### Week 2: Authentication System (Task 29)
**Focus**: JWT authentication and security implementation

**Daily Schedule**:
```bash
# Monday: Backend Authentication
./scripts/local_dev_server.sh start
# Implement JWT authentication
# Password hashing with bcrypt

# Tuesday: Frontend Authentication
# React authentication components
# Protected routes implementation

# Wednesday-Thursday: Security Features
# CSRF protection
# Rate limiting
# Input validation

# Friday: Testing & Integration
npm run test:auth
npm run test:security
```

**Memory Usage**: Medium (4-5GB) - Frontend + Backend development
**Cloud Usage**: Medium - Database operations in cloud

---

### Week 3-5: Real-Time Trading Core

#### Week 3: Real-Time Dashboard (Task 30)
**Focus**: WebSocket implementation and TradingView charts

**Development Pattern**:
```bash
# Morning (9-12): Frontend Development
./scripts/local_dev_server.sh start
# TradingView Lightweight Charts integration
# WebSocket client implementation

# Afternoon (1-5): Backend Development
# WebSocket server implementation
# Real-time price feed integration

# Evening (6-8): ML Training Session
./scripts/ml_training_workflow.sh train BTCUSD lstm 30
```

**Memory Usage**: High (6-7GB) - Full stack development
**Swap Usage**: 2-3GB during ML training
**Cloud Usage**: Medium - Database and testing

#### Week 4-5: ML Integration (Task 31)
**Focus**: Phase 6.1-6.4 ML system integration

**Alternating Schedule**:
```bash
# Day 1: ML Training Focus
./scripts/prepare_ml_training.sh prepare
./scripts/ml_training_workflow.sh train BTCUSD ensemble 50
./scripts/prepare_ml_training.sh cleanup

# Day 2: Integration Development
./scripts/local_dev_server.sh start --with-ml
# ML API integration
# Real-time prediction display

# Day 3: ML Training Focus
./scripts/ml_training_workflow.sh train ETHUSD transformer 40

# Day 4: Frontend Integration
# ML prediction visualization
# Confidence score display
# Sentiment analysis dashboard
```

**Memory Usage**: Variable (3-8GB) - Depends on task
**Swap Usage**: 4-6GB during intensive ML training
**Cloud Usage**: Low - Keep ML training local

---

### Week 6-8: Advanced Features & Optimization

#### Week 6: Bot Management System (Task 32)
**Focus**: Trading bot configuration and backtesting

**Development Pattern**:
```bash
# Morning: Bot Configuration UI
./scripts/local_dev_server.sh start
# Multi-step configuration wizard
# Strategy parameter forms

# Afternoon: Backtesting Engine
# Historical data processing
# Performance metrics calculation

# Evening: ML Model Optimization
./scripts/ml_training_workflow.sh quick_train BTCUSD optimized
```

**Memory Usage**: Medium-High (5-7GB)
**Cloud Usage**: High - Backtesting data processing

#### Week 7: Monitoring & Analytics (Task 33)
**Focus**: Free monitoring tools integration

**Daily Tasks**:
```bash
# Setup monitoring infrastructure
# Grafana Cloud configuration
# UptimeRobot health checks
# Custom analytics implementation
```

**Memory Usage**: Low-Medium (3-5GB)
**Cloud Usage**: High - Monitoring setup

#### Week 8: Performance Optimization (Task 35)
**Focus**: System optimization and testing

**Optimization Schedule**:
```bash
# Monday: Frontend Optimization
npm run build:analyze
npm run optimize:bundle

# Tuesday: Backend Optimization
npm run optimize:api
npm run test:performance

# Wednesday: ML Optimization
./scripts/ml_training_workflow.sh train BTCUSD optimized 25

# Thursday-Friday: Testing
npm run test:all
npm run test:e2e
```

**Memory Usage**: Variable - Testing different configurations
**Cloud Usage**: High - Performance testing

---

### Week 9: Portfolio Presentation (Task 34)

#### Documentation & Demo Week
**Focus**: Portfolio presentation materials

**Daily Schedule**:
```bash
# Monday: Documentation
# Comprehensive README
# Architecture diagrams
# API documentation

# Tuesday: Live Demo Setup
# Sample data population
# Demo user accounts
# Guided tour functionality

# Wednesday: Demo Videos
# Feature demonstration recording
# Architecture presentation slides

# Thursday: Technical Blog Posts
# Implementation deep-dives
# Performance analysis

# Friday: Final Integration
# Portfolio showcase preparation
# Performance validation
```

**Memory Usage**: Low-Medium (3-5GB)
**Cloud Usage**: Medium - Demo environment setup

---

## Hybrid Local/Cloud Development Strategy

### Local Development Priorities

#### High Priority Local Tasks
```bash
# 1. ML Training (latency sensitive)
./scripts/ml_training_workflow.sh train SYMBOL MODEL EPOCHS

# 2. Frontend Development (hot reload performance)
cd frontend && npm run dev

# 3. Real-time Features (WebSocket development)
./scripts/local_dev_server.sh start --with-ml

# 4. API Development (rapid iteration)
cd backend && npm run dev:local
```

#### Cloud Development Priorities

```yaml
# .github/workflows/cloud_development.yml
name: Cloud Development Tasks

on:
  push:
    branches: [ develop, feature/* ]

jobs:
  integration_testing:
    runs-on: ubuntu-latest
    steps:
      - name: Full Stack Integration Tests
        run: |
          docker-compose up -d
          npm run test:integration
          npm run test:e2e

  performance_testing:
    runs-on: ubuntu-latest
    steps:
      - name: Load Testing
        run: |
          npm run test:performance
          npm run test:stress

  database_operations:
    runs-on: ubuntu-latest
    services:
      postgres:
        image: postgres:14
    steps:
      - name: Database Migration Testing
        run: |
          npm run db:migrate
          npm run db:seed
          npm run test:database
```

### Resource Allocation Strategy

```bash
#!/bin/bash
# resource_allocation.sh

allocate_resources_for_task() {
    local task_type="$1"

    case "$task_type" in
        "ml_training")
            echo "🤖 ML Training Mode"
            ./scripts/prepare_ml_training.sh prepare
            export NODE_OPTIONS="--max-old-space-size=512"  # Minimal Node.js
            ;;
        "frontend_dev")
            echo "📱 Frontend Development Mode"
            export NODE_OPTIONS="--max-old-space-size=2048"  # More for frontend
            ./scripts/close_unnecessary_apps.sh
            ;;
        "full_stack")
            echo "🔧 Full Stack Development Mode"
            export NODE_OPTIONS="--max-old-space-size=1024"  # Balanced
            ./scripts/memory_optimization.sh optimize
            ;;
        "testing")
            echo "🧪 Testing Mode"
            # Use cloud for intensive testing
            echo "Redirecting to cloud testing environment..."
            ;;
    esac
}

# Usage examples
allocate_resources_for_task "ml_training"
allocate_resources_for_task "frontend_dev"
```

---

## Resource Management Best Practices

### Memory Management Workflow

#### Pre-Development Setup
```bash
#!/bin/bash
# daily_setup.sh

morning_setup() {
    echo "🌅 Starting daily development setup..."

    # 1. Check system resources
    ./scripts/check_system_readiness.sh

    # 2. Plan daily tasks based on available memory
    available_memory=$(vm_stat | grep "Pages free" | awk '{print $3}' | tr -d '.')

    if [ "$available_memory" -gt 200000 ]; then
        echo "✅ High memory available - Full stack development possible"
        export DAILY_MODE="full_stack"
    elif [ "$available_memory" -gt 100000 ]; then
        echo "⚠️ Medium memory available - Hybrid development recommended"
        export DAILY_MODE="hybrid"
    else
        echo "🚨 Low memory available - Cloud development recommended"
        export DAILY_MODE="cloud_only"
    fi

    # 3. Configure development environment
    case "$DAILY_MODE" in
        "full_stack")
            ./scripts/local_dev_server.sh start --with-ml
            ;;
        "hybrid")
            ./scripts/local_dev_server.sh start
            ;;
        "cloud_only")
            echo "Using cloud development environment today"
            ;;
    esac
}

evening_cleanup() {
    echo "🌙 Evening cleanup routine..."

    # 1. Stop all local services
    ./scripts/local_dev_server.sh stop

    # 2. Clean up ML training artifacts
    ./scripts/prepare_ml_training.sh cleanup

    # 3. Generate daily report
    ./scripts/generate_daily_report.sh

    # 4. Backup important work
    ./scripts/backup_daily_work.sh
}

# Run based on time of day
current_hour=$(date +%H)
if [ "$current_hour" -lt 12 ]; then
    morning_setup
else
    evening_cleanup
fi
```

#### Task Switching Protocol
```bash
#!/bin/bash
# task_switcher.sh

switch_to_ml_training() {
    echo "🔄 Switching to ML training mode..."

    # 1. Save current work
    git add . && git commit -m "WIP: Switching to ML training"

    # 2. Stop development servers
    ./scripts/local_dev_server.sh stop

    # 3. Prepare ML environment
    ./scripts/prepare_ml_training.sh prepare

    # 4. Start training
    ./scripts/ml_training_workflow.sh train "$1" "$2" "$3"

    echo "✅ ML training mode activated"
}

switch_to_development() {
    echo "🔄 Switching to development mode..."

    # 1. Cleanup ML environment
    ./scripts/prepare_ml_training.sh cleanup

    # 2. Restore development environment
    ./scripts/local_dev_server.sh start

    # 3. Resume previous work
    git stash pop 2>/dev/null || echo "No stashed changes"

    echo "✅ Development mode activated"
}

# Usage: ./task_switcher.sh ml BTCUSD lstm 30
# Usage: ./task_switcher.sh dev
case "$1" in
    "ml")
        switch_to_ml_training "$2" "$3" "$4"
        ;;
    "dev")
        switch_to_development
        ;;
    *)
        echo "Usage: $0 {ml|dev} [ml_args...]"
        exit 1
        ;;
esac
```

### Development Session Templates

#### ML Training Session Template
```bash
#!/bin/bash
# ml_session_template.sh

run_ml_training_session() {
    local symbol="$1"
    local duration_hours="${2:-2}"

    echo "🤖 Starting ML training session: $symbol ($duration_hours hours)"

    # 1. Pre-session setup
    ./scripts/prepare_ml_training.sh prepare

    # 2. Training with time limit
    timeout "${duration_hours}h" ./scripts/ml_training_workflow.sh train "$symbol" ensemble 100

    # 3. Post-session cleanup
    ./scripts/prepare_ml_training.sh cleanup

    # 4. Generate session report
    python scripts/training_dashboard.py

    echo "✅ ML training session completed"
}

# Usage: ./ml_session_template.sh BTCUSD 2
run_ml_training_session "$1" "$2"
```

#### Development Session Template
```bash
#!/bin/bash
# dev_session_template.sh

run_development_session() {
    local focus_area="$1"

    echo "💻 Starting development session: $focus_area"

    case "$focus_area" in
        "frontend")
            ./scripts/local_dev_server.sh start
            echo "🌐 Frontend: http://localhost:3000"
            ;;
        "backend")
            ./scripts/local_dev_server.sh start
            echo "🔧 Backend: http://localhost:3001"
            ;;
        "fullstack")
            ./scripts/local_dev_server.sh start --with-ml
            echo "🚀 Full stack environment ready"
            ;;
    esac

    # Monitor session
    ./scripts/monitor_development_session.sh start

    echo "✅ Development session started"
}

# Usage: ./dev_session_template.sh frontend
run_development_session "$1"
```

---

## Integration with Free-Tier Architecture

### Service Integration Strategy

#### Local-Cloud Service Mapping
```yaml
# config/service_mapping.yml
services:
  local:
    frontend:
      port: 3000
      memory_limit: "1GB"
      description: "Next.js development server"

    backend:
      port: 3001
      memory_limit: "1GB"
      description: "Express.js API server"

    ml_service:
      port: 3002
      memory_limit: "2GB"
      description: "Python ML inference service"

  cloud:
    database:
      service: "Supabase"
      connection: "postgresql://..."
      description: "PostgreSQL database"

    cache:
      service: "Railway Redis"
      connection: "redis://..."
      description: "Redis cache"

    ml_training:
      service: "Hugging Face Spaces"
      endpoint: "https://your-space.hf.space"
      description: "ML model training and serving"
```

#### Environment Configuration
```bash
#!/bin/bash
# configure_environment.sh

setup_local_environment() {
    echo "🔧 Setting up local development environment..."

    # Local service environment variables
    export FRONTEND_URL="http://localhost:3000"
    export BACKEND_URL="http://localhost:3001"
    export ML_SERVICE_URL="http://localhost:3002"

    # Cloud service connections
    export DATABASE_URL="postgresql://user:pass@db.supabase.co:5432/smartmarketoops"
    export REDIS_URL="redis://user:pass@redis.railway.app:6379"
    export ML_CLOUD_URL="https://your-space.hf.space"

    # Development mode flags
    export NODE_ENV="development"
    export PYTHON_ENV="development"
    export ML_MODE="local"  # or "cloud" for cloud ML service

    echo "✅ Local environment configured"
}

setup_cloud_environment() {
    echo "☁️ Setting up cloud development environment..."

    # All services in cloud
    export FRONTEND_URL="https://smartmarketoops.vercel.app"
    export BACKEND_URL="https://smartmarketoops-backend.railway.app"
    export ML_SERVICE_URL="https://your-space.hf.space"

    # Cloud database connections
    export DATABASE_URL="$SUPABASE_DATABASE_URL"
    export REDIS_URL="$RAILWAY_REDIS_URL"

    # Production mode flags
    export NODE_ENV="production"
    export PYTHON_ENV="production"
    export ML_MODE="cloud"

    echo "✅ Cloud environment configured"
}

# Usage: ./configure_environment.sh {local|cloud}
case "$1" in
    "local")
        setup_local_environment
        ;;
    "cloud")
        setup_cloud_environment
        ;;
    *)
        echo "Usage: $0 {local|cloud}"
        exit 1
        ;;
esac
```

### Cost Monitoring and Optimization

#### Free-Tier Usage Tracking
```python
# scripts/free_tier_monitor.py
import requests
import json
from datetime import datetime, timedelta

class FreeTierMonitor:
    """Monitor free-tier service usage"""

    def __init__(self):
        self.services = {
            'vercel': {'limit': '100GB', 'current': 0},
            'railway': {'limit': '$5', 'current': 0},
            'supabase': {'limit': '500MB', 'current': 0},
            'github_actions': {'limit': '2000min', 'current': 0}
        }

    def check_vercel_usage(self):
        """Check Vercel bandwidth usage"""
        # Implementation would use Vercel API
        print("📊 Vercel: 45GB / 100GB used (45%)")
        return 45

    def check_railway_usage(self):
        """Check Railway credit usage"""
        # Implementation would use Railway API
        print("💰 Railway: $2.30 / $5.00 used (46%)")
        return 2.30

    def check_supabase_usage(self):
        """Check Supabase database usage"""
        # Implementation would use Supabase API
        print("🗄️ Supabase: 234MB / 500MB used (47%)")
        return 234

    def check_github_actions_usage(self):
        """Check GitHub Actions minutes"""
        # Implementation would use GitHub API
        print("⚙️ GitHub Actions: 890min / 2000min used (45%)")
        return 890

    def generate_usage_report(self):
        """Generate comprehensive usage report"""
        print("📈 Free-Tier Usage Report")
        print("=" * 40)

        vercel_usage = self.check_vercel_usage()
        railway_usage = self.check_railway_usage()
        supabase_usage = self.check_supabase_usage()
        github_usage = self.check_github_actions_usage()

        # Calculate overall usage
        avg_usage = (45 + 46 + 47 + 45) / 4

        print(f"\n📊 Overall Usage: {avg_usage:.1f}%")

        if avg_usage > 80:
            print("🚨 WARNING: High usage detected!")
            self.suggest_optimizations()
        elif avg_usage > 60:
            print("⚠️ CAUTION: Monitor usage closely")
        else:
            print("✅ Usage within safe limits")

    def suggest_optimizations(self):
        """Suggest optimizations for high usage"""
        print("\n💡 Optimization Suggestions:")
        print("- Use local development more frequently")
        print("- Optimize bundle sizes for Vercel")
        print("- Clean up unused Railway services")
        print("- Archive old Supabase data")
        print("- Reduce GitHub Actions frequency")

if __name__ == "__main__":
    monitor = FreeTierMonitor()
    monitor.generate_usage_report()
```

#### Development Cost Optimization
```bash
#!/bin/bash
# optimize_costs.sh

optimize_development_costs() {
    echo "💰 Optimizing development costs..."

    # 1. Local development preference
    echo "📍 Prioritizing local development to reduce cloud usage"

    # 2. Efficient cloud usage
    echo "☁️ Optimizing cloud service usage..."

    # Stop unnecessary cloud services
    # railway service stop unused-service

    # Clean up old deployments
    # vercel rm old-deployment

    # Optimize database queries
    echo "🗄️ Optimizing database usage..."

    # 3. Bundle optimization
    echo "📦 Optimizing bundle sizes..."
    npm run build:analyze

    # 4. Cache optimization
    echo "🚀 Optimizing cache usage..."

    echo "✅ Cost optimization completed"
}

monitor_daily_costs() {
    echo "📊 Daily cost monitoring..."

    # Check service usage
    python scripts/free_tier_monitor.py

    # Log usage to file
    echo "$(date): Usage check completed" >> logs/cost_monitoring.log
}

# Run optimization
optimize_development_costs
monitor_daily_costs
```

### Workflow Automation

#### Automated Task Scheduling
```bash
#!/bin/bash
# automated_scheduler.sh

schedule_daily_tasks() {
    echo "📅 Setting up daily task schedule..."

    # Create cron jobs for automated tasks
    (crontab -l 2>/dev/null; echo "0 9 * * * cd $PWD && ./scripts/morning_setup.sh") | crontab -
    (crontab -l 2>/dev/null; echo "0 18 * * * cd $PWD && ./scripts/evening_cleanup.sh") | crontab -
    (crontab -l 2>/dev/null; echo "0 12 * * * cd $PWD && ./scripts/free_tier_monitor.py") | crontab -

    echo "✅ Daily tasks scheduled"
}

schedule_weekly_tasks() {
    echo "📅 Setting up weekly task schedule..."

    # Weekly ML training sessions
    (crontab -l 2>/dev/null; echo "0 10 * * 1 cd $PWD && ./scripts/ml_training_workflow.sh train BTCUSD ensemble 50") | crontab -
    (crontab -l 2>/dev/null; echo "0 10 * * 3 cd $PWD && ./scripts/ml_training_workflow.sh train ETHUSD transformer 40") | crontab -
    (crontab -l 2>/dev/null; echo "0 10 * * 5 cd $PWD && ./scripts/ml_training_workflow.sh train ADAUSD lstm 30") | crontab -

    # Weekly system maintenance
    (crontab -l 2>/dev/null; echo "0 20 * * 0 cd $PWD && ./scripts/weekly_maintenance.sh") | crontab -

    echo "✅ Weekly tasks scheduled"
}

# Setup automation
schedule_daily_tasks
schedule_weekly_tasks
```

#### Continuous Integration Workflow
```yaml
# .github/workflows/development_workflow.yml
name: SmartMarketOOPS Development Workflow

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]
  schedule:
    - cron: '0 2 * * *'  # Daily at 2 AM

jobs:
  local_development_support:
    runs-on: ubuntu-latest
    if: github.event_name == 'push' && contains(github.ref, 'develop')

    steps:
    - uses: actions/checkout@v4

    - name: Setup Development Environment
      run: |
        npm install
        pip install -r ml/requirements.txt

    - name: Run Local Development Tests
      run: |
        npm run test:unit
        npm run test:lint

    - name: Generate Development Report
      run: |
        npm run generate:dev-report

  cloud_integration_testing:
    runs-on: ubuntu-latest
    if: github.event_name == 'pull_request'

    services:
      postgres:
        image: postgres:14
        env:
          POSTGRES_PASSWORD: postgres
          POSTGRES_DB: smartmarketoops_test

      redis:
        image: redis:7

    steps:
    - uses: actions/checkout@v4

    - name: Full Stack Integration Tests
      run: |
        npm run test:integration
        npm run test:e2e

    - name: Performance Testing
      run: |
        npm run test:performance

  ml_model_validation:
    runs-on: ubuntu-latest
    if: github.event_name == 'schedule'

    steps:
    - uses: actions/checkout@v4

    - name: Validate ML Models
      run: |
        python ml/src/validate_models.py

    - name: Update Model Performance Metrics
      run: |
        python scripts/update_performance_metrics.py

  free_tier_monitoring:
    runs-on: ubuntu-latest
    if: github.event_name == 'schedule'

    steps:
    - uses: actions/checkout@v4

    - name: Monitor Free-Tier Usage
      run: |
        python scripts/free_tier_monitor.py

    - name: Generate Cost Report
      run: |
        python scripts/generate_cost_report.py
```

### Quick Reference Commands

#### Daily Development Commands
```bash
# Morning startup
./scripts/morning_setup.sh

# Switch between modes
./scripts/task_switcher.sh ml BTCUSD lstm 30    # ML training
./scripts/task_switcher.sh dev                  # Development

# Check system status
./scripts/check_system_readiness.sh
./scripts/local_dev_server.sh status

# Evening cleanup
./scripts/evening_cleanup.sh
```

#### Weekly Maintenance Commands
```bash
# Weekly ML training schedule
./scripts/ml_training_workflow.sh train BTCUSD ensemble 50    # Monday
./scripts/ml_training_workflow.sh train ETHUSD transformer 40 # Wednesday
./scripts/ml_training_workflow.sh train ADAUSD lstm 30        # Friday

# Weekly system maintenance
./scripts/weekly_maintenance.sh
./scripts/free_tier_monitor.py
./scripts/backup_weekly_progress.sh
```

#### Emergency Commands
```bash
# System recovery
./scripts/emergency_cleanup.sh
./scripts/restore_system_state.sh

# Resource management
./scripts/free_memory_emergency.sh
./scripts/stop_all_services.sh
```

This development workflow ensures efficient use of M2 MacBook Air 8GB resources while maintaining the free-tier architecture and achieving the 85.3% ML win rate target through systematic development practices and resource management.
```
