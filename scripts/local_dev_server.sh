#!/bin/bash
# local_dev_server.sh - Memory-efficient development server for M2 MacBook Air 8GB

# Configuration
FRONTEND_PORT=3000
BACKEND_PORT=3001
ML_SERVICE_PORT=3002

start_local_development() {
    echo "🚀 Starting local development environment..."
    
    # Create PID file directory
    mkdir -p .local_dev
    
    # Start frontend (Next.js) with memory limits
    echo "📱 Starting frontend development server..."
    cd frontend
    NODE_OPTIONS="--max-old-space-size=1024" npm run dev &
    echo $! > ../.local_dev/frontend.pid
    cd ..
    
    # Start backend (Express.js) with memory limits
    echo "🔧 Starting backend API server..."
    cd backend
    NODE_OPTIONS="--max-old-space-size=1024" npm run dev &
    echo $! > ../.local_dev/backend.pid
    cd ..
    
    # Start ML service (Python FastAPI) - only if needed
    if [ "$1" = "--with-ml" ]; then
        echo "🤖 Starting ML service..."
        cd ml
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
                NODE_OPTIONS="--max-old-space-size=1024" npm run dev &
                echo $! > ../.local_dev/backend.pid
                cd ..
                echo "🔄 Backend restarted"
            fi
            ;;
        "ml")
            if [ -f .local_dev/ml_service.pid ]; then
                kill $(cat .local_dev/ml_service.pid) 2>/dev/null
                cd ml
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
