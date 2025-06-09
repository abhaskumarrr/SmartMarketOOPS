# 🐳 SmartMarketOOPS Docker Setup Guide

Complete Docker containerization for the SmartMarketOOPS trading platform with development and production configurations.

## 📋 **Prerequisites**

### Required Software
- **Docker Desktop** (latest version)
- **Git** (for cloning the repository)
- **8GB+ RAM** (recommended for all services)

### System Requirements
- **macOS**: Docker Desktop for Mac
- **Windows**: Docker Desktop for Windows (with WSL2)
- **Linux**: Docker Engine + Docker Compose

## 🚀 **Quick Start**

### 1. Install Docker Desktop
```bash
# Download from: https://www.docker.com/products/docker-desktop
# Or use package managers:

# macOS (Homebrew)
brew install --cask docker

# Windows (Chocolatey)
choco install docker-desktop

# Linux (Ubuntu/Debian)
curl -fsSL https://get.docker.com -o get-docker.sh
sh get-docker.sh
```

### 2. Clone and Setup
```bash
git clone https://github.com/abhaskumarrr/SmartMarketOOPS.git
cd SmartMarketOOPS

# Validate Docker setup
./scripts/validate-docker-setup.sh

# Create environment file
cp .env.docker .env
# Edit .env with your actual API credentials
```

### 3. Start Development Environment
```bash
# One-command setup
./scripts/docker-dev.sh dev

# Or step by step
./scripts/docker-dev.sh build
./scripts/docker-dev.sh start
```

## 🏗️ **Architecture Overview**

### Services
| Service | Port | Description |
|---------|------|-------------|
| **Frontend** | 3000 | Next.js 15 + React 19 UI |
| **Backend** | 3005 | Node.js + TypeScript API |
| **ML System** | 8000 | Python + FastAPI ML Engine |
| **PostgreSQL** | 5432 | Primary database |
| **Redis** | 6379 | Caching & sessions |
| **QuestDB** | 9000 | Time-series data |
| **Prometheus** | 9090 | Metrics collection |
| **Grafana** | 3001 | Monitoring dashboards |

### Container Structure
```
smartmarket/
├── frontend/          # Next.js frontend container
├── backend/           # Node.js API container  
├── ml-system/         # Python ML container
├── postgres/          # PostgreSQL database
├── redis/             # Redis cache
├── questdb/           # Time-series database
└── monitoring/        # Prometheus + Grafana
```

## ⚙️ **Configuration**

### Environment Variables
```bash
# Core Services
FRONTEND_PORT=3000
BACKEND_PORT=3005
ML_PORT=8000

# Database
POSTGRES_USER=postgres
POSTGRES_PASSWORD=postgres
POSTGRES_DB=smoops

# Trading API (Required)
DELTA_EXCHANGE_API_KEY=your_api_key
DELTA_EXCHANGE_SECRET=your_secret
DELTA_EXCHANGE_TESTNET=true

# Security
JWT_SECRET=your_jwt_secret
ENCRYPTION_KEY=your_encryption_key
```

### Docker Compose Files
- **`docker-compose.yml`**: Base production configuration
- **`docker-compose.override.yml`**: Development overrides (auto-loaded)
- **`.env.docker`**: Environment template

## 🛠️ **Development Workflow**

### Daily Development
```bash
# Start development environment
./scripts/docker-dev.sh dev

# View logs
./scripts/docker-dev.sh logs
./scripts/docker-dev.sh logs backend  # Specific service

# Check health
./scripts/docker-dev.sh health

# Restart services
./scripts/docker-dev.sh restart
```

### Hot Reloading
All services support hot reloading in development:
- **Frontend**: Next.js Fast Refresh
- **Backend**: Nodemon auto-restart
- **ML System**: Python file watching

### Volume Mounts
Development containers mount source code for live editing:
```yaml
volumes:
  - ./frontend:/app          # Frontend source
  - ./backend:/app           # Backend source  
  - .:/app                   # ML system source
  - ./data:/app/data         # Persistent data
```

## 🔍 **Monitoring & Health Checks**

### Health Endpoints
- **Frontend**: http://localhost:3000/api/health
- **Backend**: http://localhost:3005/health
- **ML System**: http://localhost:8000/health

### Monitoring Stack
- **Prometheus**: http://localhost:9090
- **Grafana**: http://localhost:3001 (admin/admin)
- **QuestDB Console**: http://localhost:9000

### Service Status
```bash
# Check all services
./scripts/docker-dev.sh status

# Docker native commands
docker-compose ps
docker-compose logs -f
```

## 🚀 **Production Deployment**

### Production Build
```bash
# Build production images
docker-compose -f docker-compose.yml build

# Start production stack
docker-compose -f docker-compose.yml up -d
```

### Production Considerations
- Use external databases for scalability
- Configure proper secrets management
- Set up SSL/TLS termination
- Configure monitoring and alerting
- Use container orchestration (Kubernetes)

## 🔧 **Troubleshooting**

### Common Issues

#### Port Conflicts
```bash
# Check port usage
lsof -i :3000
lsof -i :3005
lsof -i :8000

# Kill conflicting processes
./scripts/docker-dev.sh stop
```

#### Container Build Failures
```bash
# Clean rebuild
./scripts/docker-dev.sh cleanup
./scripts/docker-dev.sh build
```

#### Database Connection Issues
```bash
# Reset database
docker-compose down -v
docker-compose up -d postgres
# Wait for postgres to be ready
./scripts/docker-dev.sh start
```

#### Memory Issues
```bash
# Check Docker resource usage
docker stats

# Increase Docker Desktop memory allocation
# Docker Desktop > Settings > Resources > Memory
```

### Debug Commands
```bash
# Enter container shell
docker-compose exec backend sh
docker-compose exec frontend sh
docker-compose exec ml-system bash

# View container logs
docker-compose logs -f backend
docker-compose logs -f frontend
docker-compose logs -f ml-system

# Inspect container
docker inspect smartmarket-backend
```

## 📊 **Performance Optimization**

### Development Performance
- Use volume mounts for hot reloading
- Exclude node_modules from mounts
- Use .dockerignore files
- Enable BuildKit for faster builds

### Production Performance
- Multi-stage builds for smaller images
- Non-root users for security
- Health checks for reliability
- Resource limits and requests

## 🔐 **Security Best Practices**

### Container Security
- Non-root users in all containers
- Minimal base images (Alpine Linux)
- No secrets in Dockerfiles
- Regular security updates

### Network Security
- Internal Docker networks
- Exposed ports only when necessary
- SSL/TLS for external communication
- Firewall rules for production

## 📚 **Additional Resources**

### Documentation
- [Docker Documentation](https://docs.docker.com/)
- [Docker Compose Reference](https://docs.docker.com/compose/)
- [Next.js Docker Guide](https://nextjs.org/docs/deployment#docker-image)

### Scripts Reference
- `./scripts/docker-dev.sh` - Development management
- `./scripts/validate-docker-setup.sh` - Configuration validation

## 🎯 **Success Criteria**

After successful setup, you should have:
- ✅ All services running and healthy
- ✅ Frontend accessible at http://localhost:3000
- ✅ Backend API responding at http://localhost:3005
- ✅ ML system operational at http://localhost:8000
- ✅ Hot reloading working for development
- ✅ Monitoring dashboards accessible

## 🆘 **Support**

If you encounter issues:
1. Run `./scripts/validate-docker-setup.sh`
2. Check service logs with `./scripts/docker-dev.sh logs`
3. Verify environment configuration in `.env`
4. Ensure Docker Desktop has sufficient resources
5. Check for port conflicts with existing services

---

**🎉 Your SmartMarketOOPS Docker environment is ready for professional trading operations!**
