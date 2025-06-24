# SmartMarketOOPS Project Structure

## Overview

This document describes the organized file structure of the SmartMarketOOPS trading platform after comprehensive cleanup and reorganization.

## Directory Structure

```
SmartMarketOOPS/
├── 📁 backend/                    # Backend Express.js application
│   ├── 📁 src/                    # Source code
│   │   ├── 📁 config/             # Configuration modules
│   │   ├── 📁 controllers/        # API controllers
│   │   ├── 📁 middleware/         # Express middleware
│   │   ├── 📁 routes/             # API routes
│   │   ├── 📁 services/           # Business logic services
│   │   ├── 📁 types/              # TypeScript type definitions
│   │   ├── 📁 utils/              # Utility functions
│   │   └── 📄 server.ts           # Main server entry point
│   ├── 📁 tests/                  # Test files
│   ├── 📁 prisma/                 # Database schema and migrations
│   ├── 📁 scripts/                # Backend-specific scripts
│   ├── 📄 package.json            # Backend dependencies
│   ├── 📄 tsconfig.json           # TypeScript configuration
│   └── 📄 .eslintrc.js            # ESLint configuration
│
├── 📁 frontend/                   # Frontend Next.js application
│   ├── 📁 src/                    # Source code
│   │   ├── 📁 app/                # Next.js app directory
│   │   ├── 📁 components/         # React components
│   │   ├── 📁 hooks/              # Custom React hooks
│   │   ├── 📁 lib/                # Utility libraries
│   │   ├── 📁 services/           # API services
│   │   └── 📁 types/              # TypeScript types
│   ├── 📁 public/                 # Static assets
│   ├── 📄 package.json            # Frontend dependencies
│   ├── 📄 next.config.ts          # Next.js configuration
│   └── 📄 tailwind.config.js      # Tailwind CSS configuration
│
├── 📁 ml/                         # Machine Learning system
│   ├── 📁 src/                    # ML source code
│   │   ├── 📁 models/             # ML model implementations
│   │   ├── 📁 training/           # Training scripts
│   │   ├── 📁 api/                # ML API service
│   │   ├── 📁 backtesting/        # Backtesting engine
│   │   └── 📁 strategy/           # Trading strategies
│   ├── 📁 models/                 # Trained model files
│   ├── 📄 requirements.txt        # Python dependencies
│   └── 📄 README.md               # ML system documentation
│
├── 📁 data/                       # Project data files
│   ├── 📁 backtest/               # Backtest results
│   ├── 📁 trading/                # Trading data
│   ├── 📁 models/                 # Model registry
│   └── 📁 sample/                 # Sample data files
│
├── 📁 docs/                       # Documentation
│   ├── 📁 architecture/           # System architecture
│   ├── 📁 api/                    # API documentation
│   ├── 📁 deployment/             # Deployment guides
│   ├── 📁 development/            # Development guides
│   └── 📁 user/                   # User documentation
│
├── 📁 scripts/                    # Utility scripts
│   ├── 📁 setup/                  # Setup and installation
│   ├── 📁 deployment/             # Deployment scripts
│   ├── 📁 testing/                # Testing utilities
│   └── 📁 maintenance/            # Maintenance scripts
│
├── 📁 config/                     # Configuration files
│   ├── 📁 development/            # Development configs
│   ├── 📁 production/             # Production configs
│   └── 📁 docker/                 # Docker configurations
│
├── 📁 monitoring/                 # Monitoring configuration
│   ├── 📁 grafana/                # Grafana dashboards
│   └── 📁 prometheus/             # Prometheus configuration
│
├── 📁 .github/                    # GitHub workflows and templates
├── 📁 .cursor/                    # Cursor IDE configuration
├── 📁 tools/                      # Development tools
│
├── 📄 package.json                # Root package.json
├── 📄 docker-compose.yml          # Main Docker Compose file
├── 📄 .gitignore                  # Git ignore rules
├── 📄 README.md                   # Main project README
└── 📄 example.env                 # Environment variables template
```

## Key Directories Explained

### 📁 backend/
Contains the Express.js backend application with TypeScript, handling API requests, authentication, trading logic, and database operations.

**Key Files:**
- `src/server.ts` - Main server entry point
- `src/config/environment.ts` - Centralized environment configuration
- `src/utils/errorHandler.ts` - Centralized error handling
- `src/services/DeltaExchangeUnified.ts` - Delta Exchange integration

### 📁 frontend/
Next.js 15 frontend application with React 19, providing the trading dashboard and user interface.

**Key Files:**
- `src/app/layout.tsx` - Root layout component
- `src/components/dashboard/` - Trading dashboard components
- `src/hooks/` - Custom React hooks for trading functionality

### 📁 ml/
Python-based machine learning system for trade signal generation and market analysis.

**Key Files:**
- `src/api/app.py` - FastAPI ML service
- `src/models/` - ML model implementations
- `src/backtesting/` - Backtesting engine

### 📁 data/
Organized data storage for different types of project data.

**Subdirectories:**
- `backtest/` - Backtest results and analysis
- `trading/` - Real trading data and logs
- `models/` - Model registry and metadata
- `sample/` - Sample data for testing

### 📁 docs/
Comprehensive documentation organized by category.

**Subdirectories:**
- `architecture/` - System design and architecture
- `api/` - API documentation and references
- `deployment/` - Deployment and infrastructure guides
- `development/` - Development setup and guidelines
- `user/` - User guides and tutorials

### 📁 scripts/
Utility scripts organized by purpose.

**Subdirectories:**
- `setup/` - Installation and setup scripts
- `deployment/` - Deployment automation
- `testing/` - Testing utilities and validation
- `maintenance/` - System maintenance scripts

### 📁 config/
Configuration files organized by environment.

**Subdirectories:**
- `development/` - Development environment configs
- `production/` - Production environment configs
- `docker/` - Docker and containerization configs

## File Naming Conventions

### TypeScript/JavaScript Files
- Use kebab-case for file names: `api-key-controller.ts`
- Use PascalCase for class files: `DeltaExchangeService.ts`
- Use camelCase for utility files: `errorHandler.ts`

### Documentation Files
- Use UPPER_CASE for major documentation: `README.md`
- Use Title Case for guides: `Installation_Guide.md`
- Use descriptive names: `API_Reference.md`

### Configuration Files
- Use lowercase with hyphens: `docker-compose.yml`
- Use dots for environment-specific: `.env.development`

## Build Artifacts (Ignored)

The following directories are generated during build and are ignored by Git:

- `backend/dist/` - Compiled TypeScript output
- `frontend/.next/` - Next.js build output
- `node_modules/` - NPM dependencies
- `*.log` - Log files
- `.env` - Environment variables (use example.env as template)

## Navigation Tips

### Finding Files
- **API endpoints**: `backend/src/routes/`
- **React components**: `frontend/src/components/`
- **ML models**: `ml/src/models/`
- **Documentation**: `docs/`
- **Configuration**: `config/`

### Common Tasks
- **Add new API route**: Create in `backend/src/routes/`
- **Add new component**: Create in `frontend/src/components/`
- **Add new ML model**: Create in `ml/src/models/`
- **Update documentation**: Edit files in `docs/`
- **Configure environment**: Edit `example.env` and create `.env`

## Maintenance

### Regular Cleanup
- Remove old backtest results from `data/backtest/`
- Clean build artifacts: `npm run clean`
- Update dependencies: `npm update`
- Review and archive old logs

### Adding New Features
1. Create feature branch
2. Add code in appropriate directory
3. Add tests in corresponding test directory
4. Update documentation in `docs/`
5. Update this structure guide if needed

## Conclusion

This organized structure provides:
- **Clear separation of concerns**
- **Easy navigation and file discovery**
- **Consistent naming conventions**
- **Proper organization of different file types**
- **Scalable architecture for future growth**

The structure supports the complex nature of the SmartMarketOOPS trading platform while maintaining clarity and organization for development, deployment, and maintenance.