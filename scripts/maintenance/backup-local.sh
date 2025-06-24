#!/bin/bash

# SmartMarketOOPS Local Backup Script
# Creates backups of database and important data

set -e

echo "💾 Creating backup of SmartMarketOOPS data..."

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Create backup directory with timestamp
BACKUP_DIR="./backups/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"

print_status "Backup directory: $BACKUP_DIR"

# Load environment variables
if [ -f ".env.local" ]; then
    export $(cat .env.local | grep -v '^#' | xargs)
fi

# Backup database
print_status "Backing up PostgreSQL database..."
if docker-compose -f docker-compose.local.yml ps postgres | grep -q "Up"; then
    docker-compose -f docker-compose.local.yml exec -T postgres pg_dump -U smartmarket smartmarket > "$BACKUP_DIR/database.sql"
    print_success "Database backup completed"
else
    print_warning "PostgreSQL container not running, skipping database backup"
fi

# Backup configuration files
print_status "Backing up configuration files..."
cp .env.local "$BACKUP_DIR/" 2>/dev/null || print_warning ".env.local not found"
cp docker-compose.local.yml "$BACKUP_DIR/"
cp -r monitoring/grafana/dashboards "$BACKUP_DIR/" 2>/dev/null || true

# Backup logs
print_status "Backing up logs..."
mkdir -p "$BACKUP_DIR/logs"
cp -r logs/* "$BACKUP_DIR/logs/" 2>/dev/null || true
cp -r backend/logs/* "$BACKUP_DIR/logs/" 2>/dev/null || true
cp -r ml/logs/* "$BACKUP_DIR/logs/" 2>/dev/null || true

# Backup ML models (if any custom trained models)
print_status "Backing up ML models..."
mkdir -p "$BACKUP_DIR/models"
cp -r models/* "$BACKUP_DIR/models/" 2>/dev/null || true
cp -r ml/models/* "$BACKUP_DIR/models/" 2>/dev/null || true

# Create backup info file
cat > "$BACKUP_DIR/backup_info.txt" << EOF
SmartMarketOOPS Backup Information
==================================
Backup Date: $(date)
System: MacBook Air M2
Version: $(git rev-parse HEAD 2>/dev/null || echo "unknown")
Environment: Local Development

Contents:
- database.sql: PostgreSQL database dump
- .env.local: Environment configuration
- docker-compose.local.yml: Docker configuration
- dashboards/: Grafana dashboards
- logs/: Application logs
- models/: ML model files

Restore Instructions:
1. Start the system: ./start-local.sh
2. Restore database: docker-compose -f docker-compose.local.yml exec -T postgres psql -U smartmarket smartmarket < database.sql
3. Restart services: ./restart-local.sh
EOF

# Compress backup
print_status "Compressing backup..."
cd backups
tar -czf "$(basename $BACKUP_DIR).tar.gz" "$(basename $BACKUP_DIR)"
rm -rf "$(basename $BACKUP_DIR)"
cd ..

# Cleanup old backups (keep last 7 days)
print_status "Cleaning up old backups..."
find ./backups -name "*.tar.gz" -mtime +7 -delete 2>/dev/null || true

BACKUP_SIZE=$(du -h "backups/$(basename $BACKUP_DIR).tar.gz" | cut -f1)
print_success "✅ Backup completed successfully!"
echo ""
echo "📦 Backup file: backups/$(basename $BACKUP_DIR).tar.gz"
echo "📏 Size: $BACKUP_SIZE"
echo ""
echo "To restore from this backup:"
echo "1. Extract: tar -xzf backups/$(basename $BACKUP_DIR).tar.gz -C backups/"
echo "2. Follow instructions in backup_info.txt"