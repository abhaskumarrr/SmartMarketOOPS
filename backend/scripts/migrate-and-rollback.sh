#!/bin/bash
# Automated migration and rollback script for Prisma
# Usage: ./migrate-and-rollback.sh migrate|rollback

set -e

LOGFILE="migration.log"

if [ "$1" == "migrate" ]; then
  echo "[INFO] Running Prisma migrate deploy..." | tee -a $LOGFILE
  npx prisma migrate deploy | tee -a $LOGFILE
  echo "[INFO] Migration completed at $(date)" | tee -a $LOGFILE
elif [ "$1" == "rollback" ]; then
  echo "[INFO] Rolling back last Prisma migration..." | tee -a $LOGFILE
  npx prisma migrate resolve --applied "$(npx prisma migrate status --json | jq -r '.migrations[-2].name')" | tee -a $LOGFILE
  echo "[INFO] Rollback completed at $(date)" | tee -a $LOGFILE
else
  echo "Usage: $0 migrate|rollback" | tee -a $LOGFILE
  exit 1
fi 