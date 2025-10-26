#!/bin/bash
# HIMAS Pipeline Simple Cleanup Script
# Cleans up logs, DVC cache, and data folders

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Get the Data-Pipeline directory (parent of scripts folder)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}HIMAS Pipeline Simple Cleanup${NC}"
echo -e "${BLUE}========================================${NC}"
echo -e "${YELLOW}Project Root: ${PROJECT_ROOT}${NC}"
echo ""

# Function to get directory size
get_size() {
    if [ -d "$1" ]; then
        du -sh "$1" 2>/dev/null | cut -f1
    else
        echo "N/A"
    fi
}

# Define paths
LOGS_DIR="${PROJECT_ROOT}/logs"
DVC_DIR="${PROJECT_ROOT}/.dvc"
DVC_CACHE="${PROJECT_ROOT}/.dvc/cache"
DATA_DIR="${PROJECT_ROOT}/dags/data"

# Show sizes BEFORE cleanup
echo -e "${YELLOW}Current Sizes:${NC}"
echo "─────────────────────────────────────────"
echo "logs/               $(get_size "${LOGS_DIR}")"
echo ".dvc/cache/         $(get_size "${DVC_CACHE}")"
echo "dags/data/          $(get_size "${DATA_DIR}")"
echo "  ├─ bigquery/      $(get_size "${DATA_DIR}/bigquery")"
echo "  ├─ reports/       $(get_size "${DATA_DIR}/reports")"
echo "  ├─ processed/     $(get_size "${DATA_DIR}/processed")"
echo "  ├─ raw/           $(get_size "${DATA_DIR}/raw")"
echo "  ├─ models/        $(get_size "${DATA_DIR}/models")"
echo "  └─ metadata/      $(get_size "${DATA_DIR}/metadata")"
echo "─────────────────────────────────────────"
TOTAL_BEFORE=$(du -sh "${PROJECT_ROOT}" 2>/dev/null | cut -f1)
echo "Total Project:      ${TOTAL_BEFORE}"
echo ""

# Check if directories exist
echo -e "${BLUE}Directory Status:${NC}"
[ -d "${LOGS_DIR}" ] && echo "✓ logs/ exists" || echo "✗ logs/ not found"
[ -d "${DVC_CACHE}" ] && echo "✓ .dvc/cache/ exists" || echo "✗ .dvc/cache/ not found"
[ -d "${DATA_DIR}" ] && echo "✓ dags/data/ exists" || echo "✗ dags/data/ not found"
echo ""

# Confirmation
read -p "$(echo -e ${RED}Delete all logs, DVC cache, and data? ${NC}[y/N]: )" -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${BLUE}Cleanup cancelled.${NC}"
    exit 0
fi

echo ""
echo -e "${GREEN}Cleaning...${NC}"
echo ""

# Clean logs
echo -e "${YELLOW}[1/3] Cleaning logs...${NC}"
if [ -d "${LOGS_DIR}" ]; then
    rm -rf "${LOGS_DIR}"/*
    echo -e "${GREEN}✓ Cleaned: logs/${NC}"
else
    echo -e "${BLUE}  Skipped: logs/ not found${NC}"
fi

# Clean .dvc cache (keep config)
echo -e "${YELLOW}[2/3] Cleaning .dvc cache...${NC}"
if [ -d "${DVC_CACHE}" ]; then
    rm -rf "${DVC_CACHE}"/*
    echo -e "${GREEN}✓ Cleaned: .dvc/cache/ (kept .dvc/config)${NC}"
else
    echo -e "${BLUE}  Skipped: .dvc/cache/ not found${NC}"
fi

# Clean .dvc/tmp if exists
if [ -d "${DVC_DIR}/tmp" ]; then
    rm -rf "${DVC_DIR}/tmp"/*
    echo -e "${GREEN}✓ Cleaned: .dvc/tmp/${NC}"
fi

# Clean data directory
echo -e "${YELLOW}[3/3] Cleaning data directory...${NC}"
if [ -d "${DATA_DIR}" ]; then
    # Clean subdirectories but keep the structure
    rm -rf "${DATA_DIR}/bigquery"/* 2>/dev/null || true
    rm -rf "${DATA_DIR}/reports"/* 2>/dev/null || true
    rm -rf "${DATA_DIR}/processed"/* 2>/dev/null || true
    rm -rf "${DATA_DIR}/raw"/* 2>/dev/null || true
    rm -rf "${DATA_DIR}/models"/* 2>/dev/null || true
    
    # Keep metadata directory but clean old files (>7 days)
    if [ -d "${DATA_DIR}/metadata" ]; then
        find "${DATA_DIR}/metadata" -name "version_metadata_*.json" -mtime +7 -delete 2>/dev/null || true
    fi
    
    # Remove .dvc tracking files
    find "${DATA_DIR}" -name "*.dvc" -delete 2>/dev/null || true
    
    echo -e "${GREEN}✓ Cleaned: dags/data/${NC}"
else
    echo -e "${BLUE}  Skipped: dags/data/ not found${NC}"
fi

# Clean Python cache
echo -e "${YELLOW}[Extra] Cleaning Python cache...${NC}"
find "${PROJECT_ROOT}/dags" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find "${PROJECT_ROOT}/dags" -type f -name "*.pyc" -delete 2>/dev/null || true
echo -e "${GREEN}✓ Cleaned: Python cache${NC}"

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Cleanup Complete${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# Show sizes AFTER cleanup
echo -e "${YELLOW}Sizes After Cleanup:${NC}"
echo "─────────────────────────────────────────"
echo "logs/               $(get_size "${LOGS_DIR}")"
echo ".dvc/cache/         $(get_size "${DVC_CACHE}")"
echo "dags/data/          $(get_size "${DATA_DIR}")"
echo "  ├─ bigquery/      $(get_size "${DATA_DIR}/bigquery")"
echo "  ├─ reports/       $(get_size "${DATA_DIR}/reports")"
echo "  ├─ processed/     $(get_size "${DATA_DIR}/processed")"
echo "  ├─ raw/           $(get_size "${DATA_DIR}/raw")"
echo "  ├─ models/        $(get_size "${DATA_DIR}/models")"
echo "  └─ metadata/      $(get_size "${DATA_DIR}/metadata")"
echo "─────────────────────────────────────────"
TOTAL_AFTER=$(du -sh "${PROJECT_ROOT}" 2>/dev/null | cut -f1)
echo "Total Project:      ${TOTAL_AFTER}"
echo ""
echo -e "${BLUE}Before: ${TOTAL_BEFORE}  →  After: ${TOTAL_AFTER}${NC}"
echo ""

echo -e "${YELLOW}To regenerate data, run:${NC}"
echo "docker compose exec airflow-worker airflow dags trigger himas_bigquery_demo_dvc"
echo ""
echo -e "${GREEN}Done!${NC}"