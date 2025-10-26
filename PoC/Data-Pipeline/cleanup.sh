#!/bin/bash
# HIMAS Pipeline Simple Cleanup Script
# Cleans up logs, DVC, and data folders

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Get the actual project root (where the script is located)
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

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
DVC_DIR="${PROJECT_ROOT}/dags/.dvc"
DVC_STORAGE="${PROJECT_ROOT}/dags/.dvc_storage"
DATA_DIR="${PROJECT_ROOT}/dags/data"

# Show sizes BEFORE cleanup
echo -e "${YELLOW}Current Sizes:${NC}"
echo "─────────────────────────────────────────"
echo "logs/               $(get_size "${LOGS_DIR}")"
echo "dags/.dvc/          $(get_size "${DVC_DIR}")"
echo "dags/.dvc_storage/  $(get_size "${DVC_STORAGE}")"
echo "dags/data/          $(get_size "${DATA_DIR}")"
echo "─────────────────────────────────────────"
TOTAL_BEFORE=$(du -sh "${PROJECT_ROOT}" 2>/dev/null | cut -f1)
echo "Total Project:      ${TOTAL_BEFORE}"
echo ""

# Check if directories exist
echo -e "${BLUE}Directory Status:${NC}"
[ -d "${LOGS_DIR}" ] && echo "✓ logs/ exists" || echo "✗ logs/ not found"
[ -d "${DVC_DIR}" ] && echo "✓ dags/.dvc/ exists" || echo "✗ dags/.dvc/ not found"
[ -d "${DVC_STORAGE}" ] && echo "✓ dags/.dvc_storage/ exists" || echo "✗ dags/.dvc_storage/ not found"
[ -d "${DATA_DIR}" ] && echo "✓ dags/data/ exists" || echo "✗ dags/data/ not found"
echo ""

# Confirmation
read -p "$(echo -e ${RED}Delete all logs, DVC, and data? ${NC}[y/N]: )" -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${BLUE}Cleanup cancelled.${NC}"
    exit 0
fi

echo ""
echo -e "${GREEN}Cleaning...${NC}"
echo ""

# Clean logs
echo -e "${YELLOW}[1/4] Cleaning logs...${NC}"
if [ -d "${LOGS_DIR}" ]; then
    rm -rf "${LOGS_DIR}"/*
    echo -e "${GREEN}✓ Cleaned: logs/${NC}"
else
    echo -e "${BLUE}  Skipped: logs/ not found${NC}"
fi

# Clean .dvc directory (except config)
echo -e "${YELLOW}[2/4] Cleaning .dvc directory...${NC}"
if [ -d "${DVC_DIR}" ]; then
    find "${DVC_DIR}" -mindepth 1 ! -name 'config' -exec rm -rf {} + 2>/dev/null || true
    echo -e "${GREEN}✓ Cleaned: dags/.dvc/ (kept config)${NC}"
else
    echo -e "${BLUE}  Skipped: dags/.dvc/ not found${NC}"
fi

# Clean .dvc_storage
echo -e "${YELLOW}[3/4] Cleaning .dvc_storage...${NC}"
if [ -d "${DVC_STORAGE}" ]; then
    rm -rf "${DVC_STORAGE}"/*
    echo -e "${GREEN}✓ Cleaned: dags/.dvc_storage/${NC}"
else
    echo -e "${BLUE}  Skipped: dags/.dvc_storage/ not found${NC}"
fi

# Clean data directory
echo -e "${YELLOW}[4/4] Cleaning data directory...${NC}"
if [ -d "${DATA_DIR}" ]; then
    rm -rf "${DATA_DIR}"/*
    echo -e "${GREEN}✓ Cleaned: dags/data/${NC}"
else
    echo -e "${BLUE}  Skipped: dags/data/ not found${NC}"
fi

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Cleanup Complete${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# Show sizes AFTER cleanup
echo -e "${YELLOW}Sizes After Cleanup:${NC}"
echo "─────────────────────────────────────────"
echo "logs/               $(get_size "${LOGS_DIR}")"
echo "dags/.dvc/          $(get_size "${DVC_DIR}")"
echo "dags/.dvc_storage/  $(get_size "${DVC_STORAGE}")"
echo "dags/data/          $(get_size "${DATA_DIR}")"
echo "─────────────────────────────────────────"
TOTAL_AFTER=$(du -sh "${PROJECT_ROOT}" 2>/dev/null | cut -f1)
echo "Total Project:      ${TOTAL_AFTER}"
echo ""
echo -e "${BLUE}Before: ${TOTAL_BEFORE}  →  After: ${TOTAL_AFTER}${NC}"
echo ""

echo -e "${YELLOW}To regenerate data, run:${NC}"
echo "docker compose exec airflow-worker airflow dags trigger himas_bigquery_demo_dvc"
echo ""
echo -e "${GREEN}Done! ${NC}"