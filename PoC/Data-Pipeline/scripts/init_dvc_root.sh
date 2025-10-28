#!/bin/bash
# Initialize DVC in Data-Pipeline subdirectory with existing Git repo
# AND track data files with DVC
# Run this on your HOST machine (not in container)

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_PIPELINE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
GIT_ROOT="$(cd "$DATA_PIPELINE_ROOT/../.." && pwd)"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}DVC Initialization with Git${NC}"
echo -e "${BLUE}========================================${NC}"
echo -e "${YELLOW}Git Root:           ${GIT_ROOT}${NC}"
echo -e "${YELLOW}Data-Pipeline Root: ${DATA_PIPELINE_ROOT}${NC}"
echo ""

# Verify we're in the right location
if [ ! -d "$GIT_ROOT/.git" ]; then
    echo -e "${RED}ERROR: Git repository not found at ${GIT_ROOT}${NC}"
    echo -e "${RED}Expected: himas-mlops/.git${NC}"
    exit 1
fi

if [ ! -f "$DATA_PIPELINE_ROOT/docker-compose.yaml" ]; then
    echo -e "${RED}ERROR: docker-compose.yaml not found at ${DATA_PIPELINE_ROOT}${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Git repository found: ${GIT_ROOT}${NC}"
echo -e "${GREEN}✓ Data-Pipeline found: ${DATA_PIPELINE_ROOT}${NC}"
echo ""

# Check Git remote
echo -e "${BLUE}Git Configuration:${NC}"
cd "$GIT_ROOT"
GIT_REMOTE=$(git config --get remote.origin.url || echo "No remote configured")
echo -e "Remote: ${GIT_REMOTE}"
GIT_BRANCH=$(git branch --show-current || echo "main")
echo -e "Branch: ${GIT_BRANCH}"
echo ""

cd "$DATA_PIPELINE_ROOT"

# Create .gitignore if it doesn't exist
if [ ! -f ".gitignore" ]; then
    echo -e "${YELLOW}Creating .gitignore...${NC}"
    cat > .gitignore << 'EOF'
# DVC - track .dvc files but not the actual data
/dags/data/bigquery
/dags/data/reports
/dags/data/processed
/dags/data/raw
/dags/data/models

# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python

# Airflow
logs/
airflow.db
*.pid
*.log

# Environment
.env

# OS
.DS_Store
Thumbs.db

# IDE
.vscode/
.idea/
*.swp
EOF
    echo -e "${GREEN}✓ .gitignore created${NC}"
fi

# Check if DVC is already initialized
if [ -d ".dvc" ]; then
    echo -e "${GREEN}✓ DVC already initialized${NC}"
else
    echo -e "${YELLOW}Initializing DVC with --subdir flag...${NC}"
    dvc init --subdir
    echo -e "${GREEN}✓ DVC initialized${NC}"
fi

# Get USE_GCS flag from environment or .env file
if [ -f ".env" ]; then
    # Source .env to get variables into current shell
    set -a
    source .env
    set +a
fi

USE_GCS=${USE_GCS:-False}
GCS_BUCKET=${GCS_BUCKET:-}
PROJECT_ID=${GOOGLE_CLOUD_PROJECT:-}

echo ""
echo -e "${BLUE}DVC Remote Configuration${NC}"
echo -e "USE_GCS: ${USE_GCS}"
echo -e "GCS_BUCKET: ${GCS_BUCKET}"
echo -e "PROJECT_ID: ${PROJECT_ID}"
echo ""

# Configure DVC remote
if [ "$USE_GCS" = "True" ] || [ "$USE_GCS" = "true" ]; then
    echo -e "${YELLOW}Configuring GCS remote...${NC}"
    
    if [ -z "$PROJECT_ID" ]; then
        echo -e "${RED}ERROR: GOOGLE_CLOUD_PROJECT not set in .env${NC}"
        exit 1
    fi
    
    # Remove existing remotes
    dvc remote remove local_storage 2>/dev/null || true
    dvc remote remove gcs_storage 2>/dev/null || true
    
    # Add GCS remote
    GCS_URL="gs://${GCS_BUCKET}/dvc-storage"
    dvc remote add -d gcs_storage "$GCS_URL"
    dvc remote modify gcs_storage projectname "$PROJECT_ID"
    
    echo -e "${GREEN}✓ GCS remote configured: ${GCS_URL}${NC}"
else
    echo -e "${YELLOW}Configuring local remote...${NC}"
    
    # Create local storage directory at Data-Pipeline level
    mkdir -p .dvc/cache
    
    # Remove existing remotes
    dvc remote remove local_storage 2>/dev/null || true
    dvc remote remove gcs_storage 2>/dev/null || true
    
    # Add local remote
    dvc remote add -d local_storage .dvc/cache
    
    echo -e "${GREEN}✓ Local remote configured: .dvc/cache${NC}"
fi

# Set DVC config
dvc config core.autostage true
echo -e "${GREEN}✓ DVC autostage enabled${NC}"

# Create data directories if they don't exist
echo ""
echo -e "${YELLOW}Ensuring data directories exist...${NC}"
mkdir -p dags/data/bigquery/{curated,federated,verification}
mkdir -p dags/data/reports
mkdir -p dags/data/processed
mkdir -p dags/data/raw
mkdir -p dags/data/models
mkdir -p dags/data/metadata
echo -e "${GREEN}✓ Data directories ready${NC}"

# Track existing data files with DVC
echo ""
echo -e "${YELLOW}Tracking existing data files with DVC...${NC}"

# Function to get directory size
get_dir_size() {
    if [ -d "$1" ]; then
        du -sh "$1" 2>/dev/null | cut -f1
    else
        echo "N/A"
    fi
}

# Function to count files in directory
count_files() {
    if [ -d "$1" ]; then
        find "$1" -type f | wc -l | tr -d ' '
    else
        echo "0"
    fi
}

DATA_DIRS=("dags/data/bigquery" "dags/data/reports" "dags/data/processed" "dags/data/raw" "dags/data/models")

for data_dir in "${DATA_DIRS[@]}"; do
    FILE_COUNT=$(count_files "$data_dir")
    DIR_SIZE=$(get_dir_size "$data_dir")
    
    if [ -d "$data_dir" ] && [ "$FILE_COUNT" -gt 0 ]; then
        echo -e "${YELLOW}  ${data_dir}${NC}"
        echo -e "    Files: ${FILE_COUNT}, Size: ${DIR_SIZE}"
        echo -e "    Adding to DVC..."
        
        if dvc add "$data_dir"; then
            echo -e "${GREEN}    ✓ ${data_dir}.dvc created${NC}"
        else
            echo -e "${RED}    ✗ Failed to add ${data_dir}${NC}"
        fi
    else
        echo -e "${BLUE}  ⊘ ${data_dir} is empty or doesn't exist (${FILE_COUNT} files)${NC}"
    fi
done

# Show created .dvc files
echo ""
echo -e "${BLUE}Created DVC tracking files:${NC}"
ls -lh dags/data/*.dvc 2>/dev/null || echo "No .dvc files created"

# Push data to remote if USE_GCS=True
if [ "$USE_GCS" = "True" ] || [ "$USE_GCS" = "true" ]; then
    echo ""
    echo -e "${YELLOW}Pushing data to GCS remote...${NC}"
    echo -e "${BLUE}This may take a while depending on data size...${NC}"
    
    # Check for GCP credentials
    echo -e "${YELLOW}Checking GCP authentication...${NC}"
    
    # Try to find credentials file
    CREDS_FILE="$HOME/.config/gcloud/application_default_credentials.json"
    
    if [ -f "$CREDS_FILE" ]; then
        echo -e "${GREEN}✓ Found credentials: ${CREDS_FILE}${NC}"
        export GOOGLE_APPLICATION_CREDENTIALS="$CREDS_FILE"
    else
        echo -e "${YELLOW}⚠ No application default credentials found${NC}"
        echo -e "${YELLOW}Attempting to use gcloud auth...${NC}"
    fi
    
    # Test GCS access
    if gsutil ls "gs://${GCS_BUCKET}/" >/dev/null 2>&1; then
        echo -e "${GREEN}✓ GCS bucket accessible${NC}"
    else
        echo -e "${RED}✗ Cannot access GCS bucket${NC}"
        echo -e "${YELLOW}Please authenticate first:${NC}"
        echo "  gcloud auth application-default login"
        echo "  gcloud config set project ${PROJECT_ID}"
        echo ""
        echo -e "${YELLOW}Skipping GCS push for now...${NC}"
        echo -e "${YELLOW}You can push later with: dvc push${NC}"
        USE_GCS="False"  # Override to skip push
    fi
    
    # Only push if authentication worked
    if [ "$USE_GCS" = "True" ]; then
        # Export credentials for DVC
        export GOOGLE_APPLICATION_CREDENTIALS="$CREDS_FILE"
        
        if dvc push; then
            echo -e "${GREEN}✓ Data pushed to GCS successfully${NC}"
            echo -e "${GREEN}  Location: gs://${GCS_BUCKET}/dvc-storage/${NC}"
        else
            echo -e "${YELLOW}⚠ Push failed${NC}"
            echo -e "${YELLOW}You can retry later with: dvc push${NC}"
        fi
    fi
else
    echo ""
    echo -e "${BLUE}Skipping remote push - USE_GCS=False${NC}"
    echo -e "${BLUE}Data stored locally in: .dvc/cache/${NC}"
    
    # Show local cache size
    if [ -d ".dvc/cache" ]; then
        CACHE_SIZE=$(get_dir_size ".dvc/cache")
        echo -e "${BLUE}Local cache size: ${CACHE_SIZE}${NC}"
    fi
fi

# Commit to Git
echo ""
echo -e "${YELLOW}Committing DVC setup to Git...${NC}"
cd "$GIT_ROOT"

# Stage all DVC-related files
git add PoC/Data-Pipeline/.gitignore 2>/dev/null || true
git add PoC/Data-Pipeline/.dvc/config 2>/dev/null || true
git add PoC/Data-Pipeline/.dvc/.gitignore 2>/dev/null || true
git add PoC/Data-Pipeline/dags/data/*.dvc 2>/dev/null || true

# Check if there are changes to commit
if git diff --cached --quiet; then
    echo -e "${BLUE}No changes to commit${NC}"
else
    if git commit -m "Setup DVC for Data-Pipeline with --subdir mode and track data files"; then
        echo -e "${GREEN}✓ Changes committed to Git${NC}"
        echo ""
        echo -e "${YELLOW}Ready to push to GitHub!${NC}"
        echo "Run: git push origin ${GIT_BRANCH}"
    else
        echo -e "${YELLOW}⚠ Commit failed (may already be committed)${NC}"
    fi
fi

cd "$DATA_PIPELINE_ROOT"

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}DVC Setup Complete!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo -e "${BLUE}Configuration Summary:${NC}"
echo "─────────────────────────────────────────"
echo "Git Repository:  ${GIT_REMOTE}"
echo "Git Branch:      ${GIT_BRANCH}"
echo "Git Root:        ${GIT_ROOT}"
echo "DVC Location:    ${DATA_PIPELINE_ROOT}/.dvc"
echo "DVC Mode:        --subdir (within existing Git repo)"
echo "─────────────────────────────────────────"
echo ""
echo -e "${BLUE}DVC Remote Configuration:${NC}"
dvc remote list
echo ""
echo -e "${BLUE}Tracked Data Files (.dvc files):${NC}"
for dvc_file in dags/data/*.dvc; do
    if [ -f "$dvc_file" ]; then
        SIZE=$(grep -A 1 "size:" "$dvc_file" | grep "size:" | awk '{print $2}')
        echo "  ✓ $(basename $dvc_file) (size: ${SIZE:-unknown} bytes)"
    fi
done
echo ""
echo -e "${BLUE}Data Storage Summary:${NC}"
echo "Git tracks:      .dvc files (metadata), code, SQL, configs"
echo "DVC tracks:      Actual data files (CSV, JSON, models)"
if [ "$USE_GCS" = "True" ] || [ "$USE_GCS" = "true" ]; then
    echo "Data location:   gs://${GCS_BUCKET}/dvc-storage/"
    echo "Metadata:        gs://${GCS_BUCKET}/metadata/"
else
    echo "Data location:   ${DATA_PIPELINE_ROOT}/.dvc/cache/"
    echo "Metadata:        ${DATA_PIPELINE_ROOT}/dags/data/metadata/"
fi
echo ""
echo -e "${YELLOW}Next steps:${NC}"
echo "1. If GCS push failed, authenticate and retry:"
echo "   gcloud auth application-default login"
echo "   cd $DATA_PIPELINE_ROOT && dvc push"
echo ""
echo "2. Push to GitHub:"
echo "   cd $GIT_ROOT && git push origin ${GIT_BRANCH}"
echo ""
echo "3. Start Airflow:"
echo "   cd $DATA_PIPELINE_ROOT && docker compose up -d"
echo ""
echo "4. Verify DVC in container:"
echo "   docker compose exec airflow-worker ls -la /opt/airflow/.dvc"
echo "   docker compose exec airflow-worker bash -c 'cd /opt/airflow && dvc status'"
echo ""
echo "5. Run DAG:"
echo "   docker compose exec airflow-worker airflow dags trigger himas_bigquery_demo_dvc"
echo ""
echo "6. Team members can clone and pull data:"
echo "   git clone ${GIT_REMOTE}"
echo "   cd himas-mlops/PoC/Data-Pipeline"
echo "   gcloud auth application-default login"
echo "   dvc pull"
echo ""