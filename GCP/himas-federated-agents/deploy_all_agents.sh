#!/bin/bash
# =============================================================================
# HIMAS Multi-Agent Deployment Script
# Deploys all 5 agents to Google Cloud Run
# =============================================================================

set -e

# =============================================================================
# CONFIGURATION
# =============================================================================

export GOOGLE_CLOUD_PROJECT="${GOOGLE_CLOUD_PROJECT:-erudite-carving-472018-r5}"
export GOOGLE_CLOUD_LOCATION="${GOOGLE_CLOUD_LOCATION:-us-central1}"

# Base path to agents (adjust as needed)
AGENTS_BASE_PATH="${AGENTS_BASE_PATH:-./himas-federated-agents}"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

echo ""
echo -e "${BLUE}╔═══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║          HIMAS Multi-Agent Deployment System                  ║${NC}"
echo -e "${BLUE}╚═══════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "Project:  ${GREEN}$GOOGLE_CLOUD_PROJECT${NC}"
echo -e "Region:   ${GREEN}$GOOGLE_CLOUD_LOCATION${NC}"
echo ""

# =============================================================================
# 1. HOSPITAL A - Community Hospital (with UI)
# =============================================================================

echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${YELLOW}[1/5] Deploying Hospital A Agent (Community Hospital)${NC}"
echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

adk deploy cloud_run \
    --project=$GOOGLE_CLOUD_PROJECT \
    --region=$GOOGLE_CLOUD_LOCATION \
    --service_name=hospital-a \
    --app_name=hospital_a_agent \
    --port=8001 \
    --with_ui \
    ${AGENTS_BASE_PATH}/agent_hospital_a

echo -e "${GREEN}✓ Hospital A deployed${NC}"
echo ""

# =============================================================================
# 2. HOSPITAL B - Tertiary Care Center (with UI)
# =============================================================================

echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${YELLOW}[2/5] Deploying Hospital B Agent (Tertiary Care Center)${NC}"
echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

adk deploy cloud_run \
    --project=$GOOGLE_CLOUD_PROJECT \
    --region=$GOOGLE_CLOUD_LOCATION \
    --service_name=hospital-b \
    --app_name=hospital_b_agent \
    --port=8002 \
    --with_ui \
    ${AGENTS_BASE_PATH}/agent_hospital_b

echo -e "${GREEN}✓ Hospital B deployed${NC}"
echo ""

# =============================================================================
# 3. HOSPITAL C - Rural Hospital (with UI)
# =============================================================================

echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${YELLOW}[3/5] Deploying Hospital C Agent (Rural Hospital)${NC}"
echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

adk deploy cloud_run \
    --project=$GOOGLE_CLOUD_PROJECT \
    --region=$GOOGLE_CLOUD_LOCATION \
    --service_name=hospital-c \
    --app_name=hospital_c_agent \
    --port=8003 \
    --with_ui \
    ${AGENTS_BASE_PATH}/agent_hospital_c

echo -e "${GREEN}✓ Hospital C deployed${NC}"
echo ""

# =============================================================================
# 4. FEDERATED COORDINATOR - With UI
# =============================================================================

echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${YELLOW}[4/5] Deploying Federated Coordinator (with UI)${NC}"
echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

adk deploy cloud_run \
    --project=$GOOGLE_CLOUD_PROJECT \
    --region=$GOOGLE_CLOUD_LOCATION \
    --service_name=federated-coordinator-ui \
    --app_name=federated_coordinator_ui \
    --port=8000 \
    --with_ui \
    ${AGENTS_BASE_PATH}/federated_coordinator

echo -e "${GREEN}✓ Federated Coordinator (UI) deployed${NC}"
echo ""

# =============================================================================
# 5. FEDERATED COORDINATOR - A2A Only (No UI)
# =============================================================================

echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${YELLOW}[5/5] Deploying Federated Coordinator (A2A - No UI)${NC}"
echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

cd ${AGENTS_BASE_PATH}/federated_coordinator

gcloud run deploy federated-coordinator-a2a \
    --port=8080 \
    --source="." \
    --allow-unauthenticated \
    --region=$GOOGLE_CLOUD_LOCATION \
    --project=$GOOGLE_CLOUD_PROJECT \
    --memory=2Gi \
    --cpu=1 \
    --timeout=300 \
    --set-env-vars="GOOGLE_CLOUD_PROJECT=$GOOGLE_CLOUD_PROJECT"

cd - > /dev/null

echo -e "${GREEN}✓ Federated Coordinator (A2A) deployed${NC}"
echo ""

# =============================================================================
# DEPLOYMENT SUMMARY
# =============================================================================

echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}                    DEPLOYMENT COMPLETE                        ${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
echo ""

# Get service URLs
echo -e "${GREEN}Service URLs:${NC}"
echo "─────────────────────────────────────────────────────────────────"

for service in hospital-a hospital-b hospital-c federated-coordinator-ui federated-coordinator-a2a; do
    URL=$(gcloud run services describe $service \
        --region=$GOOGLE_CLOUD_LOCATION \
        --project=$GOOGLE_CLOUD_PROJECT \
        --format="value(status.url)" 2>/dev/null || echo "Not deployed")
    printf "%-30s %s\n" "$service:" "$URL"
done

echo ""
echo -e "${GREEN}A2A Agent Cards:${NC}"
echo "─────────────────────────────────────────────────────────────────"

for service in hospital-a hospital-b hospital-c federated-coordinator-a2a; do
    URL=$(gcloud run services describe $service \
        --region=$GOOGLE_CLOUD_LOCATION \
        --project=$GOOGLE_CLOUD_PROJECT \
        --format="value(status.url)" 2>/dev/null)
    if [ -n "$URL" ]; then
        echo "$URL/.well-known/agent-card.json"
    fi
done

echo ""
echo -e "${YELLOW}Next Steps:${NC}"
echo "1. Test health endpoints: curl <URL>/health"
echo "2. Access UI agents in browser"
echo "3. Test A2A communication between agents"
echo ""