# SAIA-RAG Server Deployment Guide

## 🏗️ Server Architecture (134.209.10.163)

### Multi-Project Setup
The server hosts multiple SAIA projects with subdomain routing:
- **demo-salesagent.bineyes.com** → Port 8000 (Django app)
- **demo-law.bineyes.com** → Port 8001 (SAIA-RAG FastAPI)

### Nginx Reverse Proxy
- SSL termination via Let's Encrypt
- Routes by subdomain to different ports
- Config: `/etc/nginx/sites-available/demo-law`

---

## 📁 Directory Structure (MANDATORY)

### `/opt/saia/` - Git Repository (Source of Truth)
```
/opt/saia/
├── app/                    # Application code
├── docker-compose.dev.yml  # Development config
├── Dockerfile              # Container build
├── requirements.txt        # Python dependencies
└── .env.prod              # Production environment variables
```

**Purpose**: Git repository for SAIA-RAG project
**Branch**: `feature/rag-conversation-context-v2`
**Remote**: `https://github.com/osamanoman/SAIA-RAG.git`

### `/opt/saia-rag/` - DEPRECATED (To Be Removed)
**Status**: Old deployment directory, no longer used
**Action**: Will be removed after migration

---

## 🚀 Deployment Process (SIMPLE)

### Step 1: Update Code
```bash
ssh root@134.209.10.163
cd /opt/saia
git pull origin feature/rag-conversation-context-v2
```

### Step 2: Rebuild Container
```bash
cd /opt/saia
docker build -t saia-law-api .
```

### Step 3: Stop Old Container (if running)
```bash
docker stop saia-law-api 2>/dev/null || true
docker rm saia-law-api 2>/dev/null || true
```

### Step 4: Start New Container
```bash
docker run -d \
  --name saia-law-api \
  --network saia-law-network \
  --link saia-law-qdrant:qdrant \
  -p 8001:8000 \
  --env-file /opt/saia/.env.prod \
  --restart unless-stopped \
  saia-law-api
```

### Step 5: Verify Deployment
```bash
# Check container is running
docker ps | grep saia-law-api

# Check logs
docker logs saia-law-api --tail=50

# Test health endpoint
curl https://demo-law.bineyes.com/health
```

---

## 🔧 Docker Network Setup

### Network: `saia-law-network`
```bash
# Create network if it doesn't exist
docker network create saia-law-network

# Qdrant container (if not running)
docker run -d \
  --name saia-law-qdrant \
  --network saia-law-network \
  -v saia-law-qdrant-data:/qdrant/storage \
  --restart unless-stopped \
  qdrant/qdrant:v1.12.1
```

---

## 📝 Quick Deployment Script

Create `/opt/saia/deploy-production.sh`:
```bash
#!/bin/bash
set -e

echo "🚀 Deploying SAIA-RAG to production..."

# Step 1: Pull latest code
cd /opt/saia
git pull origin feature/rag-conversation-context-v2

# Step 2: Rebuild image
docker build -t saia-law-api .

# Step 3: Stop old container
docker stop saia-law-api 2>/dev/null || true
docker rm saia-law-api 2>/dev/null || true

# Step 4: Start new container
docker run -d \
  --name saia-law-api \
  --network saia-law-network \
  --link saia-law-qdrant:qdrant \
  -p 8001:8000 \
  --env-file /opt/saia/.env.prod \
  --restart unless-stopped \
  saia-law-api

# Step 5: Wait for health check
echo "⏳ Waiting for container to be healthy..."
sleep 5

# Step 6: Verify
docker ps | grep saia-law-api
curl -s https://demo-law.bineyes.com/health | python3 -m json.tool

echo "✅ Deployment complete!"
```

Make it executable:
```bash
chmod +x /opt/saia/deploy-production.sh
```

---

## 🎯 Future Deployments

**From now on, ALWAYS use**:
```bash
ssh root@134.209.10.163
cd /opt/saia
./deploy-production.sh
```

**NEVER**:
- ❌ Use `/opt/saia-rag/` directory
- ❌ Use `docker-compose` (conflicts with other projects)
- ❌ Manually copy files between directories
- ❌ Use port 8000 (reserved for salesagent)

---

## 🔍 Troubleshooting

### Container won't start
```bash
# Check logs
docker logs saia-law-api

# Check if port 8001 is free
netstat -tulpn | grep 8001

# Check network exists
docker network ls | grep saia-law-network
```

### Qdrant connection issues
```bash
# Check Qdrant is running
docker ps | grep saia-law-qdrant

# Check they're on same network
docker network inspect saia-law-network
```

### Git issues
```bash
cd /opt/saia
git status
git remote -v  # Should show SAIA-RAG repo
git branch     # Should show feature/rag-conversation-context-v2
```

