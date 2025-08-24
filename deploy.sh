#!/bin/bash

# SAIA-RAG Production Deployment Script
# Usage: ./deploy.sh [server_ip] [user]

set -e  # Exit on any error

# Configuration
SERVER_IP=${1:-"134.209.10.163"}
USER=${2:-"root"}
PROJECT_NAME="SAIA-RAG"
REMOTE_PATH="/opt/saia-rag"

echo "🚀 Starting SAIA-RAG deployment to $USER@$SERVER_IP"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Check if we can connect to the server
echo "🔍 Testing connection to server..."
if ! ssh -o ConnectTimeout=10 $USER@$SERVER_IP "echo 'Connection successful'" > /dev/null 2>&1; then
    print_error "Cannot connect to $USER@$SERVER_IP"
    echo "Please ensure:"
    echo "1. SSH key is set up"
    echo "2. Server is accessible"
    echo "3. User has sudo privileges"
    exit 1
fi
print_status "Server connection verified"

# Create deployment directory on server
echo "📁 Setting up deployment directory..."
ssh $USER@$SERVER_IP "mkdir -p $REMOTE_PATH"
print_status "Deployment directory created"

# Copy project files to server
echo "📤 Uploading project files..."
rsync -avz --progress \
    --exclude='.git' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='.env' \
    --exclude='data/' \
    --exclude='logs/' \
    ./ $USER@$SERVER_IP:$REMOTE_PATH/
print_status "Project files uploaded"

# Copy production environment file
echo "🔧 Setting up production environment..."
scp .env.prod $USER@$SERVER_IP:$REMOTE_PATH/.env.prod
print_status "Production environment configured"

# Install Docker and Docker Compose on server if needed
echo "🐳 Checking Docker installation..."
ssh $USER@$SERVER_IP << 'EOF'
    # Check if Docker is installed
    if ! command -v docker &> /dev/null; then
        echo "Installing Docker..."
        curl -fsSL https://get.docker.com -o get-docker.sh
        sh get-docker.sh
        systemctl enable docker
        systemctl start docker
        rm get-docker.sh
    fi
    
    # Check if Docker Compose is installed
    if ! command -v docker-compose &> /dev/null; then
        echo "Installing Docker Compose..."
        curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
        chmod +x /usr/local/bin/docker-compose
    fi
    
    echo "Docker version: $(docker --version)"
    echo "Docker Compose version: $(docker-compose --version)"
EOF
print_status "Docker installation verified"

# Deploy the application
echo "🚀 Deploying SAIA-RAG application..."
ssh $USER@$SERVER_IP << EOF
    cd $REMOTE_PATH
    
    # Stop existing containers if running
    if [ -f docker-compose.prod.yml ]; then
        docker-compose -f docker-compose.prod.yml down || true
    fi
    
    # Build and start production containers
    docker-compose -f docker-compose.prod.yml up -d --build
    
    # Wait for services to be healthy
    echo "⏳ Waiting for services to start..."
    sleep 30
    
    # Check service status
    docker-compose -f docker-compose.prod.yml ps
EOF

# Verify deployment
echo "🔍 Verifying deployment..."
sleep 10

# Test health endpoint
if curl -f -s http://$SERVER_IP:8000/health > /dev/null; then
    print_status "API health check passed"
else
    print_warning "API health check failed - checking logs..."
    ssh $USER@$SERVER_IP "cd $REMOTE_PATH && docker-compose -f docker-compose.prod.yml logs api"
fi

# Test web interface
if curl -f -s http://$SERVER_IP/health > /dev/null; then
    print_status "Web interface accessible via Caddy"
else
    print_warning "Web interface not accessible via Caddy (port 80)"
fi

echo ""
echo "🎉 Deployment completed!"
echo ""
echo "📋 Access Information:"
echo "   • API Direct: http://$SERVER_IP:8000"
echo "   • Web Interface: http://$SERVER_IP/ui"
echo "   • API Docs: http://$SERVER_IP:8000/docs"
echo "   • Health Check: http://$SERVER_IP/health"
echo ""
echo "🔧 Management Commands:"
echo "   • View logs: ssh $USER@$SERVER_IP 'cd $REMOTE_PATH && docker-compose -f docker-compose.prod.yml logs -f'"
echo "   • Restart: ssh $USER@$SERVER_IP 'cd $REMOTE_PATH && docker-compose -f docker-compose.prod.yml restart'"
echo "   • Stop: ssh $USER@$SERVER_IP 'cd $REMOTE_PATH && docker-compose -f docker-compose.prod.yml down'"
echo ""
print_status "SAIA-RAG is now running on $SERVER_IP"
EOF
