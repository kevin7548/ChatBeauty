#!/usr/bin/env bash
# =============================================================
# Run this INSIDE the GCP VM after SSH-ing in
# Installs Docker, Nginx, and prepares the environment
# =============================================================
set -euo pipefail

echo "=== Updating system ==="
sudo apt update && sudo apt upgrade -y

echo "=== Installing Docker ==="
sudo apt install -y docker.io docker-compose-plugin
sudo systemctl enable docker
sudo usermod -aG docker "$USER"

echo "=== Installing Nginx ==="
sudo apt install -y nginx certbot python3-certbot-nginx

echo "=== Configuring Nginx ==="
sudo cp ~/chatbeauty/deploy/nginx/chatbeauty.conf /etc/nginx/sites-available/chatbeauty
sudo ln -sf /etc/nginx/sites-available/chatbeauty /etc/nginx/sites-enabled/chatbeauty
sudo rm -f /etc/nginx/sites-enabled/default
sudo nginx -t && sudo systemctl restart nginx

echo "=== Creating project directory ==="
mkdir -p ~/chatbeauty/ml/model/retrieval
mkdir -p ~/chatbeauty/ml/model/reranking
mkdir -p ~/chatbeauty/ml/data/chromadb
mkdir -p ~/chatbeauty/ml/data/cleaned/item_v1

echo ""
echo "=== Setup complete! ==="
echo ""
echo "Next steps:"
echo "  1. Log out and back in (for Docker group)"
echo "  2. Upload model files:"
echo "     - ml/model/retrieval/bge-m3-finetuned-20260202-120852/"
echo "     - ml/model/reranking/lgbm_reranker_current_features_v1.pkl"
echo "     - ml/data/chromadb/"
echo "     - ml/data/cleaned/item_v1/item_features_v1.csv"
echo "  3. Create .env file (copy from .env.example)"
echo "  4. Run: cd ~/chatbeauty && docker compose up -d --build"
echo "  5. Verify: curl http://localhost:8000/health"
echo ""
echo "For SSL (optional, requires domain):"
echo "  sudo certbot --nginx -d your-domain.com"
