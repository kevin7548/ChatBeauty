#!/usr/bin/env bash
# =============================================================
# ChatBeauty GCP Deployment Script
# Run this from your LOCAL machine (not the VM)
# =============================================================
set -euo pipefail

PROJECT_ID="your-gcp-project-id"    # <-- CHANGE THIS
ZONE="asia-northeast3-a"
VM_NAME="chatbeauty-backend"
MACHINE_TYPE="e2-standard-2"
DISK_SIZE="50GB"

echo "=== Step 1: Create VM ==="
gcloud compute instances create "$VM_NAME" \
  --project="$PROJECT_ID" \
  --zone="$ZONE" \
  --machine-type="$MACHINE_TYPE" \
  --boot-disk-size="$DISK_SIZE" \
  --boot-disk-type=pd-ssd \
  --image-family=ubuntu-2204-lts \
  --image-project=ubuntu-os-cloud \
  --tags=http-server,https-server

echo "=== Step 2: Reserve static IP ==="
gcloud compute addresses create chatbeauty-ip \
  --project="$PROJECT_ID" \
  --region=asia-northeast3

STATIC_IP=$(gcloud compute addresses describe chatbeauty-ip \
  --project="$PROJECT_ID" \
  --region=asia-northeast3 \
  --format='get(address)')
echo "Static IP: $STATIC_IP"

echo "=== Step 3: Firewall rules ==="
gcloud compute firewall-rules create allow-http \
  --project="$PROJECT_ID" \
  --allow tcp:80 \
  --target-tags http-server 2>/dev/null || echo "Rule allow-http already exists"

gcloud compute firewall-rules create allow-https \
  --project="$PROJECT_ID" \
  --allow tcp:443 \
  --target-tags https-server 2>/dev/null || echo "Rule allow-https already exists"

echo ""
echo "=== Done! ==="
echo "Static IP: $STATIC_IP"
echo ""
echo "Next steps:"
echo "  1. SSH into VM:  gcloud compute ssh $VM_NAME --zone=$ZONE"
echo "  2. Run setup-vm.sh on the VM"
echo "  3. Upload model files to the VM"
echo "  4. Start docker compose"
