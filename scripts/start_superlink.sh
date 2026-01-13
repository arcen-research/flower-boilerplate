#!/bin/bash
# Start Flower SuperLink (Server)
#
# Run this script on your server machine (MacBook M4).
# The SuperLink coordinates the federated learning process.
#
# Usage:
#   ./scripts/start_superlink.sh              # Local testing (127.0.0.1)
#   ./scripts/start_superlink.sh distributed  # Distributed (0.0.0.0, accessible from network)
#
# The SuperLink exposes these ports:
#   - 9092: SuperNode connections (gRPC)
#   - 9093: flwr run connections (REST API)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

MODE="${1:-local}"

echo "============================================================"
echo "Flower SuperLink (Server)"
echo "============================================================"

if [ "$MODE" = "distributed" ]; then
    echo "Mode: DISTRIBUTED (listening on all interfaces)"
    echo "Other machines can connect to this server's IP address"
    echo ""
    echo "To find your IP address:"
    echo "  macOS: ipconfig getifaddr en0"
    echo "  Linux: hostname -I | awk '{print \$1}'"
    echo ""
    FLEET_API_ADDRESS="0.0.0.0:9092"
    EXEC_API_ADDRESS="0.0.0.0:9093"
else
    echo "Mode: LOCAL (localhost only)"
    echo "For distributed deployment, run: $0 distributed"
    echo ""
    FLEET_API_ADDRESS="127.0.0.1:9092"
    EXEC_API_ADDRESS="127.0.0.1:9093"
fi

echo "Fleet API (SuperNodes): $FLEET_API_ADDRESS"
echo "Exec API (flwr run):    $EXEC_API_ADDRESS"
echo "============================================================"
echo ""

flower-superlink \
    --insecure \
    --fleet-api-address "$FLEET_API_ADDRESS" \
    --exec-api-address "$EXEC_API_ADDRESS"
