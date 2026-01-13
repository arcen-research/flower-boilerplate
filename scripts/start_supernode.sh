#!/bin/bash
# Start Flower SuperNode (Client)
#
# Run this script on each client machine (Linux PC, Raspberry Pi).
# Each SuperNode connects to the SuperLink and participates in training.
#
# Usage:
#   ./scripts/start_supernode.sh <partition-id> <num-partitions> [superlink-address]
#
# Examples:
#   # Local testing (2 clients on same machine)
#   ./scripts/start_supernode.sh 0 2                    # Client 0, localhost
#   ./scripts/start_supernode.sh 1 2                    # Client 1, localhost
#
#   # Distributed deployment (connecting to server at 192.168.1.100)
#   ./scripts/start_supernode.sh 0 2 192.168.1.100:9092  # Linux PC
#   ./scripts/start_supernode.sh 1 2 192.168.1.100:9092  # Raspberry Pi

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

# Parse arguments
PARTITION_ID="${1:-0}"
NUM_PARTITIONS="${2:-2}"
SUPERLINK_ADDRESS="${3:-127.0.0.1:9092}"

# Generate unique ClientAppIO port based on partition ID
# This allows running multiple SuperNodes on the same machine
CLIENTAPPIO_PORT=$((9094 + PARTITION_ID))

echo "============================================================"
echo "Flower SuperNode (Client)"
echo "============================================================"
echo "Partition ID:      $PARTITION_ID"
echo "Num Partitions:    $NUM_PARTITIONS"
echo "SuperLink Address: $SUPERLINK_ADDRESS"
echo "ClientAppIO Port:  $CLIENTAPPIO_PORT"
echo "============================================================"
echo ""

# Verify connection to SuperLink
echo "Connecting to SuperLink at $SUPERLINK_ADDRESS..."
echo ""

flower-supernode \
    --insecure \
    --superlink "$SUPERLINK_ADDRESS" \
    --clientappio-api-address "0.0.0.0:$CLIENTAPPIO_PORT" \
    --node-config "partition-id=$PARTITION_ID num-partitions=$NUM_PARTITIONS"
