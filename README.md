# Flower Federated Learning Boilerplate

A production-ready Flower FL boilerplate using the Deployment Engine (SuperLink/SuperNode architecture) for heterogeneous hardware deployments.

## Supported Platforms

| Node | Role | Architecture | Device |
|------|------|--------------|--------|
| MacBook M4 | SuperLink (Server) | ARM64 | MPS |
| Linux PC | SuperNode (Client) | x86_64 | CUDA/CPU |
| Raspberry Pi 4B | SuperNode (Client) | ARM64 | CPU |

## Quick Start

### 1. Clone and Setup

```bash
# Clone the repository
rad clone <your-repo-url> flower-boilerplate
cd flower-boilerplate

# Run setup script (creates venv, installs dependencies)
./scripts/setup_node.sh

# Activate the virtual environment
source .venv/bin/activate
```

### 2. Local Testing (Single Machine)

Test the full FL pipeline on your MacBook before distributed deployment:

```bash
# Terminal 1: Start SuperLink
./scripts/start_superlink.sh

# Terminal 2: Start first SuperNode (Client 0)
./scripts/start_supernode.sh 0 2

# Terminal 3: Start second SuperNode (Client 1)
./scripts/start_supernode.sh 1 2

# Terminal 4: Run federated learning
flwr run . local-deployment --stream
```

### 3. View TensorBoard Metrics

```bash
tensorboard --logdir=logs/
# Open http://localhost:6006 in your browser
```

## Distributed Deployment

### Step 1: Setup Server (MacBook M4)

```bash
# Get your server's IP address
ipconfig getifaddr en0  # e.g., 192.168.1.100

# Start SuperLink in distributed mode
./scripts/start_superlink.sh distributed
```

### Step 2: Setup Clients (Linux PC & Raspberry Pi)

On each client machine:

```bash
# Clone the repo (via git or Radicle)
git clone <your-repo-url> flower-boilerplate
cd flower-boilerplate

# Run setup
./scripts/setup_node.sh
source .venv/bin/activate

# Start SuperNode (replace IP with your server's address)
# Linux PC (partition 0):
./scripts/start_supernode.sh 0 2 192.168.1.100:9092

# Raspberry Pi (partition 1):
./scripts/start_supernode.sh 1 2 192.168.1.100:9092
```

### Step 3: Run Federated Learning

From the server (MacBook):

```bash
# Run FL with the distributed federation
flwr run . distributed --stream
```

## Configuration

### Training Parameters

Edit `pyproject.toml` or override via command line:

```bash
# Run with custom parameters
flwr run . local-deployment --stream \
    --run-config "num-server-rounds=5 local-epochs=2 batch-size=16"
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num-server-rounds` | 3 | Number of FL rounds |
| `local-epochs` | 1 | Local training epochs per round |
| `batch-size` | 32 | Batch size (reduce for Pi) |
| `learning-rate` | 0.01 | SGD learning rate |
| `fraction-evaluate` | 1.0 | Fraction of clients for evaluation |

### Raspberry Pi Recommendations

Due to limited RAM (4GB), use smaller batch sizes:

```bash
./scripts/start_supernode.sh 1 2 192.168.1.100:9092
# Then from server:
flwr run . distributed --stream --run-config "batch-size=16"
```

## Project Structure

```
flower-boilerplate/
├── fl_boilerplate/
│   ├── __init__.py
│   ├── client_app.py      # ClientApp (train/evaluate)
│   ├── server_app.py      # ServerApp (FedAvg coordination)
│   ├── task.py            # Model, training, data loading
│   └── tensorboard_utils.py
├── scripts/
│   ├── start_superlink.sh # Server startup
│   ├── start_supernode.sh # Client startup
│   └── setup_node.sh      # Environment setup
├── configs/
│   └── node_configs.yaml  # Reference configuration
├── logs/                  # TensorBoard logs
├── outputs/               # Saved models
├── pyproject.toml         # Project config
└── README.md
```

## Architecture

```
┌─────────────────┐
│  MacBook M4     │
│  (SuperLink)    │
│  Port 9092/9093 │
└────────┬────────┘
         │ gRPC
    ┌────┴────┐
    │         │
    ▼         ▼
┌────────┐ ┌────────┐
│Linux PC│ │Rasp Pi │
│Client 0│ │Client 1│
│(x86)   │ │(ARM)   │
└────────┘ └────────┘
```

## Troubleshooting

### Connection Issues

```bash
# Check if SuperLink is running
curl -s http://192.168.1.100:9093/v1/info || echo "SuperLink not reachable"

# Verify firewall allows ports 9092-9095
# macOS:
sudo pfctl -s rules | grep 909

# Linux:
sudo iptables -L -n | grep 909
```

### Memory Issues on Raspberry Pi

```bash
# Monitor memory usage
free -h
htop

# Use smaller batch size
flwr run . distributed --run-config "batch-size=8"
```

### Checking Logs

```bash
# TensorBoard for metrics visualization
tensorboard --logdir=logs/

# Check saved models
ls -la outputs/
```

## Version Control with Radicle

This project is designed to work with Radicle for decentralized git:

```bash
# Initialize Radicle (on your main machine)
rad init

# Push to Radicle
rad push

# On other nodes, clone via Radicle
rad clone <project-id>
```

## Customization

### Using a Different Model

Edit `fl_boilerplate/task.py` and replace the `Net` class with your model.

### Using a Different Dataset

Modify the `load_data()` function in `task.py` to load your dataset.

### Adding Centralized Evaluation

Implement the `get_global_evaluate_fn()` in `server_app.py` to evaluate on a global test set.

## License

Apache 2.0
