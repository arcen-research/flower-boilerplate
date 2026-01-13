"""Flower ServerApp for federated learning coordination."""

from pathlib import Path

import torch
from flwr.common import Context
from flwr.common.record import ArrayRecord, ConfigRecord
from flwr.server import Grid, ServerApp
from flwr.server.strategy import FedAvg

from fl_boilerplate.task import Net, get_device
from fl_boilerplate.tensorboard_utils import close_all_loggers, get_server_logger

# Create the ServerApp
app = ServerApp()


def get_global_evaluate_fn():
    """Create a global evaluation function for centralized evaluation.

    For this boilerplate, we rely on federated evaluation from clients.
    To add centralized evaluation, implement a function here that:
    1. Loads a global test dataset
    2. Evaluates the model on it
    3. Returns metrics

    Returns:
        None (no centralized evaluation by default)
    """
    return None


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Main entry point for the ServerApp.

    Coordinates federated learning by:
    1. Initializing the global model
    2. Running FedAvg strategy for multiple rounds
    3. Logging metrics to TensorBoard
    4. Saving the final model

    Args:
        grid: Flower Grid for communicating with clients
        context: Flower context with run configuration
    """
    # Read run configuration
    num_rounds: int = context.run_config["num-server-rounds"]
    lr: float = context.run_config["learning-rate"]
    fraction_evaluate: float = context.run_config["fraction-evaluate"]
    tensorboard_enabled: bool = context.run_config.get("tensorboard-enabled", True)
    log_dir: str = context.run_config.get("log-dir", "logs")

    print(f"\n{'='*60}")
    print("Flower Federated Learning - Server Starting")
    print(f"{'='*60}")
    print(f"Configuration:")
    print(f"  - Rounds: {num_rounds}")
    print(f"  - Learning rate: {lr}")
    print(f"  - Fraction evaluate: {fraction_evaluate}")
    print(f"  - TensorBoard: {'enabled' if tensorboard_enabled else 'disabled'}")
    print(f"{'='*60}\n")

    # Initialize TensorBoard logger
    if tensorboard_enabled:
        logger = get_server_logger(log_dir=log_dir)
    else:
        logger = None

    # Get device for any server-side computation
    device = get_device()
    print(f"Server device: {device}")

    # Initialize global model
    global_model = Net()
    arrays = ArrayRecord(global_model.state_dict())

    # Initialize FedAvg strategy
    strategy = FedAvg(fraction_evaluate=fraction_evaluate)

    # Get global evaluation function (optional)
    evaluate_fn = get_global_evaluate_fn()

    # Run federated learning
    print(f"\nStarting FedAvg for {num_rounds} rounds...\n")

    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=ConfigRecord({"lr": lr}),
        num_rounds=num_rounds,
        evaluate_fn=evaluate_fn,
    )

    # Log final results
    print(f"\n{'='*60}")
    print("Federated Learning Complete")
    print(f"{'='*60}")

    # Extract and log final metrics
    if result.train_metrics:
        print("\nFinal Training Metrics:")
        for round_num, metrics in result.train_metrics.items():
            print(f"  Round {round_num}: {metrics}")
            if logger:
                for key, value in metrics.items():
                    if isinstance(value, (int, float)):
                        logger.log_scalar(f"aggregated/train_{key}", float(value), round_num)

    if result.evaluate_metrics:
        print("\nFinal Evaluation Metrics:")
        for round_num, metrics in result.evaluate_metrics.items():
            print(f"  Round {round_num}: {metrics}")
            if logger:
                for key, value in metrics.items():
                    if isinstance(value, (int, float)):
                        logger.log_scalar(f"aggregated/eval_{key}", float(value), round_num)

    # Save final model
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    model_path = output_dir / "final_model.pt"

    print(f"\nSaving final model to {model_path}...")
    final_state_dict = result.arrays.to_torch_state_dict()
    torch.save(final_state_dict, model_path)
    print("Model saved successfully!")

    # Also save a checkpoint with metadata
    checkpoint_path = output_dir / "checkpoint.pt"
    checkpoint = {
        "model_state_dict": final_state_dict,
        "num_rounds": num_rounds,
        "learning_rate": lr,
    }
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path}")

    # Clean up TensorBoard loggers
    if logger:
        logger.flush()
    close_all_loggers()

    print(f"\n{'='*60}")
    print("Server shutdown complete")
    print(f"{'='*60}\n")
