"""Flower ServerApp for federated learning coordination."""

from pathlib import Path

import torch
from flwr.common import Context, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server import ServerApp
from flwr.server.strategy import FedAvg
from flwr.server.workflow import DefaultWorkflow

from fl_boilerplate.task import Net, get_device
from fl_boilerplate.tensorboard_utils import close_all_loggers, get_server_logger

# Create the ServerApp
app = ServerApp()


def get_initial_parameters():
    """Get initial model parameters."""
    model = Net()
    # Convert model state dict to list of numpy arrays
    ndarrays = [val.cpu().numpy() for val in model.state_dict().values()]
    return ndarrays_to_parameters(ndarrays)


@app.main()
def main(driver, context: Context) -> None:
    """Main entry point for the ServerApp.

    Coordinates federated learning by:
    1. Initializing the global model
    2. Running FedAvg strategy for multiple rounds
    3. Logging metrics to TensorBoard
    4. Saving the final model

    Args:
        driver: Flower Driver for communicating with clients
        context: Flower context with run configuration
    """
    # Read run configuration
    num_rounds: int = context.run_config["num-server-rounds"]
    lr: float = context.run_config["learning-rate"]
    fraction_fit: float = context.run_config.get("fraction-fit", 1.0)
    fraction_evaluate: float = context.run_config["fraction-evaluate"]
    min_fit_clients: int = context.run_config.get("min-fit-clients", 2)
    min_evaluate_clients: int = context.run_config.get("min-evaluate-clients", 2)
    min_available_clients: int = context.run_config.get("min-available-clients", 2)
    tensorboard_enabled: bool = context.run_config.get("tensorboard-enabled", True)
    log_dir: str = context.run_config.get("log-dir", "logs")

    print(f"\n{'='*60}")
    print("Flower Federated Learning - Server Starting")
    print(f"{'='*60}")
    print(f"Configuration:")
    print(f"  - Rounds: {num_rounds}")
    print(f"  - Learning rate: {lr}")
    print(f"  - Fraction fit: {fraction_fit}")
    print(f"  - Fraction evaluate: {fraction_evaluate}")
    print(f"  - Min clients: {min_available_clients}")
    print(f"  - TensorBoard: {'enabled' if tensorboard_enabled else 'disabled'}")
    print(f"{'='*60}\n")

    # Initialize TensorBoard logger
    if tensorboard_enabled:
        tb_logger = get_server_logger(log_dir=log_dir)
    else:
        tb_logger = None

    # Get device for any server-side computation
    device = get_device()
    print(f"Server device: {device}")

    # Get initial model parameters
    initial_parameters = get_initial_parameters()

    # Define fit and evaluate config functions
    def fit_config(server_round: int):
        """Return training configuration for each round."""
        return {
            "lr": lr,
            "local_epochs": context.run_config.get("local-epochs", 1),
            "batch_size": context.run_config.get("batch-size", 32),
            "server_round": server_round,
        }

    def evaluate_config(server_round: int):
        """Return evaluation configuration for each round."""
        return {
            "batch_size": context.run_config.get("batch-size", 32),
            "server_round": server_round,
        }

    # Custom metric aggregation function
    def fit_metrics_aggregation_fn(metrics):
        """Aggregate training metrics from all clients."""
        if not metrics:
            return {}

        total_examples = sum(num_examples for num_examples, _ in metrics)

        # Weighted average of training loss
        train_loss = sum(
            num_examples * m.get("train_loss", 0) for num_examples, m in metrics
        ) / total_examples if total_examples > 0 else 0

        aggregated = {"train_loss": train_loss, "num_examples": total_examples}

        # Log to TensorBoard
        if tb_logger and metrics:
            # Get round from first client's metrics if available
            _, first_metrics = metrics[0]
            server_round = first_metrics.get("server_round", 0)
            tb_logger.log_scalar("aggregated/train_loss", train_loss, server_round)
            tb_logger.log_scalar("aggregated/train_examples", total_examples, server_round)
            tb_logger.flush()

        return aggregated

    def evaluate_metrics_aggregation_fn(metrics):
        """Aggregate evaluation metrics from all clients."""
        if not metrics:
            return {}

        total_examples = sum(num_examples for num_examples, _ in metrics)

        # Weighted average of evaluation metrics
        eval_loss = sum(
            num_examples * m.get("eval_loss", 0) for num_examples, m in metrics
        ) / total_examples if total_examples > 0 else 0

        eval_acc = sum(
            num_examples * m.get("eval_acc", 0) for num_examples, m in metrics
        ) / total_examples if total_examples > 0 else 0

        aggregated = {
            "eval_loss": eval_loss,
            "eval_acc": eval_acc,
            "num_examples": total_examples,
        }

        # Log to TensorBoard
        if tb_logger and metrics:
            _, first_metrics = metrics[0]
            server_round = first_metrics.get("server_round", 0)
            tb_logger.log_scalar("aggregated/eval_loss", eval_loss, server_round)
            tb_logger.log_scalar("aggregated/eval_accuracy", eval_acc, server_round)
            tb_logger.flush()

        return aggregated

    # Initialize FedAvg strategy
    strategy = FedAvg(
        fraction_fit=fraction_fit,
        fraction_evaluate=fraction_evaluate,
        min_fit_clients=min_fit_clients,
        min_evaluate_clients=min_evaluate_clients,
        min_available_clients=min_available_clients,
        initial_parameters=initial_parameters,
        on_fit_config_fn=fit_config,
        on_evaluate_config_fn=evaluate_config,
        fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
        evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
    )

    # Run federated learning using DefaultWorkflow
    print(f"\nStarting FedAvg for {num_rounds} rounds...\n")

    workflow = DefaultWorkflow(strategy)
    workflow(driver, context, num_rounds)

    # Get final parameters from strategy
    print(f"\n{'='*60}")
    print("Federated Learning Complete")
    print(f"{'='*60}")

    # Save final model
    if strategy.parameters is not None:
        output_dir = Path("outputs")
        output_dir.mkdir(exist_ok=True)
        model_path = output_dir / "final_model.pt"

        print(f"\nSaving final model to {model_path}...")

        # Convert parameters back to state dict
        final_ndarrays = parameters_to_ndarrays(strategy.parameters)
        model = Net()
        state_dict = model.state_dict()

        # Map ndarrays back to state dict keys
        for key, ndarray in zip(state_dict.keys(), final_ndarrays):
            state_dict[key] = torch.from_numpy(ndarray)

        torch.save(state_dict, model_path)
        print("Model saved successfully!")

        # Also save a checkpoint with metadata
        checkpoint_path = output_dir / "checkpoint.pt"
        checkpoint = {
            "model_state_dict": state_dict,
            "num_rounds": num_rounds,
            "learning_rate": lr,
        }
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")
    else:
        print("\nWarning: No final parameters available to save.")

    # Clean up TensorBoard loggers
    if tb_logger:
        tb_logger.flush()
    close_all_loggers()

    print(f"\n{'='*60}")
    print("Server shutdown complete")
    print(f"{'='*60}\n")
