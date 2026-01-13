"""Flower ClientApp for federated learning on CIFAR-10."""

from collections import OrderedDict

import torch
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context

from fl_boilerplate.task import Net, get_device, load_data, train, test
from fl_boilerplate.tensorboard_utils import get_client_logger


class FlowerClient(NumPyClient):
    """Flower client for federated learning."""

    def __init__(
        self,
        partition_id: int,
        num_partitions: int,
        local_epochs: int,
        batch_size: int,
        tensorboard_enabled: bool,
        log_dir: str,
    ):
        self.partition_id = partition_id
        self.num_partitions = num_partitions
        self.local_epochs = local_epochs
        self.batch_size = batch_size
        self.tensorboard_enabled = tensorboard_enabled
        self.log_dir = log_dir

        # Initialize model and device
        self.model = Net()
        self.device = get_device()
        self.model.to(self.device)

        # Load data for this partition
        self.trainloader, self.testloader = load_data(
            partition_id, num_partitions, batch_size
        )

        print(f"[Client {partition_id}] Initialized on {self.device}")
        print(f"[Client {partition_id}] Training samples: {len(self.trainloader.dataset)}")
        print(f"[Client {partition_id}] Test samples: {len(self.testloader.dataset)}")

    def get_parameters(self, config):
        """Return model parameters as a list of numpy arrays."""
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def set_parameters(self, parameters):
        """Set model parameters from a list of numpy arrays."""
        state_dict = OrderedDict(
            {k: torch.from_numpy(v) for k, v in zip(self.model.state_dict().keys(), parameters)}
        )
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        """Train model on local data."""
        self.set_parameters(parameters)

        # Get training config
        lr = config.get("lr", 0.01)
        local_epochs = config.get("local_epochs", self.local_epochs)
        server_round = config.get("server_round", 0)

        print(f"[Client {self.partition_id}] Round {server_round}: Training for {local_epochs} epochs")

        # Train the model
        train_loss = train(
            self.model,
            self.trainloader,
            local_epochs,
            lr,
            self.device,
        )

        print(f"[Client {self.partition_id}] Round {server_round}: Loss = {train_loss:.4f}")

        # Log to TensorBoard
        if self.tensorboard_enabled:
            logger = get_client_logger(self.partition_id, log_dir=self.log_dir)
            logger.log_scalar("train/loss", train_loss, server_round)
            logger.flush()

        # Return updated parameters and metrics
        return (
            self.get_parameters(config={}),
            len(self.trainloader.dataset),
            {"train_loss": train_loss, "server_round": server_round},
        )

    def evaluate(self, parameters, config):
        """Evaluate model on local test data."""
        self.set_parameters(parameters)

        server_round = config.get("server_round", 0)

        print(f"[Client {self.partition_id}] Round {server_round}: Evaluating")

        # Evaluate the model
        eval_loss, eval_accuracy = test(self.model, self.testloader, self.device)

        print(f"[Client {self.partition_id}] Round {server_round}: Loss = {eval_loss:.4f}, Accuracy = {eval_accuracy:.4f}")

        # Log to TensorBoard
        if self.tensorboard_enabled:
            logger = get_client_logger(self.partition_id, log_dir=self.log_dir)
            logger.log_scalar("eval/loss", eval_loss, server_round)
            logger.log_scalar("eval/accuracy", eval_accuracy, server_round)
            logger.flush()

        return (
            eval_loss,
            len(self.testloader.dataset),
            {"eval_loss": eval_loss, "eval_acc": eval_accuracy, "server_round": server_round},
        )


def client_fn(context: Context):
    """Create a Flower client for this partition."""
    # Get node configuration
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]

    # Get run configuration
    local_epochs = context.run_config.get("local-epochs", 1)
    batch_size = context.run_config.get("batch-size", 32)
    tensorboard_enabled = context.run_config.get("tensorboard-enabled", True)
    log_dir = context.run_config.get("log-dir", "logs")

    return FlowerClient(
        partition_id=partition_id,
        num_partitions=num_partitions,
        local_epochs=local_epochs,
        batch_size=batch_size,
        tensorboard_enabled=tensorboard_enabled,
        log_dir=log_dir,
    ).to_client()


# Create the ClientApp
app = ClientApp(client_fn=client_fn)
