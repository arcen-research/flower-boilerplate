"""Flower ClientApp for federated learning on CIFAR-10."""

from flwr.client import ClientApp
from flwr.common import Context, Message
from flwr.common.record import ArrayRecord, MetricRecord, RecordDict

from fl_boilerplate.task import Net, get_device, load_data, train, test
from fl_boilerplate.tensorboard_utils import get_client_logger

# Create the ClientApp
app = ClientApp()


@app.train()
def train_fn(msg: Message, context: Context) -> Message:
    """Train the model on local data.

    Receives model weights from server, trains locally, and returns updated weights.

    Args:
        msg: Message containing model weights and training config
        context: Flower context with run and node configuration

    Returns:
        Message with updated model weights and training metrics
    """
    # Get node configuration
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]

    # Get run configuration
    batch_size = context.run_config["batch-size"]
    local_epochs = context.run_config["local-epochs"]
    tensorboard_enabled = context.run_config.get("tensorboard-enabled", True)
    log_dir = context.run_config.get("log-dir", "logs")

    # Get training hyperparameters from server
    lr = msg.content["config"]["lr"]

    # Initialize model with received weights
    model = Net()
    state_dict = msg.content["arrays"].to_torch_state_dict()
    model.load_state_dict(state_dict)

    # Get device and load data
    device = get_device()
    trainloader, _ = load_data(partition_id, num_partitions, batch_size)

    print(f"[Client {partition_id}] Training on {device} with {len(trainloader.dataset)} samples")

    # Train the model
    train_loss = train(model, trainloader, local_epochs, lr, device)

    print(f"[Client {partition_id}] Training loss: {train_loss:.4f}")

    # Log to TensorBoard
    if tensorboard_enabled:
        logger = get_client_logger(partition_id, log_dir=log_dir)
        # Note: We don't have round number in client context, log with step=0
        # Server-side logging tracks per-round metrics
        logger.log_scalar("local/train_loss", train_loss, 0)
        logger.flush()

    # Build response message
    model_record = ArrayRecord(model.state_dict())
    metrics = {
        "train_loss": train_loss,
        "num-examples": len(trainloader.dataset),
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"arrays": model_record, "metrics": metric_record})

    return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate_fn(msg: Message, context: Context) -> Message:
    """Evaluate the model on local test data.

    Args:
        msg: Message containing model weights to evaluate
        context: Flower context with configuration

    Returns:
        Message with evaluation metrics
    """
    # Get node configuration
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]

    # Get run configuration
    batch_size = context.run_config["batch-size"]
    tensorboard_enabled = context.run_config.get("tensorboard-enabled", True)
    log_dir = context.run_config.get("log-dir", "logs")

    # Initialize model with received weights
    model = Net()
    state_dict = msg.content["arrays"].to_torch_state_dict()
    model.load_state_dict(state_dict)

    # Get device and load data
    device = get_device()
    _, testloader = load_data(partition_id, num_partitions, batch_size)

    print(f"[Client {partition_id}] Evaluating on {device} with {len(testloader.dataset)} samples")

    # Evaluate the model
    eval_loss, eval_accuracy = test(model, testloader, device)

    print(f"[Client {partition_id}] Eval loss: {eval_loss:.4f}, Accuracy: {eval_accuracy:.4f}")

    # Log to TensorBoard
    if tensorboard_enabled:
        logger = get_client_logger(partition_id, log_dir=log_dir)
        logger.log_scalar("local/eval_loss", eval_loss, 0)
        logger.log_scalar("local/eval_accuracy", eval_accuracy, 0)
        logger.flush()

    # Build response message (no model weights needed for evaluation)
    metrics = {
        "eval_loss": eval_loss,
        "eval_acc": eval_accuracy,
        "num-examples": len(testloader.dataset),
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"metrics": metric_record})

    return Message(content=content, reply_to=msg)
