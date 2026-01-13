"""fl_boilerplate: A Flower / PyTorch app for heterogeneous edge devices."""

import torch
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from fl_boilerplate.task import Net, load_data
from fl_boilerplate.task import test as test_fn
from fl_boilerplate.task import train as train_fn
from fl_boilerplate.tensorboard_utils import get_client_logger

# Flower ClientApp
app = ClientApp()


@app.train()
def train(msg: Message, context: Context):
    """Train the model on local data."""

    # Load the model and initialize it with the received weights
    model = Net()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load the data
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    batch_size = context.run_config["batch-size"]
    trainloader, _ = load_data(partition_id, num_partitions, batch_size)

    # Initialize TensorBoard logger
    logger = get_client_logger(partition_id)
    # Try to get round number from metadata, default to 0 if not available
    try:
        server_round = int(msg.metadata.get("round", 0)) if hasattr(msg.metadata, "get") else 0
    except (AttributeError, KeyError, TypeError):
        server_round = 0

    # Call the training function
    train_loss = train_fn(
        model,
        trainloader,
        context.run_config["local-epochs"],
        msg.content["config"]["lr"],
        device,
    )

    # Log metrics to TensorBoard
    logger.log_round_metrics(
        round_num=server_round,
        train_loss=train_loss,
        num_examples=len(trainloader.dataset),
    )
    logger.flush()

    # Construct and return reply Message
    model_record = ArrayRecord(model.state_dict())
    metrics = {
        "train_loss": train_loss,
        "num-examples": len(trainloader.dataset),
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"arrays": model_record, "metrics": metric_record})
    return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context):
    """Evaluate the model on local data."""

    # Load the model and initialize it with the received weights
    model = Net()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load the data
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    batch_size = context.run_config["batch-size"]
    _, valloader = load_data(partition_id, num_partitions, batch_size)

    # Initialize TensorBoard logger
    logger = get_client_logger(partition_id)
    # Try to get round number from metadata, default to 0 if not available
    try:
        server_round = int(msg.metadata.get("round", 0)) if hasattr(msg.metadata, "get") else 0
    except (AttributeError, KeyError, TypeError):
        server_round = 0

    # Call the evaluation function
    eval_loss, eval_acc = test_fn(
        model,
        valloader,
        device,
    )

    # Log metrics to TensorBoard
    logger.log_round_metrics(
        round_num=server_round,
        eval_loss=eval_loss,
        eval_accuracy=eval_acc,
        num_examples=len(valloader.dataset),
    )
    logger.flush()

    # Construct and return reply Message
    metrics = {
        "eval_loss": eval_loss,
        "eval_acc": eval_acc,
        "num-examples": len(valloader.dataset),
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"metrics": metric_record})
    return Message(content=content, reply_to=msg)
