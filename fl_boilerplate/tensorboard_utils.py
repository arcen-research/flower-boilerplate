"""TensorBoard logging utilities for Federated Learning."""

import os
from datetime import datetime
from pathlib import Path
from typing import Optional

from torch.utils.tensorboard import SummaryWriter


class FLTensorBoardLogger:
    """TensorBoard logger for Federated Learning experiments.

    Supports logging from both server and client side with separate subdirectories.
    Creates timestamped run directories for easy experiment comparison.
    """

    def __init__(
        self,
        log_dir: str = "logs",
        run_name: Optional[str] = None,
        node_type: str = "server",
        node_id: Optional[int] = None,
    ) -> None:
        """Initialize the TensorBoard logger.

        Args:
            log_dir: Base directory for TensorBoard logs
            run_name: Optional name for this run (defaults to timestamp)
            node_type: Either "server" or "client"
            node_id: Client ID (required if node_type is "client")
        """
        self.log_dir = Path(log_dir)
        self.node_type = node_type
        self.node_id = node_id

        # Create run name with timestamp if not provided
        if run_name is None:
            run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_name = run_name

        # Build log path: logs/<run_name>/<node_type>[_<node_id>]
        if node_type == "client" and node_id is not None:
            subdir = f"client_{node_id}"
        else:
            subdir = node_type

        self.writer_path = self.log_dir / run_name / subdir
        self.writer: Optional[SummaryWriter] = None
        self._enabled = True

    def enable(self) -> None:
        """Enable TensorBoard logging."""
        self._enabled = True

    def disable(self) -> None:
        """Disable TensorBoard logging."""
        self._enabled = False
        self.close()

    def _get_writer(self) -> SummaryWriter:
        """Lazily initialize the SummaryWriter."""
        if self.writer is None:
            os.makedirs(self.writer_path, exist_ok=True)
            self.writer = SummaryWriter(log_dir=str(self.writer_path))
        return self.writer

    def log_scalar(self, tag: str, value: float, step: int) -> None:
        """Log a scalar value.

        Args:
            tag: Name of the metric (e.g., "train/loss", "eval/accuracy")
            value: The scalar value to log
            step: Global step (usually the round number)
        """
        if not self._enabled:
            return
        self._get_writer().add_scalar(tag, value, step)

    def log_scalars(self, main_tag: str, tag_scalar_dict: dict, step: int) -> None:
        """Log multiple scalars under the same main tag.

        Args:
            main_tag: Main tag name (e.g., "loss")
            tag_scalar_dict: Dictionary of {sub_tag: value}
            step: Global step
        """
        if not self._enabled:
            return
        self._get_writer().add_scalars(main_tag, tag_scalar_dict, step)

    def log_round_metrics(
        self,
        round_num: int,
        train_loss: Optional[float] = None,
        eval_loss: Optional[float] = None,
        eval_accuracy: Optional[float] = None,
        num_examples: Optional[int] = None,
    ) -> None:
        """Log standard FL round metrics.

        Convenience method for logging common FL metrics.

        Args:
            round_num: Current FL round number
            train_loss: Training loss (if available)
            eval_loss: Evaluation loss (if available)
            eval_accuracy: Evaluation accuracy (if available)
            num_examples: Number of training examples used
        """
        if not self._enabled:
            return

        if train_loss is not None:
            self.log_scalar("train/loss", train_loss, round_num)
        if eval_loss is not None:
            self.log_scalar("eval/loss", eval_loss, round_num)
        if eval_accuracy is not None:
            self.log_scalar("eval/accuracy", eval_accuracy, round_num)
        if num_examples is not None:
            self.log_scalar("train/num_examples", num_examples, round_num)

    def log_aggregated_metrics(
        self,
        round_num: int,
        metrics: dict,
        prefix: str = "aggregated",
    ) -> None:
        """Log aggregated metrics from server.

        Args:
            round_num: Current FL round number
            metrics: Dictionary of metric names to values
            prefix: Prefix for metric tags
        """
        if not self._enabled:
            return

        for name, value in metrics.items():
            if isinstance(value, (int, float)):
                self.log_scalar(f"{prefix}/{name}", value, round_num)

    def log_client_participation(
        self,
        round_num: int,
        num_clients_sampled: int,
        num_clients_succeeded: int,
        num_clients_failed: int,
    ) -> None:
        """Log client participation statistics.

        Args:
            round_num: Current FL round number
            num_clients_sampled: Number of clients sampled for this round
            num_clients_succeeded: Number of clients that completed successfully
            num_clients_failed: Number of clients that failed
        """
        if not self._enabled:
            return

        self.log_scalar("clients/sampled", num_clients_sampled, round_num)
        self.log_scalar("clients/succeeded", num_clients_succeeded, round_num)
        self.log_scalar("clients/failed", num_clients_failed, round_num)

    def flush(self) -> None:
        """Flush pending writes to disk."""
        if self.writer is not None:
            self.writer.flush()

    def close(self) -> None:
        """Close the SummaryWriter."""
        if self.writer is not None:
            self.writer.close()
            self.writer = None

    def __enter__(self) -> "FLTensorBoardLogger":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()


# Global logger instances for convenience
_server_logger: Optional[FLTensorBoardLogger] = None
_client_loggers: dict[int, FLTensorBoardLogger] = {}


def get_server_logger(log_dir: str = "logs", run_name: Optional[str] = None) -> FLTensorBoardLogger:
    """Get or create the server-side TensorBoard logger.

    Args:
        log_dir: Base directory for logs
        run_name: Optional run name

    Returns:
        The server TensorBoard logger instance
    """
    global _server_logger
    if _server_logger is None:
        _server_logger = FLTensorBoardLogger(
            log_dir=log_dir,
            run_name=run_name,
            node_type="server",
        )
    return _server_logger


def get_client_logger(
    node_id: int,
    log_dir: str = "logs",
    run_name: Optional[str] = None,
) -> FLTensorBoardLogger:
    """Get or create a client-side TensorBoard logger.

    Args:
        node_id: The client's partition ID
        log_dir: Base directory for logs
        run_name: Optional run name

    Returns:
        The client TensorBoard logger instance for this node
    """
    global _client_loggers
    if node_id not in _client_loggers:
        _client_loggers[node_id] = FLTensorBoardLogger(
            log_dir=log_dir,
            run_name=run_name,
            node_type="client",
            node_id=node_id,
        )
    return _client_loggers[node_id]


def close_all_loggers() -> None:
    """Close all logger instances."""
    global _server_logger, _client_loggers

    if _server_logger is not None:
        _server_logger.close()
        _server_logger = None

    for logger in _client_loggers.values():
        logger.close()
    _client_loggers.clear()
