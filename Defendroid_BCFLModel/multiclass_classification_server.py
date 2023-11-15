import flwr as fl
import numpy as np

from typing import Callable, Dict, List, Optional, Tuple, Union
from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)

class SaveModelStrategy(fl.server.strategy.FedAvg):

    def __init__(
        self,
        *,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 4,
        min_evaluate_clients: int = 4,
        min_available_clients: int = 4
    ) -> None:

        super().__init__()

        self.fraction_fit = fraction_fit
        self.fraction_evaluate = fraction_evaluate
        self.min_fit_clients = min_fit_clients
        self.min_evaluate_clients = min_evaluate_clients
        self.min_available_clients = min_available_clients


    def aggregate_fit(self, rnd, results,failures,):
        aggregated_weights = super().aggregate_fit(rnd, results, failures)
        if aggregated_weights is not None:
            # Save aggregated_weights
            print(f"Saving round {rnd} aggregated_weights...")
            np.savez(f"multiclasss_round-{rnd}-weights.npz", *aggregated_weights)
        return aggregated_weights

    def evaluate(self, server_round: int, parameters: Parameters) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """Evaluate model parameters using an evaluation function."""
        if self.evaluate_fn is None:
            # No evaluation function provided
            return None
        parameters_ndarrays = parameters_to_ndarrays(parameters)
        eval_res = self.evaluate_fn(server_round, parameters_ndarrays, {})
        if eval_res is None:
            return None
        loss, metrics = eval_res
        return loss, metrics

# Create strategy and run server
strategy = SaveModelStrategy()

# Start Flower server for specific rounds of federated learning
fl.server.start_server(
        server_address = '0.0.0.0:8080',
        config=fl.server.ServerConfig(num_rounds=50),
        grpc_max_message_length = 1024*1024*1024,
        strategy = strategy,
)

# strategy = fl.server.strategy.FedAvg()
#
# fl.server.start_server(
#     server_address = '0.0.0.0:8080',
#     config=fl.server.ServerConfig(num_rounds=2),
#     strategy=strategy,
#
# )
