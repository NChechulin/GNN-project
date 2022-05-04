from time import time as current_time
from dataclasses import dataclass
from copy import deepcopy
from tqdm.notebook import tqdm
from GCNConv import GCNConv
from torch_geometric.data import Data
import torch
import numpy as np


@dataclass(repr=False)
class TestResult:
    model_name: str
    losses: np.array
    accuracies: np.array
    training_times: np.array

    @property
    def avg_final_loss(self) -> float:
        final_losses = [loss[-1] for loss in self.losses]
        return sum(final_losses) / len(final_losses)

    @property
    def avg_accuracy(self) -> float:
        return sum(self.accuracies) / len(self.accuracies)

    @property
    def avg_training_time(self) -> float:
        return sum(self.training_times) / len(self.training_times)

    def __repr__(self) -> str:
        lines = [
            f"Model:                  {self.model_name}",
            f"Tests ran:              {len(self.losses)}",
            f"Average training time:  {round(self.avg_training_time, 3)}s",
            f"Average accuracy:       {round(self.avg_accuracy, 5)}",
            f"Average final loss:     {round(self.avg_final_loss, 5)}",
        ]

        return "\n".join(lines)

    def concat(self, other: "TestResult"):
        self.losses = np.concatenate((self.losses, other.losses))
        self.accuracies = np.concatenate((self.accuracies, other.accuracies))
        self.training_times = np.concatenate(
            (self.training_times, other.training_times)
        )


@dataclass
class ModelTester:
    model: GCNConv
    data: Data

    def __epoch_train(self, optimizer, criterion) -> float:
        self.model.train()
        optimizer.zero_grad()  # Clear gradients.
        out = self.model(
            self.data.x, self.data.edge_index
        )  # Perform a single forward pass.
        loss = criterion(
            out[self.data.train_mask], self.data.y[self.data.train_mask]
        )  # Compute the loss solely based on the training nodes.
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        return float(loss)

    def __test(self) -> float:
        self.model.eval()
        out = self.model(self.data.x, self.data.edge_index)
        pred = out.argmax(dim=1)  # Use the class with highest probability.
        test_correct = (
            pred[self.data.test_mask] == self.data.y[self.data.test_mask]
        )  # Check against ground-truth labels.
        test_acc = int(test_correct.sum()) / int(
            self.data.test_mask.sum()
        )  # Derive ratio of correct predictions.
        return test_acc

    def train(self, epochs: int) -> TestResult:
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=0.01, weight_decay=5e-4
        )
        criterion = torch.nn.CrossEntropyLoss()

        model_backup = deepcopy(self.model)

        start_time = current_time()

        losses = [self.__epoch_train(optimizer, criterion) for _ in range(epochs)]

        end_time = current_time()

        accuracy = self.__test()

        self.model = model_backup

        return TestResult(
            model_name=self.model.name,
            training_times=np.array([end_time - start_time]),
            losses=np.array([losses]),
            accuracies=np.array([accuracy]),
        )

    def bulk_train(self, steps: int, epochs: int) -> TestResult:
        result = None

        for _ in tqdm(range(steps)):
            iter_result = self.train(epochs)
            if result is None:
                result = iter_result
            else:
                result.concat(deepcopy(iter_result))

        return result
