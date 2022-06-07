from copy import deepcopy
from dataclasses import dataclass
from time import time as current_time
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from tqdm.notebook import tqdm
from scipy.special import entr, softmax
from numpy import log2

from GCNConv import GCNConvModel


@dataclass(repr=False)
class TestResult:
    model_name: str
    losses: np.array
    accuracies: np.array
    training_times: np.array
    entropies: np.array

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

    @property
    def avg_entropy(self) -> float:
        return sum(self.entropies) / len(self.entropies)

    def __repr__(self) -> str:
        lines = [
            f"Model:                  {self.model_name}",
            f"Tests ran:              {len(self.losses)}",
            f"Average training time:  {round(self.avg_training_time, 3)}s",
            f"Average accuracy:       {round(self.avg_accuracy, 5)}",
            f"Average final loss:     {round(self.avg_final_loss, 5)}",
            f"Average entropy:        {round(self.avg_entropy, 5)}"
        ]

        return "\n".join(lines)

    def concat(self, other: "TestResult"):
        self.losses = np.concatenate((self.losses, other.losses))
        self.accuracies = np.concatenate((self.accuracies, other.accuracies))
        self.training_times = np.concatenate(
            (self.training_times, other.training_times)
        )
        self.entropies = np.concatenate((self.entropies, other.entropies))


@dataclass
class ModelTester:
    model: GCNConvModel

    def __epoch_train(self, optimizer) -> float:
        self.model.train()
        optimizer.zero_grad()

        mask = self.model.data.train_mask
        loss = F.nll_loss(self.model()[mask], self.model.data.y[mask])
        loss.backward()
        optimizer.step()

        return float(loss)

    def __entropy(self, row: np.array) -> float:
        return (row * log2(row)).sum() * (-1) / log2(len(row))

    def __test(self) -> Tuple[float, float]:
        self.model.eval()
        logits = self.model()

        mask = self.model.data.test_mask
        pred = logits[mask].max(1)[1]
        test_acc = pred.eq(self.model.data.y[mask]).sum().item() / mask.sum().item()

        sm = softmax(logits[mask].detach().numpy(), axis=1)
        entropies = np.array([self.__entropy(row) for row in sm])
        avg_entropy = entropies.sum() / len(entropies)
        # 1000x7 tensor
        # stores well something

        return test_acc, avg_entropy

    def train(self, epochs: int) -> TestResult:
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=0.01, weight_decay=5e-4
        )
        model_backup = deepcopy(self.model)

        start_time = current_time()

        losses = [self.__epoch_train(optimizer) for _ in range(epochs)]

        end_time = current_time()

        accuracy, entropy = self.__test()

        self.model = model_backup

        return TestResult(
            model_name=self.model.name,
            training_times=np.array([end_time - start_time]),
            losses=np.array([losses]),
            accuracies=np.array([accuracy]),
            entropies=np.array([entropy]),
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

#%%
