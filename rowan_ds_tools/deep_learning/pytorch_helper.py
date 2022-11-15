import torch
import numpy as np
from rowan_ds_tools.utils._param_validation import validate_params, Interval

from numbers import Integral
import matplotlib.pyplot as plt


class PytorchTraining:
    @validate_params(
        {
            "model": [torch.nn.Module],
            "train_loader": [torch.utils.data.dataloader.DataLoader],
            "val_loader": [torch.utils.data.dataloader.DataLoader],
            "optimizer": [torch.optim.Optimizer],
            "loss_fn": [torch.nn.modules.loss._Loss, int],
            "device": [torch.device],
        }
    )
    def __init__(
        self, model, train_loader, val_loader, optimizer, loss_fn, device=None
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.loss_fn = loss_fn

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

    @validate_params(
        {
            "epochs": [Interval(Integral, 1, None, closed="left")],
            "path": [str],
            "print_freq": [Interval(Integral, 1, None, closed="left"), None],
        }
    )
    def train(self, epochs, path, print_freq=None):
        """Train the model.

        Args:
            epochs (int): Number of epochs to train for
            path (str): Path to save the model to
            print_freq (str, optional): frequency to print the loss. Defaults to None.
        """
        self.model.to(self.device)
        self.train_losses = []
        self.val_losses = []
        for epoch in range(epochs):
            epoch_train_losses = []
            for batch_idx, (data, targets) in enumerate(self.train_loader):
                data = data.to(self.device)
                targets = targets.to(self.device)

                # forward
                scores = self.model(data)
                loss = self.loss_fn(scores, targets)

                # backward
                self.optimizer.zero_grad()
                loss.backward()

                # gradient descent or adam step
                self.optimizer.step()

                # Record loss
                epoch_train_losses.append(loss.item())

                # Print training loss every print_freq batches
                try:
                    if print_freq and batch_idx % print_freq == 0:
                        print(
                            f"Epoch [{epoch}/{epochs}] Batch {batch_idx}/{len(self.train_loader)} Loss: {loss.item():.4f}"
                        )
                except:
                    pass

            train_loss_for_epoch = np.mean(epoch_train_losses)
            self.train_losses.append(train_loss_for_epoch)
            self.val_losses.append(self.evaluate())
            print(
                f"Epoch [{epoch}/{epochs}] Train loss: {loss.item():.4f} Val loss: {self.val_losses[-1]:.4f}"
            )

        torch.save(self.model.state_dict(), path)

    def plot_losses(self):
        """Plot the training and validation losses."""

        plt.plot(self.train_losses, label="train")
        plt.plot(self.val_losses, label="val")
        plt.title("Training and Validation Losses")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()

    def evaluate(self):
        """Evaluate the model on the validation set."""
        self.model.eval()
        with torch.no_grad():
            losses = []
            for batch_idx, (data, targets) in enumerate(self.val_loader):
                data = data.to(self.device)
                targets = targets.to(self.device)

                scores = self.model(data)
                loss = self.loss_fn(scores, targets)
                losses.append(loss.item())
        self.model.train()
        return np.mean(losses)
