import copy
import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.amp import autocast, GradScaler

class LRFinder:
    def __init__(
        self,
        model,
        optimizer,
        criterion,
        device=None,
        memory_cache=True,
        cache_dir=None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.history = {"lr": [], "loss": []}
        self.best_loss = None
        self.memory_cache = memory_cache
        self.cache_dir = cache_dir
        self.scaler = GradScaler('cuda')

        # Save the original state of the model and optimizer
        self.model_state = copy.deepcopy(model.state_dict())
        self.optimizer_state = copy.deepcopy(optimizer.state_dict())

        if device:
            self.device = device
        else:
            self.device = next(model.parameters()).device

    def reset(self):
        """Restores the model and optimizer to their initial states."""
        self.model.load_state_dict(self.model_state)
        self.optimizer.load_state_dict(self.optimizer_state)

    def range_test(
        self,
        train_loader,
        start_lr=1e-7,
        end_lr=1,
        num_iter=100,
        step_mode="exp",
        smooth_f=0.05,
        diverge_th=5,
    ):
        """Performs the learning rate range test."""
        # Reset test results
        self.history = {"lr": [], "loss": []}
        self.best_loss = None

        # Move the model to the proper device
        self.model.to(self.device)

        # Set the starting learning rate
        self.optimizer.param_groups[0]["lr"] = start_lr

        # Initialize the learning rate policy
        if step_mode.lower() == "exp":
            lr_schedule = np.exp(np.linspace(np.log(start_lr), np.log(end_lr), num_iter))
        else:
            lr_schedule = np.linspace(start_lr, end_lr, num_iter)

        if smooth_f < 0 or smooth_f >= 1:
            raise ValueError("smooth_f is outside the range [0, 1]")

        # Initialize the data iterator
        iterator = iter(train_loader)
        for iteration in tqdm(range(num_iter)):
            # Get a new set of inputs and labels
            try:
                inputs, labels = next(iterator)
            except StopIteration:
                iterator = iter(train_loader)
                inputs, labels = next(iterator)

            # Move data to the correct device
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            
            # Ensure labels are properly shaped (squeeze any extra dimensions)
            labels = labels.squeeze()

            # Forward pass with mixed precision
            self.optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

            # Backward pass with gradient scaling
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            # Store the values
            self.history["lr"].append(self.optimizer.param_groups[0]["lr"])
            loss_value = loss.item()

            # Update the learning rate
            self.optimizer.param_groups[0]["lr"] = lr_schedule[iteration]

            if iteration == 0:
                self.best_loss = loss_value
                self.history["loss"].append(loss_value)
            else:
                loss_value = smooth_f * loss_value + (1 - smooth_f) * self.history["loss"][-1]
                self.history["loss"].append(loss_value)

                # Check if the loss has diverged
                if loss_value > diverge_th * self.best_loss:
                    print("Stopping early, the loss has diverged")
                    break

        print("Learning rate search finished. See the graph with {finder_name}.plot()")

    def plot(self, skip_start=10, skip_end=5, log_lr=True, suggest=True):
        """Plots the learning rate range test.
        
        Arguments:
            skip_start (int): number of batches to trim from the start.
            skip_end (int): number of batches to trim from the end.
            log_lr (bool): True to plot the learning rate in a logarithmic scale.
            suggest (bool): True to suggest a learning rate based on the loss curve.
        """
        if skip_start < 0:
            skip_start = 0
        if skip_end < 0:
            skip_end = 0

        # Get the data to plot from the history dictionary
        lrs = self.history["lr"]
        losses = self.history["loss"]

        # Plot it
        plt.figure(figsize=(10, 6))
        if log_lr:
            plt.semilogx(lrs[skip_start:len(lrs) - skip_end], 
                        losses[skip_start:len(lrs) - skip_end])
        else:
            plt.plot(lrs[skip_start:len(lrs) - skip_end], 
                    losses[skip_start:len(lrs) - skip_end])

        plt.grid(True, alpha=0.1)
        plt.xlabel("Learning Rate")
        plt.ylabel("Loss")
        
        if suggest:
            # Find the learning rate with minimum loss
            min_grad_idx = (np.gradient(np.array(losses[skip_start:len(lrs) - skip_end]))).argmin()
            suggested_lr = lrs[min_grad_idx + skip_start]
            plt.axvline(x=suggested_lr, color='r', linestyle='--', alpha=0.5)
            plt.text(suggested_lr, plt.ylim()[0], f'Suggested LR: {suggested_lr:.2E}')
            print(f'Suggested Learning Rate: {suggested_lr:.2E}')
        
        plt.title("Learning Rate Range Test")
        plt.savefig('lr_finder_plot.png')
        plt.close()

    def get_suggested_lr(self, skip_start=10, skip_end=5):
        """Returns the suggested learning rate based on the loss curve."""
        lrs = self.history["lr"]
        losses = self.history["loss"]
        
        # Find the learning rate with minimum loss
        min_grad_idx = (np.gradient(np.array(losses[skip_start:len(lrs) - skip_end]))).argmin()
        return lrs[min_grad_idx + skip_start] 