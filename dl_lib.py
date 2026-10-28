from typing import Literal
import torch


def pick_device() -> Literal["cuda", "mps", "cpu"]:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def train(
    model: torch.nn.Module,
    criterion: torch.nn.modules.loss._Loss,
    optimizer: torch.optim.Optimizer,
    data: torch.utils.data.DataLoader,
    device: Literal["cuda", "mps", "cpu"],
) -> tuple[float, torch.nn.Module]:
    model.train()
    train_loss = 0
    for batch in data:
        X, y = batch
        X = X.to(device).float()
        y = y.to(device).float()
        y = y.reshape(-1)
        predictions = model(X)
        loss = criterion(predictions, y.unsqueeze(1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    return (train_loss / len(data)), model


def validate(
    model: torch.nn.Module,
    criterion: torch.nn.modules.loss._Loss,
    data: torch.utils.data.DataLoader,
    device: Literal["cuda", "mps", "cpu"],
) -> float:
    model.eval()
    validation_loss = 0
    with torch.no_grad():
        for batch in data:
            X, y = batch
            X = X.to(device).float()
            y = y.to(device).float()

            y = y.reshape(-1)

            predictions = model(X)
            loss = criterion(predictions, y.unsqueeze(1))
            validation_loss += loss.item()

    return validation_loss / len(data)


def train_model(
    model: torch.nn.Module,
    criterion: torch.nn.modules.loss._Loss,
    optimizer: torch.optim.Optimizer,
    train_data: torch.utils.data.DataLoader,
    validation_data: torch.utils.data.DataLoader,
    epochs: int = 10,
    patience: int = 5,
    device: Literal["cuda", "mps", "cpu"] = "cpu",
) -> tuple[torch.nn.Module, tuple[list[float], list[float]]]:
    model = model.to(device)
    criterion = criterion.to(device)
    best_loss = float("inf")
    epoch_no_improvement = 0
    train_losses = []
    val_losses = []
    for epoch in range(epochs):
        train_loss, model = train(model, criterion, optimizer, train_data, device)
        model = model.to(device)
        validation_loss = validate(model, criterion, validation_data, device)
        print(
            f"Epoch: {epoch+1}\nTraining Loss: {train_loss:.3f}\nValidation Loss: {validation_loss:.3f}"
        )

        # Save Losses:
        train_losses.append(train_loss)
        val_losses.append(validation_loss)
        # Early Stop
        if validation_loss < best_loss:
            best_loss = validation_loss
            epoch_no_improvement = 0
        else:
            epoch_no_improvement += 1

        if epoch_no_improvement > patience:
            break
    return model, (train_losses, val_losses)


def calc_mean_std(
    dataloader: torch.utils.data.DataLoader,
) -> tuple[torch.Tensor | float, torch.Tensor | float]:
    channels_sum, channels_squared_sum, num_batches = 0, 0, 0
    for data, _ in dataloader:
        # Average across Batch (0), Height (2), and Width (3)
        channels_sum += torch.mean(data, dim=(0, 2, 3))
        channels_squared_sum += torch.mean(data**2, dim=(0, 2, 3))
        num_batches += 1

    mean = channels_sum / num_batches
    std = torch.sqrt((channels_squared_sum / num_batches - mean**2))  # type: ignore

    return mean, std


if __name__ == "__main__":
    pass
