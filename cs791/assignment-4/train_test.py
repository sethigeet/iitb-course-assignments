import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models import SimpleNN


def train_and_test_NN(datasets, hyperparams):
    """
    Train and test a Neural Network model
    datasets: tuple of (train_dataset, test_dataset)
    hyperparams: data structure containing hyperparameters like learning rate, epochs, etc.

    Returns:
    accuracy: accuracy on validation dataset
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset, validation_dataset = datasets

    batch_size = hyperparams.get("batch_size", 64)
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size, shuffle=False)

    hidden_size = hyperparams.get("hidden_size", 500)
    dropout_rate = hyperparams.get("dropout_rate", 0.5)
    model = SimpleNN(
        input_size=784,
        num_classes=10,
        hidden_size=hidden_size,
        dropout_rate=dropout_rate,
    ).to(device)

    ## ==== Train the model ====

    learning_rate = hyperparams.get("learning_rate", 0.001)
    weight_decay = hyperparams.get("weight_decay", 0.0)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    criterion = nn.CrossEntropyLoss()

    epochs = hyperparams.get("epochs", 10)
    model.train()

    for _ in range(epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)

            loss.backward()
            optimizer.step()

    ## ==== Evaluate the model on the validation dataset ====

    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in validation_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    accuracy = correct / total
    return accuracy


def train_and_test_CNN(datasets, hyperparams):
    """
    Train and test a Convolutional Neural Network model
    datasets: tuple of (train_dataset, test_dataset)
    hyperparams: data structure containing hyperparameters like learning rate, epochs, etc.

    Returns:
    accuracy: accuracy on validation dataset
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_dataset, validation_dataset = datasets

    ### Implement training loop here ###

    accuracy = None

    ### Implement validation loop here ###

    return accuracy
