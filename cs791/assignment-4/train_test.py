### Add necessary imports ###
from torch.utils.data import DataLoader
import torch
from models import SimpleNN, CNN

def train_and_test_NN(datasets, hyperparams, seed=42):
    """
    Train and test a Neural Network model
    datasets: tuple of (train_dataset, test_dataset)
    hyperparams: data structure containing hyperparameters like learning rate, epochs, etc.

    Returns:
    accuracy: accuracy on validation dataset
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    train_dataset, validation_dataset = datasets

    ### Implement training loop here ###

    accuracy = None

    ### Implement validation loop here ###

    return accuracy

def train_and_test_CNN(datasets, hyperparams, seed=42):
    """
    Train and test a Convolutional Neural Network model
    datasets: tuple of (train_dataset, test_dataset)
    hyperparams: data structure containing hyperparameters like learning rate, epochs, etc.

    Returns:
    accuracy: accuracy on validation dataset
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    train_dataset, validation_dataset = datasets

    ### Implement training loop here ###

    accuracy = None 

    ### Implement validation loop here ###

    return accuracy
