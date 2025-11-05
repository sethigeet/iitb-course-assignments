import torch.nn.functional as F
from torch import nn


class SimpleNN(nn.Module):
    def __init__(
        self, input_size=784, hidden_size=500, num_classes=10, dropout_rate=0.5
    ):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        out = self.fc1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out


class CNN(nn.Module):
    def __init__(
        self,
        num_classes=10,
        hidden_size_fc=256,
        dropout_rate_conv=0.25,
        dropout_rate_fc=0.5,
        kernel_size=3,
    ):
        super().__init__()

        # --- Block 1 ---
        # Input: (Batch_Size, 1, 28, 28)
        self.conv1_1 = nn.Conv2d(1, 32, kernel_size=kernel_size, padding="same")
        self.bn1_1 = nn.BatchNorm2d(32)
        self.conv1_2 = nn.Conv2d(32, 32, kernel_size=kernel_size, padding="same")
        self.bn1_2 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.drop1 = nn.Dropout2d(
            p=dropout_rate_conv
        )  # Dropout for 2D spatial features
        # Output: (Batch_Size, 32, 14, 14)

        # --- Block 2 ---
        self.conv2_1 = nn.Conv2d(32, 64, kernel_size=kernel_size, padding="same")
        self.bn2_1 = nn.BatchNorm2d(64)
        self.conv2_2 = nn.Conv2d(64, 64, kernel_size=kernel_size, padding="same")
        self.bn2_2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.drop2 = nn.Dropout2d(p=dropout_rate_conv)
        # Output: (Batch_Size, 64, 7, 7)

        # --- Classifier ---
        # Flattened size: 64 channels * 7 height * 7 width = 3136
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 7 * 7, hidden_size_fc)
        self.bn_fc1 = nn.BatchNorm1d(hidden_size_fc)  # 1D Batch Norm for Dense layers
        self.drop_fc1 = nn.Dropout(p=dropout_rate_fc)
        self.fc2 = nn.Linear(
            hidden_size_fc, num_classes
        )  # Output layer (num_classes classes)

    def forward(self, x):
        # Block 1
        x = F.relu(self.bn1_1(self.conv1_1(x)))
        x = F.relu(self.bn1_2(self.conv1_2(x)))
        x = self.pool1(x)
        x = self.drop1(x)

        # Block 2
        x = F.relu(self.bn2_1(self.conv2_1(x)))
        x = F.relu(self.bn2_2(self.conv2_2(x)))
        x = self.pool2(x)
        x = self.drop2(x)

        # Classifier
        x = self.flatten(x)
        x = F.relu(self.bn_fc1(self.fc1(x)))
        x = self.drop_fc1(x)
        x = self.fc2(x)  # Output raw logits

        return x
